import torch
from dataclasses import dataclass
from typing import List, Optional

from casa.backends.base import InferenceBackend
from casa.utils.oracle_trie import Trie
from casa.utils.profiling import ProfileTimer
from casa.utils.helpers import print_progress

import xgrammar
apply_bitmask = xgrammar.apply_token_bitmask_inplace


@dataclass
class SamplingResult:
	tokens: List[str]
	token_ids: List[int]
	text: str
	raw_logprob: float
	constrained_logprob: float
	success: bool
	n_attempts: int = 1


class CARS:
	def __init__(
		self,
		backend: InferenceBackend,
		grammar,
		max_new_tokens: int = 512,
		verbose: bool = False,
		is_chat_model: bool = True
	):
		self.backend = backend
		self.tokenizer = backend.tokenizer
		self.device = backend.device
		self.grammar = grammar
		self.max_new_tokens = max_new_tokens
		self.verbose = verbose
		self.timer = ProfileTimer()
		self.trie = Trie()
		self.is_chat_model = is_chat_model

	def _encode_prompt(self, prompt: str) -> List[int]:
		if self.is_chat_model:
			messages = [{"role": "user", "content": prompt}]
			formatted = self.tokenizer.apply_chat_template(
				messages, tokenize=False, add_generation_prompt=True,
			)
		else:
			formatted = prompt
		return self.tokenizer.encode(formatted, add_special_tokens=False)

	def sample(
		self,
		prompt: str,
		n_samples: int = 1,
		max_attempts: int = 1000,
	) -> List[SamplingResult]:
		prompt_ids = self._encode_prompt(prompt)
		results = []
		total_attempts = 0

		for sample_idx in range(n_samples):
			n_attempts = 0
			success = False

			for attempt in range(max_attempts):
				n_attempts += 1
				total_attempts += 1

				result = self._generate_one(prompt_ids)

				if result is not None:
					result.n_attempts = n_attempts
					results.append(result)
					print_progress(
						sample_idx + 1, n_samples, n_attempts,
						max_attempts, self.verbose, timeout=False,
					)
					success = True
					break

			if not success:
				print_progress(
					sample_idx + 1, n_samples, n_attempts,
					max_attempts, self.verbose, timeout=True,
				)

		print(f"\n  Total attempts across all samples: {total_attempts}")
		print(f"  Successful samples: {len(results)}")
		if results:
			print(f"  Average attempts per sample: {total_attempts / len(results):.1f}")

		self.timer.report(
			title=f"CARS Profiling ({len(results)} samples, {total_attempts} attempts)"
		)
		return results

	def _generate_one(self, prompt_ids: List[int]) -> Optional[SamplingResult]:
		self.backend.reset_cache() 
		context = []
		oracle_node = self.trie.root
		oracle_depth = 0
		raw_logprobs = []
		recompute_needed = False

		self.grammar.recognizer.reset()

		for step in range(self.max_new_tokens):
			# Step 1: Advance grammar
			if step > 0:
				with self.timer("try_advance"):
					ok = self.grammar.recognizer.try_advance_token_ids(
						torch.tensor(context)
					)
					if not ok:
						self._handle_rejection(
							oracle_node, oracle_depth, context, context[-1]
						)
						return None

					if (self.grammar.recognizer.ll_matcher.is_accepting() and
						self.grammar.recognizer.ll_matcher.is_stopped()):
						if recompute_needed:
							self._recompute_in_trie(oracle_node, oracle_depth, context)
						text = self.tokenizer.decode(context)
						return SamplingResult(
							tokens=[self.tokenizer.decode([t]) for t in context],
							token_ids=context,
							text=text,
							raw_logprob=sum(raw_logprobs),
							constrained_logprob=0.0,
							success=True,
						)

			# Step 2: Navigate trie
			with self.timer("trie_navigate"):
				if step > 0:
					last_token = context[-1]
					if last_token not in oracle_node.children:
						oracle_node.create_child(last_token)
					oracle_node = oracle_node.children[last_token]
					oracle_depth += 1

			# Step 3: Get logprobs (with caching)
			if oracle_node.raw_logprob is not None:
				with self.timer("trie_cache_hit"):
					logps = oracle_node.raw_logprob[0].to(self.device)
				is_new_node = False
			else:
				with self.timer("backend_logprobs"):
					logps_cpu = self.backend.get_next_token_logprobs(
						prompt_ids + context
					)
					logps = logps_cpu.to(self.device)

				with self.timer("store_logprobs"):
					oracle_node.raw_logprob = logps_cpu.unsqueeze(0)
					oracle_node.log_theta = torch.zeros(1, logps_cpu.shape[-1])

				with self.timer("filter_vocab"):
					acceptance = self.grammar.recognizer.filter_vocab()
				with self.timer("apply_bitmask"):
					apply_bitmask(oracle_node.log_theta, acceptance)
				recompute_needed = True
				is_new_node = True

			# Step 4: Sample
			is_root = (step == 0)
			should_adjust = is_root or not is_new_node

			with self.timer("reweight_and_sample"):
				if should_adjust:
					adjusted = logps + oracle_node.log_theta[0].to(self.device)
				else:
					adjusted = logps

				probs = torch.softmax(adjusted, dim=-1)
				if probs.sum() < 1e-10:
					return None

				next_token = torch.multinomial(probs, 1).item()

			raw_logprobs.append(oracle_node.raw_logprob[0, next_token].item())

			# Step 5: Check for EOS and grammar acceptance
			if next_token == self.tokenizer.eos_token_id:
				ok = self.grammar.recognizer.try_advance_token_ids(
					torch.tensor(context + [next_token])
				)
				if ok and self.grammar.recognizer.ll_matcher.is_accepting():
					if recompute_needed:
						self._recompute_in_trie(oracle_node, oracle_depth, context)
					text = self.tokenizer.decode(context)
					return SamplingResult(
						tokens=[self.tokenizer.decode([t]) for t in context],
						token_ids=context,
						text=text,
						raw_logprob=sum(raw_logprobs[:-1]),
						constrained_logprob=0.0,
						success=True,
					)
				else:
					self._handle_rejection(
						oracle_node, oracle_depth, context, next_token
					)
					return None

			context.append(next_token)

		if recompute_needed:
			self._recompute_in_trie(oracle_node, oracle_depth, context)
		return None

	def _handle_rejection(self, oracle_node, oracle_depth, context, failed_token):
		with self.timer("mark_invalid_token"):
			oracle_node.log_theta[0, failed_token] = float('-inf')
		self._recompute_in_trie(oracle_node, oracle_depth, context)

	def _recompute_in_trie(self, node, depth, context):
		with self.timer("recompute_in_trie"):
			while depth > 0:
				new_log_theta = torch.log(
					torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum()
				)
				depth -= 1
				node = node.parent
				node.log_theta[0, context[depth]] = new_log_theta