import torch
from typing import Optional
from dataclasses import dataclass, field

from casa.utils.oracle_trie import Trie, TrieNode


@dataclass
class CheckResult:
	is_valid: bool
	is_complete: bool
	shortest_invalid_prefix: Optional[str]


@dataclass
class PrefixLearnEvent:
	invalid_prefix: str
	blocked_token_str: str
	blocked_token_id: int
	blocked_token_raw_prob: float
	total_blocked_mass_at_node: float
	propagated_mass_at_root: float
	block_depth: int
	attempt_number: int


@dataclass 
class ARSStats:
	prefix_events: list[PrefixLearnEvent] = field(default_factory=list)
	
	@property
	def total_root_mass_blocked(self) -> float:
		"""Total probability mass blocked at root level across all learned prefixes."""
		return sum(e.propagated_mass_at_root for e in self.prefix_events)
	
	@property
	def total_prefixes_learned(self) -> int:
		return len(self.prefix_events)


@dataclass
class SampleResult:
    proofs: list[str]
    attempts_used: int
    invalid_prefixes: list[str]
    stats: ARSStats = field(default_factory=ARSStats)


class LeanARS:    
	def __init__(self, llm, max_new_tokens: int = 512, verbose: bool = True, learn: bool = True):
		self.llm = llm
		self.max_new_tokens = max_new_tokens
		self.trie = Trie()
		self.verbose = verbose
		self.learn = learn
		self.stats = ARSStats()
		self._current_attempt = 0
	
	def _log(self, msg: str):
		if self.verbose:
			print(msg, flush=True)
		
	def generate_partial_proof(self, prompt: str) -> tuple[str, list[torch.Tensor]]:
		prompt_ids = self.llm.encode(prompt).to(self.llm.device)
		
		generated_tokens = []
		token_logprobs = []
		node = self.trie.root
		
		with torch.no_grad():
			for step in range(self.max_new_tokens):
				if step % 10 == 0:
					self._log(f"[generate] Step {step}/{self.max_new_tokens}")
				
				input_ids = torch.cat([
					prompt_ids, 
					torch.tensor([generated_tokens], device=self.llm.device)
				], dim=1) if generated_tokens else prompt_ids
				
				outputs = self.llm.model(input_ids)
				logits = outputs.logits[0, -1, :]
				raw_logprobs = torch.log_softmax(logits, dim=-1)
				
				if generated_tokens:
					last_token = generated_tokens[-1]
					if last_token not in node.children:
						node.create_child(last_token)
					node = node.children[last_token]
				
				if node.raw_logprob is None:
					node.raw_logprob = raw_logprobs.cpu().unsqueeze(0)
					node.log_theta = torch.zeros(1, raw_logprobs.size(0))
				
				adjusted_logprobs = raw_logprobs + node.log_theta[0].to(self.llm.device)
				
				probs = torch.softmax(adjusted_logprobs, dim=-1)
				
				if probs.sum() < 1e-10:
					break
				
				next_token = torch.multinomial(probs, 1).item()
				
				generated_tokens.append(next_token)
				token_logprobs.append(raw_logprobs.cpu())
				
				if next_token == self.llm.tokenizer.eos_token_id:
					break
		
		generated_text = self.llm.tokenizer.decode(generated_tokens)
		self._log(f"[generate] Text: {repr(generated_text[:238])}...")
		
		return generated_text, token_logprobs
	
	def update_from_invalid_prefix(self, invalid_prefix: str, generated_tokens: list[int]):
		"""Block at the point where the invalid prefix ends."""
		if not invalid_prefix or len(generated_tokens) == 0:
			return
		
		self._log(f"[update] Invalid prefix: {repr(invalid_prefix[:50])}")
		
		text = ""
		split_idx = 0
		for i, token in enumerate(generated_tokens):
			text = self.llm.tokenizer.decode(generated_tokens[:i+1])
			if len(text.strip()) >= len(invalid_prefix.strip()):
				split_idx = i
				break
		
		self._log(f"[update] Invalid at token {split_idx}")
		
		node = self.trie.root
		for token in generated_tokens[:split_idx]:
			if token not in node.children:
				return
			node = node.children[token]
		
		if node.log_theta is None:
			return
		
		token_to_block = generated_tokens[split_idx]
		
		# Measure mass before blocking
		raw_prob_of_blocked = torch.exp(node.raw_logprob[0, token_to_block]).item() if node.raw_logprob is not None else 0.0
		
		# Total mass already blocked at this node (before this update)
		blocked_mask_before = torch.isinf(node.log_theta[0]) & (node.log_theta[0] < 0)
		total_blocked_before = torch.exp(node.raw_logprob[0][blocked_mask_before]).sum().item() if node.raw_logprob is not None else 0.0
		
		# Root-level reachable mass before
		root_mass_before = self._get_root_reachable_mass()
		
		# Block
		node.log_theta[0, token_to_block] = float('-inf')
		self._log(f"[update] Blocked token {token_to_block} = {repr(self.llm.tokenizer.decode([token_to_block]))}")
		
		self._propagate_up(node, generated_tokens[:split_idx])
		
		# Measure
		total_blocked_after = total_blocked_before + raw_prob_of_blocked
		root_mass_after = self._get_root_reachable_mass()
		root_mass_removed = root_mass_before - root_mass_after
		
		event = PrefixLearnEvent(
			invalid_prefix=invalid_prefix[:80],
			blocked_token_str=repr(self.llm.tokenizer.decode([token_to_block])),
			blocked_token_id=token_to_block,
			blocked_token_raw_prob=raw_prob_of_blocked,
			total_blocked_mass_at_node=total_blocked_after,
			propagated_mass_at_root=max(root_mass_removed, 0.0),
			block_depth=split_idx,
			attempt_number=self._current_attempt,
		)
		self.stats.prefix_events.append(event)
		
		self._log(
			f"[update] Blocked raw_prob={raw_prob_of_blocked:.6f}, "
			f"total_blocked_at_node={total_blocked_after:.6f}, "
			f"root_mass_removed={root_mass_removed:.6f}"
		)
		
	def _get_root_reachable_mass(self) -> float:
		root = self.trie.root
		if root.raw_logprob is None or root.log_theta is None:
			return 1.0
		reachable = torch.exp(root.raw_logprob[0] + root.log_theta[0])
		return reachable.sum().item()
		
	def _propagate_up(self, node: TrieNode, tokens: list[int]):
		for i in range(len(tokens) - 1, -1, -1):
			if node.raw_logprob is None or node.log_theta is None:
				break
				
			p_u = torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum()
			new_log_theta = torch.log(p_u) if p_u > 0 else torch.tensor(float('-inf'))
			
			node = node.parent
			if node is None or node.log_theta is None:
				break
			node.log_theta[0, tokens[i]] = new_log_theta

	def generate_and_check(
		self, 
		prompt: str, 
		check_fn
	) -> tuple[Optional[str], Optional[str], Optional[list[int]]]:
		prompt_ids = self.llm.encode(prompt).to(self.llm.device)
		generated_tokens = []
		node = self.trie.root
		
		with torch.no_grad():
			for step in range(self.max_new_tokens):
				if step % 20 == 0:
					self._log(f"[generate] Step {step}/{self.max_new_tokens}")
				
				input_ids = torch.cat([
					prompt_ids, 
					torch.tensor([generated_tokens], device=self.llm.device)
				], dim=1) if generated_tokens else prompt_ids
				
				outputs = self.llm.model(input_ids)
				logits = outputs.logits[0, -1, :]
				raw_logprobs = torch.log_softmax(logits, dim=-1)
				
				if node.raw_logprob is None:
					node.raw_logprob = raw_logprobs.cpu().unsqueeze(0)
					node.log_theta = torch.zeros(1, raw_logprobs.size(0))
				
				adjusted_logprobs = raw_logprobs + node.log_theta[0].to(self.llm.device)
				probs = torch.softmax(adjusted_logprobs, dim=-1)
				
				if probs.sum() < 1e-10:
					return (None, None, None)
				
				next_token = torch.multinomial(probs, 1).item()
				generated_tokens.append(next_token)
				
				if next_token not in node.children:
					node.create_child(next_token)
				node = node.children[next_token]
				
				current_text = self.llm.tokenizer.decode(generated_tokens)
				for eos in ['<|end▁of▁sentence|>', '</s>', '<|endoftext|>']:
					current_text = current_text.replace(eos, '')
				
				# Check at EOS
				if next_token == self.llm.tokenizer.eos_token_id:
					current_text = current_text.strip()
					result = check_fn(current_text)
					if result and result.is_complete:
						return (current_text, None, None)
					elif result and not result.is_valid:
						return (None, result.shortest_invalid_prefix, generated_tokens)
					return (None, None, None)
				
				last_char = current_text[-1] if current_text else ''
				if last_char in ['\n', ';']:
					current_text = current_text.strip()
					if len(current_text) > 0:
						result = check_fn(current_text)
						
						if result is None:
							continue
						
						if result.is_complete:
							return (current_text, None, None)
						
						if result.is_valid and not result.is_complete:
							continue
						
						if not result.is_valid:
							return (None, result.shortest_invalid_prefix, generated_tokens)
		
		# Max tokens - final check
		current_text = self.llm.tokenizer.decode(generated_tokens)
		for eos in ['<|end▁of▁sentence|>', '</s>', '<|endoftext|>']:
			current_text = current_text.replace(eos, '')
		current_text = current_text.strip()
		
		result = check_fn(current_text)
		if result and result.is_complete:
			return (current_text, None, None)
		elif result and not result.is_valid:
			return (None, result.shortest_invalid_prefix, generated_tokens)
		
		return (None, None, None)


	def sample(
		self,
		prompt: str,
		check_fn,
		n_samples: int = 1,
		max_attempts: int = 100,
	) -> SampleResult:
		results = []
		total_attempts = 0
		invalid_prefixes_found = []
		self.stats = ARSStats()  # Reset for this sample call
		
		for sample_idx in range(n_samples):
			self._log(f"\n[sample] --- Sample {sample_idx + 1}/{n_samples} ---")
			
			for attempt in range(max_attempts):
				total_attempts += 1
				self._current_attempt = total_attempts
				self._log(f"[sample] Attempt {attempt + 1}")
				
				valid_proof, invalid_prefix, generated_tokens = \
					self.generate_and_check(prompt, check_fn)
				
				if valid_proof:
					self._log("[sample] Found valid proof")
					results.append(valid_proof)
					break
				
				if invalid_prefix and generated_tokens:
					invalid_prefixes_found.append(invalid_prefix)
					if self.learn:
						self._log(f"[sample] Learning prefix: {repr(invalid_prefix[:30])}")
						self.update_from_invalid_prefix(invalid_prefix, generated_tokens)

		return SampleResult(
			proofs=results,
			attempts_used=total_attempts,
			invalid_prefixes=invalid_prefixes_found,
			stats=self.stats,
		)