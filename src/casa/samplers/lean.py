import torch
from typing import Optional
from dataclasses import dataclass

from casa.utils.oracle_trie import Trie, TrieNode


@dataclass
class CheckResult:
	is_valid: bool
	is_complete: bool
	shortest_invalid_prefix: Optional[str]


class LeanCARS:    
	def __init__(self, llm, max_new_tokens: int = 512, verbose: bool = True):
		self.llm = llm
		self.max_new_tokens = max_new_tokens
		self.trie = Trie()
		self.verbose = verbose
	
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
	
	"""TODO: When the first token of the shortest invalid prefix is a whitespace,
	         it becomes a bit tricky to handle. We could block a whitespace and its non-whitespace suffixes,
             but the sampler just samples a different whitespace character (e.g., tab) and repeats the same mistake.
             For now, we ignore this issue. But ideally, the first token of an invalid prefix should never be a whitespace?
 	"""
	def update_from_invalid_prefix(self, invalid_prefix: str, generated_tokens: list[int]):
		# self._log(f"[update] Invalid prefix: {repr(invalid_prefix)}")
		# self._log(f"[update] Generated {len(generated_tokens)} tokens")
		
		invalid_prefix_stripped = invalid_prefix.strip()
		if not invalid_prefix_stripped:
			return
		
		node = self.trie.root
		
		for i, token in enumerate(generated_tokens):
			if node.log_theta is None:
				return
			
			decoded_token = self.llm.tokenizer.decode([token])
			decoded_stripped = decoded_token.strip()
			
			if decoded_stripped and (decoded_stripped.startswith(invalid_prefix_stripped) or 
				invalid_prefix_stripped.startswith(decoded_stripped)):
				# self._log(f"[update] Blocking at depth {i}")
				
				# blocked = 0
				for token_id in range(len(self.llm.tokenizer)):
					decoded = self.llm.tokenizer.decode([token_id]).strip()
					if decoded and (decoded.startswith(invalid_prefix_stripped) or 
								invalid_prefix_stripped.startswith(decoded)):
						node.log_theta[0, token_id] = float('-inf')
						# blocked += 1
				
				# self._log(f"[update] Blocked {blocked} tokens at depth {i}")
				self._propagate_up(node, generated_tokens[:i])
				return
			
			if token not in node.children:
				# self._log(f"[update] Token {token} not in children, stopping")
				return
			node = node.children[token]
		
		# self._log(f"[update] Could not find matching token")
	
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

	"""TODO: The Lean proof checker functions as our oracle. Any prefix and its continuations we block,
		     are dependent on the shortest invalid prefix it returns. While checking incrementally at
             token level, sometimes an incomplete tactic produces an invalid prefix that should 
             not be blocked ideally (it should return Optional[None] instead). We need to refine
			 the checker to avoid such invalid prefixes. For now, we try to check at tactic boundaries
			 (newline or semicolon) to reduce such cases. 
    
			 We also need to profile the sampler and checker in loop to see which part acts as the bottleneck.
   
	"""
	def generate_and_check(
		self, 
		prompt: str, 
		check_fn
	) -> tuple[Optional[str], Optional[str], Optional[list[int]]]:
		prompt_ids = self.llm.encode(prompt).to(self.llm.device)
		generated_tokens = []
		node = self.trie.root
		
        # TODO: A lot of the EOS handling, and incremental checking needs to be cleaned
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
				
				# Decode current text
				current_text = self.llm.tokenizer.decode(generated_tokens)
				for eos in ['<|end▁of▁sentence|>', '</s>', '<|endoftext|>']:
					current_text = current_text.replace(eos, '')
				
				# Check at EOS
				if next_token == self.llm.tokenizer.eos_token_id:
					# self._log(f"[generate] EOS at step {step}")
					current_text = current_text.strip()
					result = check_fn(current_text)
					if result and result.is_complete:
						return (current_text, None, None)
					elif result and not result.is_valid:
						return (None, result.shortest_invalid_prefix, generated_tokens)
					return (None, None, None)
				
				# TODO: Remove this check after fixing the checker (or) define better boundaries
				last_char = current_text[-1] if current_text else ''
				if last_char in ['\n', ';']:
					current_text = current_text.strip()
					if len(current_text) > 0:
						# self._log(f"[generate] Boundary at step {step}: {repr(current_text[:50])}")
						result = check_fn(current_text)
						
						if result is None:
							continue
						
						if result.is_complete:
							# self._log(f"[generate] Valid at step {step}")
							return (current_text, None, None)
						
						if not result.is_valid:
							# self._log(f"[generate] Invalid at step {step}")
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
	) -> list[str]:
		results = []
		
		for sample_idx in range(n_samples):
			self._log(f"\n[sample] === Sample {sample_idx + 1}/{n_samples} ===")
			
			for attempt in range(max_attempts):
				self._log(f"[sample] Attempt {attempt + 1}")
				
				valid_proof, invalid_prefix, generated_tokens = self.generate_and_check(prompt, check_fn)
				
				if valid_proof:
					self._log(f"[sample] Found valid proof")
					results.append(valid_proof)
					break
				
				if invalid_prefix and generated_tokens:
					self._log(f"[sample] Learning prefix: {repr(invalid_prefix[:30])}")
					self.update_from_invalid_prefix(invalid_prefix, generated_tokens)

		return results