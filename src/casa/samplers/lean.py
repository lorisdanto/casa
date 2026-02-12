from typing import Callable as Fn, Optional

import torch
import math
from dataclasses import dataclass

from casa.utils.oracle_trie import Trie, TrieNode


@dataclass
class CheckResult:
    is_valid: bool
    is_complete: bool
    shortest_invalid_prefix: Optional[str]


@dataclass
class AnnotatedInvalidPrefix:
    prefix: str
    raw_probibility_mass: float
    probability_mass_pruned: float


@dataclass
class SampleResult:
    proofs: list[str]
    attempts_used: int
    invalid_prefixes: list[AnnotatedInvalidPrefix]


def total_pruned_mass(annotated_prefixes: list[AnnotatedInvalidPrefix]) -> float:
    total_mass = 0.0
    for prefix in annotated_prefixes:
        total_mass += prefix.probability_mass_pruned
    return total_mass


@dataclass
class GenerateAndCheckResult:
    valid_proof: Optional[str]
    invalid_prefix: Optional[str]
    generated_tokens: list[int]


class LeanARS:
    def __init__(
        self, llm, max_new_tokens: int = 512, verbose: bool = True, learn: bool = True
    ):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.trie = Trie()
        self.verbose = verbose
        self.learn = learn

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

                input_ids = (
                    torch.cat(
                        [
                            prompt_ids,
                            torch.tensor([generated_tokens], device=self.llm.device),
                        ],
                        dim=1,
                    )
                    if generated_tokens
                    else prompt_ids
                )

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

    def update_from_generated_tokens(
        self, generated_tokens: list[int]
    ) -> Optional[AnnotatedInvalidPrefix]:
        """
        When an entire attempted proof is incorrect, one can block the entire sequence.
        The sequence should end with an EOS token.
        """
        assert 0 < len(generated_tokens)
        if generated_tokens[-1] != self.llm.tokenizer.eos_token_id:
            return None

        node = self.trie.root
        raw_logprob_to_node = 0.0
        adjusted_logprob_to_node = 0.0
        for token in generated_tokens[:-1]:
            raw_logprob_to_node += node.raw_logprob[0, token].item()
            adjusted_logprob_to_node += (
                node.raw_logprob[0, token].item() + node.log_theta[0, token].item()
            )
            node = node.children[token]

        if node.log_theta is None or node.raw_logprob is None:
            return None

        token_to_block = generated_tokens[-1]
        node_total_logprob = (
            raw_logprob_to_node + node.raw_logprob[0, token_to_block].item()
        )
        adjusted_total_logprob = (
            adjusted_logprob_to_node
            + node.raw_logprob[0, token_to_block].item()
            + node.log_theta[0, token_to_block].item()
        )

        node_total_prob = math.exp(node_total_logprob)
        node_adjusted_prob = math.exp(adjusted_total_logprob)
        node.log_theta[0, token_to_block] = float("-inf")
        self._log(
            f"[update] Blocked token {token_to_block} = {repr(self.llm.tokenizer.decode([token_to_block]))}"
        )
        self._propagate_up(node, generated_tokens[:-1])
        return AnnotatedInvalidPrefix(
            prefix=self.llm.tokenizer.decode(generated_tokens),
            raw_probibility_mass=node_total_prob,
            probability_mass_pruned=node_adjusted_prob,
        )

    def update_from_invalid_prefix(
        self, invalid_prefix: str, generated_tokens: list[int]
    ) -> Optional[AnnotatedInvalidPrefix]:
        """
        Block at the point where the invalid prefix ends.
        Returns the logprob of the blocked token before blocking if a token was blocked.
        """
        if not invalid_prefix or len(generated_tokens) == 0:
            return

        self._log(f"[update] Invalid prefix: {repr(invalid_prefix[:50])}")

        text = ""
        split_idx = 0
        for i, token in enumerate(generated_tokens):
            text = self.llm.tokenizer.decode(generated_tokens[: i + 1])
            if len(text.strip()) >= len(invalid_prefix.strip()):
                split_idx = i
                break

        self._log(f"[update] Invalid at token {split_idx}")

        node = self.trie.root
        raw_logprob_to_node = 0.0
        adjusted_logprob_to_node = 0.0
        for token in generated_tokens[:split_idx]:
            if token not in node.children:
                return None
            probs = torch.exp(node.raw_logprob[0])
            prob_sum = probs.sum().item()
            prob_idx_max = probs.argmax().item()
            print(
                f"Token: {token}, Prob sum: {prob_sum:.4f}, Max prob token: {prob_idx_max} ({probs[prob_idx_max]:.4f}) ({repr(self.llm.tokenizer.decode([prob_idx_max]))})"
            )
            raw_logprob_to_node += node.raw_logprob[0, token].item()
            decoded_token = self.llm.tokenizer.decode([token])
            print(
                f"Token: {token} ({repr(decoded_token)}), Raw logprob: {node.raw_logprob[0, token].item():.4f}, Log theta: {node.log_theta[0, token].item():.4f}"
            )
            adjusted_logprob_to_node += (
                node.raw_logprob[0, token] + node.log_theta[0, token]
            ).item()
            node = node.children[token]

        if node.log_theta is None:
            return None

        if node.raw_logprob is None:
            return None

        token_to_block = generated_tokens[split_idx]
        node_total_logprob = (
            raw_logprob_to_node + node.raw_logprob[0, token_to_block].item()
        )
        adjusted_total_logprob = (
            adjusted_logprob_to_node
            + node.raw_logprob[0, token_to_block].item()
            + node.log_theta[0, token_to_block].item()
        )

        node_total_prob = math.exp(node_total_logprob)
        node_adjusted_prob = math.exp(adjusted_total_logprob)

        node.log_theta[0, token_to_block] = float("-inf")
        self._log(
            f"[update] Blocked token {token_to_block} = {repr(self.llm.tokenizer.decode([token_to_block]))}"
        )
        self._propagate_up(node, generated_tokens[:split_idx])
        return AnnotatedInvalidPrefix(
            prefix=invalid_prefix,
            raw_probibility_mass=node_total_prob,
            probability_mass_pruned=node_adjusted_prob,
        )

    def _propagate_up(self, node: TrieNode, tokens: list[int]):
        for i in range(len(tokens) - 1, -1, -1):
            if node.raw_logprob is None or node.log_theta is None:
                break

            p_u = torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum()
            new_log_theta = torch.log(p_u) if p_u > 0 else torch.tensor(float("-inf"))

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
        self, prompt: str, check_fn: Fn[[str], Optional[CheckResult]]
    ) -> GenerateAndCheckResult:
        prompt_ids = self.llm.encode(prompt).to(self.llm.device)
        generated_tokens = []
        node = self.trie.root

        # TODO: A lot of the EOS handling, and incremental checking needs to be cleaned
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                if step % 20 == 0:
                    self._log(
                        f"[generate] Step {step}/{self.max_new_tokens}: {repr(self.llm.tokenizer.decode(generated_tokens))}..."
                    )

                input_ids = (
                    torch.cat(
                        [
                            prompt_ids,
                            torch.tensor([generated_tokens], device=self.llm.device),
                        ],
                        dim=1,
                    )
                    if generated_tokens
                    else prompt_ids
                )

                outputs = self.llm.model(input_ids)
                logits = outputs.logits[0, -1, :]
                raw_logprobs = torch.log_softmax(logits, dim=-1)

                if node.raw_logprob is None:
                    node.raw_logprob = raw_logprobs.cpu().unsqueeze(0)
                    node.log_theta = torch.zeros(1, raw_logprobs.size(0))

                adjusted_logprobs = raw_logprobs + node.log_theta[0].to(self.llm.device)
                probs = torch.softmax(adjusted_logprobs, dim=-1)

                # if probs.sum() < 1e-10:
                #     return (None, None, None)

                next_token = torch.multinomial(probs, 1).item()
                generated_tokens.append(next_token)

                if next_token not in node.children:
                    node.create_child(next_token)
                node = node.children[next_token]

                current_text = self.llm.tokenizer.decode(generated_tokens)
                for eos in ["<|end▁of▁sentence|>", "</s>", "<|endoftext|>"]:
                    current_text = current_text.replace(eos, "")

                # Check at EOS
                if next_token == self.llm.tokenizer.eos_token_id:
                    print("HIT EOS!!!!")
                    # self._log(f"[generate] EOS at step {step}")
                    current_text = current_text.strip()
                    result = check_fn(current_text)
                    if result is None:
                        raise ValueError(f"Checker didn't run")
                    if result.is_valid:
                        return GenerateAndCheckResult(
                            valid_proof=current_text,
                            invalid_prefix=None,
                            generated_tokens=generated_tokens,
                        )
                    elif result and result.shortest_invalid_prefix is not None:
                        return GenerateAndCheckResult(
                            valid_proof=None,
                            invalid_prefix=result.shortest_invalid_prefix,
                            generated_tokens=generated_tokens,
                        )
                    else:
                        return GenerateAndCheckResult(
                            valid_proof=None,
                            invalid_prefix=None,
                            generated_tokens=generated_tokens,
                        )

                # TODO: Remove this check after fixing the checker (or) define better boundaries
                last_char = current_text[-1] if current_text else ""
                if last_char in ["\n", ";"]:
                    current_text = current_text.strip()
                    if len(current_text) > 0:
                        # self._log(f"[generate] Boundary at step {step}: {repr(current_text[:50])}")
                        result = check_fn(current_text)
                        if result is None:
                            raise ValueError(f"Checker didn't run")

                        if result.is_valid:
                            return GenerateAndCheckResult(
                                valid_proof=current_text,
                                invalid_prefix=None,
                                generated_tokens=generated_tokens,
                            )
                        if result.shortest_invalid_prefix is not None:
                            return GenerateAndCheckResult(
                                valid_proof=None,
                                invalid_prefix=result.shortest_invalid_prefix,
                                generated_tokens=generated_tokens,
                            )

        # Max tokens - final check
        current_text = self.llm.tokenizer.decode(generated_tokens)
        for eos in ["<|end▁of▁sentence|>", "</s>", "<|endoftext|>"]:
            current_text = current_text.replace(eos, "")
        current_text = current_text.strip()

        result = check_fn(current_text)
        if result is None:
            raise ValueError(f"Checker didn't run")
        elif result.is_valid:
            return GenerateAndCheckResult(
                valid_proof=current_text,
                invalid_prefix=None,
                generated_tokens=generated_tokens,
            )
        elif result.shortest_invalid_prefix is not None:
            return GenerateAndCheckResult(
                valid_proof=None,
                invalid_prefix=result.shortest_invalid_prefix,
                generated_tokens=generated_tokens,
            )
        else:
            return GenerateAndCheckResult(
                valid_proof=None, invalid_prefix=None, generated_tokens=generated_tokens
            )

    def sample(
        self,
        prompt: str,
        check_fn: Fn[[str], Optional[CheckResult]],
        n_samples: int = 1,
        max_attempts: int = 100,
    ) -> SampleResult:
        results: list[str] = []
        total_attempts = 0
        invalid_prefixes_found: list[str] = []
        annotated_invalid_prefixes: list[AnnotatedInvalidPrefix] = []

        for sample_idx in range(n_samples):
            self._log(f"\n[sample] --- Sample {sample_idx + 1}/{n_samples} ---")

            for attempt in range(max_attempts):
                total_attempts += 1
                self._log(f"[sample] Attempt {attempt + 1}")

                result = self.generate_and_check(prompt, check_fn)

                self._log(
                    f"[sample] Generated:\n{self.llm.tokenizer.decode(result.generated_tokens)}"
                )

                if result.valid_proof:
                    self._log("[sample] Found valid proof")
                    results.append(result.valid_proof)
                    break

                if result.invalid_prefix:
                    invalid_prefixes_found.append(result.invalid_prefix)
                    if self.learn:
                        self._log(
                            f"[sample] Learning prefix: {repr(result.invalid_prefix)}"
                        )
                        maybe_annotated_prefix = self.update_from_invalid_prefix(
                            result.invalid_prefix, result.generated_tokens
                        )
                        if maybe_annotated_prefix:
                            annotated_invalid_prefixes.append(maybe_annotated_prefix)
                            self._log(
                                f"[sample] Annotated invalid prefix: {maybe_annotated_prefix}"
                            )
                            self._log(
                                f"[sample] Total pruned mass so far: {total_pruned_mass(annotated_invalid_prefixes)}"
                            )
                else:
                    if self.learn:
                        maybe_annotated_prefix = self.update_from_generated_tokens(
                            result.generated_tokens
                        )
                        if maybe_annotated_prefix:
                            annotated_invalid_prefixes.append(maybe_annotated_prefix)
                            self._log(
                                f"[sample] Annotated invalid prefix from full proof: {maybe_annotated_prefix}"
                            )
                            self._log(
                                f"[sample] Total pruned mass so far: {total_pruned_mass(annotated_invalid_prefixes)}"
                            )

        return SampleResult(
            proofs=results,
            attempts_used=total_attempts,
            invalid_prefixes=annotated_invalid_prefixes,
        )
