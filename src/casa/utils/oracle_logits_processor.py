import torch
from typing import Optional
from transformers.generation.logits_process import LogitsProcessor

from casa.utils.oracle_trie import Trie
from casa.utils.profiling import ProfileTimer

import xgrammar
apply_bitmask = xgrammar.apply_token_bitmask_inplace


class OracleLogitsProcessor(LogitsProcessor):
    """Logits processor that enforces grammar constraints using an oracle trie.
    
    This processor maintains a trie structure to track previously sampled sequences
    and their probabilities, enabling adaptive rejection sampling with grammar constraints.
    
    Args:
        tokenizer: The tokenizer associated with the model.
        grammar_constraint: Grammar constraint object from xgrammar.
        device: Device to run computations on.
        learn_level: Learning level (1-3) controlling constraint application.
        constrain_first: Whether to constrain the first token.
    """
    
    def __init__(
        self,
        tokenizer,
        grammar_constraint,
        device: torch.device,
        learn_level: int = 3,
        constrain_first: bool = False
    ):
        self.tokenizer = tokenizer
        self.grammar_constraint = grammar_constraint
        self.learn_level = learn_level
        self.constrain_first = constrain_first
        self.device = device
        
        self.oracle_trie = Trie()
        self.current_index: Optional[int] = None
        self.timer = ProfileTimer()
        self.reset()
    
    def reset(self) -> None:
        """Reset the processor state for a new generation."""
        self.grammar_constraint.reset()
        self.generate_start_index: Optional[int] = None
        self.generated_tokens: Optional[torch.Tensor] = None
        self.oracle_node = self.oracle_trie.root
        self.oracle_node_depth = 0
        self.recompute_needed = False 
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Process logits to enforce grammar constraints.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            scores: Raw logits of shape (batch_size, vocab_size).
            
        Returns:
            Adjusted logits with grammar constraints applied.
        """        
        
        with self.timer("set_generated_tokens"):
            self._set_generated_tokens(input_ids)
        is_root = len(self.generated_tokens) == 0

        if self.learn_level != 1:
            with self.timer("try_advance_token_ids"):
                if not self.grammar_constraint.try_advance_token_ids(self.generated_tokens):
                    self._generation_failed()
        
        with self.timer("trie_navigate"):
            if not is_root:
                assert len(self.generated_tokens) == self.oracle_node_depth + 1
                last_token = self.generated_tokens[-1].item()
                if last_token not in self.oracle_node.children:
                    self.oracle_node.create_child(last_token)
                self.oracle_node = self.oracle_node.children[last_token]
                self.oracle_node_depth += 1
        
        if self.oracle_node.raw_logprob is None:
            with self.timer("log_softmax_and_cpu_copy"):
                self.oracle_node.raw_logprob = torch.log_softmax(scores, dim=-1).cpu()
            self.oracle_node.log_theta = torch.zeros(1, scores.size(1))
            
            adjust_scores = is_root and self.constrain_first
            if self.learn_level >= 3 or adjust_scores:
                with self.timer("filter_vocab"):
                    acceptance = self.grammar_constraint.filter_vocab()
                with self.timer("apply_bitmask"):
                    apply_bitmask(self.oracle_node.log_theta, acceptance)
                self.recompute_needed = True
        else:
            adjust_scores = True
        
        if adjust_scores:
            with self.timer("apply_log_theta_to_scores"):
                scores = scores.clone()
                scores += self.oracle_node.log_theta.to(self.device, non_blocking=True)
        
        return scores
    
    def _set_generated_tokens(self, input_ids: torch.LongTensor) -> None:
        """Extract and store the generated tokens from input_ids."""
        assert len(input_ids) == 1, "Batch size must be 1"
        
        if self.generate_start_index is None:
            self.generate_start_index = input_ids.size(1)
            
        self.generated_tokens = input_ids[0, self.generate_start_index:]
    
    def _generation_failed(self) -> None:
        """Handle generation failure by updating the trie."""
        assert len(self.generated_tokens) == self.oracle_node_depth + 1
        
        if self.learn_level >= 1:
            with self.timer("mark_invalid_token"):
                self.oracle_node.log_theta[0, self.generated_tokens[-1]] = -float('inf')
            self._recompute_in_trie()
            
        raise ValueError(f"Generation failed at tokens: {self.generated_tokens}")
    
    def _recompute_in_trie(self) -> None:
        """Recompute log_theta values up the trie after a change."""
        with self.timer("recompute_in_trie"):
            node = self.oracle_node
            depth = self.oracle_node_depth
            while depth > 0:
                new_log_theta = torch.log(
                    torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum()
                )
                depth -= 1
                node = node.parent
                node.log_theta[0, self.generated_tokens[depth]] = new_log_theta
    
    def generation_ended(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Finalize generation and return the log probability.
        
        Args:
            input_ids: Final input token IDs.
            
        Returns:
            Log probability of the generated sequence.
            
        Raises:
            ValueError: If the generated sequence violates grammar constraints.
        """
        with self.timer("generation_ended_logic"):
            self._set_generated_tokens(input_ids)
            assert len(self.generated_tokens) == self.oracle_node_depth + 1
            
            if not self.grammar_constraint.try_advance_token_ids(self.generated_tokens):
                self._generation_failed()
            
            if self.generated_tokens[-1] != self.tokenizer.eos_token_id:
                if not self.grammar_constraint.ll_matcher.is_accepting():
                    self._generation_failed()
        
        if self.recompute_needed:
            self._recompute_in_trie()
        
        return self.get_logprob()
    
    def get_logprob(self) -> torch.Tensor:
        """Calculate the total log probability of the generated sequence.
        
        Returns:
            Sum of log probabilities for all generated tokens.
        """
        assert len(self.generated_tokens) == self.oracle_node_depth + 1
        
        logprobs = []
        node = self.oracle_node
        depth = self.oracle_node_depth
        
        while depth >= 0:
            logprobs.append(node.raw_logprob[0, self.generated_tokens[depth]])
            depth -= 1
            node = node.parent
            
        return torch.tensor(logprobs).flip(0).sum()