from abc import ABC, abstractmethod
from typing import List
import torch

class InferenceBackend(ABC):
    @abstractmethod
    def get_next_token_logprobs(self, token_ids: List[int]) -> torch.Tensor:
        """Return log-probabilities over full vocabulary for the next token.
        
        Args:
            token_ids: Full sequence (prompt + generated tokens so far).
            
        Returns:
            Tensor of shape (vocab_size,) with log-probabilities on CPU.
        """
        ...
    
    @property
    @abstractmethod
    def tokenizer(self):
        """Return the tokenizer."""
        ...
    
    @property
    @abstractmethod  
    def device(self) -> torch.device:
        """Return the compute device."""
        ...
    
    def shutdown(self):
        """Optional cleanup."""
        pass
    
    def reset_cache(self):
        """Reset any cached state. Called at the start of each attempt."""
        pass