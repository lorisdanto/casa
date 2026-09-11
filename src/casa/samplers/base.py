from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import torch
from casa.llm import LLM
from casa.grammar import Grammar
from casa.algebra import Envelope, Potential

@dataclass
class SamplingResult:
    """Result from a sampling operation.
    
    Attributes:
        tokens: List of token strings.
        token_ids: List of token IDs.
        text: Decoded text.
        raw_logprob: Raw log probability from the model.
        constrained_logprob: Log probability under grammar constraints.
        success: Whether the sample satisfied constraints.
    """
    tokens: List[str]
    token_ids: List[int]
    text: str
    raw_logprob: float
    constrained_logprob: Optional[float] = None
    success: bool = True
    attempts: int = 1


class BaseSampler(ABC):
    """Abstract base class for sampling algorithms.
    
    All samplers should inherit from this class and implement the sample() method.
    """
    
    #: What a sampler may be conditioned on. A grammar restricts one model to a language; an
    #: envelope expression describes a combination of models, of which a constrained single model
    #: is one case. Samplers that only understand grammars simply never see the other kind.
    TARGET_TYPES = (Grammar, Potential, Envelope)

    def __init__(
        self,
        llm,
        target,
        max_new_tokens: int = 512,
    ):
        """Initialize base sampler.

        Args:
            llm: LLM instance.
            target: A ``Grammar``, or an envelope expression from :mod:`casa.algebra`.
            max_new_tokens: Maximum number of tokens to generate.
        """

        if not isinstance(llm, LLM):
            raise TypeError(f"llm must be an LLM instance, got {type(llm)}")
        if not isinstance(target, self.TARGET_TYPES):
            names = ", ".join(t.__name__ for t in self.TARGET_TYPES)
            raise TypeError(f"target must be one of ({names}), got {type(target).__name__}")

        self.llm = llm
        self.target = target
        #: Backwards-compatible alias; ``None`` when the target is not a grammar.
        self.grammar = target if isinstance(target, Grammar) else None
        self.max_new_tokens = max_new_tokens
    
    @abstractmethod
    def sample(
        self,
        prompt: str,
        n_samples: int = 1,
        **kwargs,
    ) -> List[SamplingResult]:
        """Generate samples from the model.
        
        Args:
            prompt: Input prompt text.
            n_samples: Number of samples to generate.
            **kwargs: Additional sampler-specific arguments.
            
        Returns:
            List of sampling results.
        """
        pass
    
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode and format prompt.
        
        Args:
            prompt: Raw prompt string.
            
        Returns:
            Encoded prompt tensor on model device.
        """
        formatted_prompt = self.llm.format_prompt(prompt)
        prompt_ids = self.llm.tokenizer.encode(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.llm.device)
        return prompt_ids