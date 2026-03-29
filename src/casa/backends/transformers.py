import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from casa.backends.base import InferenceBackend


class TransformersBackend(InferenceBackend):

    def __init__(self, model_name: str, dtype=torch.bfloat16, device_map="auto", **kwargs):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device_map, **kwargs
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device
        self._past_key_values = None
        self._cached_length = 0

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        return self._device

    def reset_cache(self):
        self._past_key_values = None
        self._cached_length = 0

    def get_next_token_logprobs(self, token_ids: List[int]) -> torch.Tensor:
        if self._past_key_values is not None:
            new_ids = torch.tensor(
                [token_ids[self._cached_length:]], device=self.device
            )
        else:
            new_ids = torch.tensor([token_ids], device=self.device)

        with torch.no_grad():
            output = self._model(
                new_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
            )

        self._past_key_values = output.past_key_values
        self._cached_length = len(token_ids)

        logits = output.logits[0, -1, :]
        return torch.log_softmax(logits, dim=-1).cpu()