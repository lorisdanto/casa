import asyncio
import torch
from casa.backends.base import InferenceBackend
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.utils import Counter
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

class VLLMBackend(InferenceBackend):
	def __init__(self, model_name: str, engine_opts: dict = None):

		
		engine_opts = {
			"enable_prefix_caching": True,
			"disable_log_requests": True,
			"disable_async_output_proc": True,
			**(engine_opts or {}),
		}
		
		self._engine = AsyncLLMEngine.from_engine_args(
			AsyncEngineArgs(model=model_name, tokenizer=model_name, **engine_opts)
		)
		self._tokenizer = self._engine.engine.get_tokenizer()
		self._request_counter = Counter()
		self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	@property
	def tokenizer(self):
		return self._tokenizer
	
	@property
	def device(self) -> torch.device:
		return self._device
	
	def get_next_token_logprobs(self, token_ids: list) -> torch.Tensor:
		try:
			loop = asyncio.get_running_loop()
			return loop.run_until_complete(self._async_logprobs(token_ids))
		except RuntimeError:
			return asyncio.run(self._async_logprobs(token_ids))
	
	async def _async_logprobs(self, token_ids: list) -> torch.Tensor:
		req_id = str(next(self._request_counter))
		processor = _PassThroughLogitsProcessor()
		
		async for output in self._engine.generate(
			prompt=TokensPrompt(prompt_token_ids=token_ids),
			sampling_params=SamplingParams(
				max_tokens=1, n=1, detokenize=False,
				stop=None, ignore_eos=True,
				logits_processors=[processor],
			),
			request_id=req_id,
		):
			if output.finished:
				break
		
		return processor.log_probs.cpu()
	
	def shutdown(self):
		if not hasattr(self, '_shutdown_called'):
			self._shutdown_called = True
			try:
				self._engine.shutdown_background_loop()
				destroy_model_parallel()
				destroy_distributed_environment()
			except Exception:
				pass


class _PassThroughLogitsProcessor:
	def __init__(self):
		self.log_probs = None
	
	def __call__(self, past_token_ids, logits):
		self.log_probs = torch.log_softmax(logits, dim=-1, dtype=logits.dtype)
		return logits