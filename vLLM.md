# IMPORTANT

Note: The current build is unstable due to using vLLM v0 with Transformers 5.x. As a result, you will have to apply the following patches, AFTER installing it in your environment. Will look into a more robust solution soon.

```bash
sed -i 's/    vision_config: VisionEncoderConfig$/    vision_config: VisionEncoderConfig = None/'     .venv/lib/python3.11/site-packages/vllm/transformers_utils/configs/deepseek_vl2.py
```

```bash
sed -i 's/    projector_config: MlpProjectorConfig$/    projector_config: MlpProjectorConfig = None/'     .venv/lib/python3.11/site-packages/vllm/transformers_utils/configs/deepseek_vl2.py
```

```bash
sed -i 's/tokenizer\.all_special_tokens_extended/tokenizer.all_special_tokens/'     .venv/lib/python3.11/site-packages/vllm/transformers_utils/tokenizer.py
```

```bash
sed -i 's/super().__init__(\*args, \*\*kwargs, disable=True)/kwargs.pop("disable", None); super().__init__(*args, **kwargs, disable=True)/'     .venv/lib/python3.11/site-packages/vllm/model_executor/model_loader/weight_utils.py
```