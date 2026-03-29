from casa import Grammar, CARS
from casa import TransformersBackend
from casa import VLLMBackend
import torch
import gc


print("Starting...")

backend = TransformersBackend("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
grammar = Grammar.from_file("resources/grammars/smiles/acrylates.lark", backend.tokenizer)
with open("resources/prompts/smiles/acrylates.txt") as f:
    prompt = f.read().strip()
sampler = CARS(backend, grammar, max_new_tokens=256, verbose=True)
results = sampler.sample(prompt, n_samples=100)

if results:
	print("\nGenerated samples,")
	for i, result in enumerate(results, 1):
		print(f"  {i}. {prompt} {result.text}")
else:
	print("Failed to generate any samples")
 
del sampler
del backend
torch.cuda.empty_cache()
gc.collect()

# With vLLM
backend = VLLMBackend("meta-llama/Llama-3.1-8B-Instruct")
grammar = Grammar.from_file("resources/grammars/smiles/acrylates.lark", backend.tokenizer)
with open("resources/prompts/smiles/acrylates.txt") as f:
    prompt = f.read().strip()
sampler = CARS(backend, grammar, max_new_tokens=256, verbose=True)
results = sampler.sample(prompt, n_samples=100)

if results:
	print("\nGenerated samples,")
	for i, result in enumerate(results, 1):
		print(f"  {i}. {prompt} {result.text}")
else:
	print("Failed to generate any samples")