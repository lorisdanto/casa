"""MARS against a real Hugging Face model.

Everything else in this suite uses hand-built tables, which is how the exactness claims get
checked. None of it exercises a real tokenizer, a real fifty-thousand-entry vocabulary, a real
prompt encoding, or the ``LLM`` wrapper. This does, on GPT-2, which is small enough to run on a
laptop CPU.

The setting is the one the evaluation grid calls *within-model*: one set of weights, two prompts,
combined into a single target. That is the experiment Chan et al. run at token level, and it needs
no second model and no byte-level machinery.

Run directly; it downloads GPT-2 on first use.
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from casa import LLM
from casa.algebra import Model, intersect, mean
from casa.samplers.mars import MARS

MODEL_ID = "gpt2"
failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(label)
    print(f"  {'ok    ' if cond else 'FAIL  '}{label}{('  ' + detail) if detail else ''}")


print(f"\nLoading {MODEL_ID}")
t0 = time.time()
llm = LLM.from_pretrained(MODEL_ID, is_chat_model=False, dtype=torch.float32, device_map=None)
print(f"  loaded in {time.time() - t0:.1f}s  vocab={len(llm.tokenizer)}  device={llm.device}")
check("vocabulary is real, not a toy", len(llm.tokenizer) > 50_000, f"{len(llm.tokenizer)} tokens")

PHYS = "My favorite physicist is"
AUTH = "My favorite author is"

print("\nWithin-model intersection: one model, two prompts")
P = Model(llm, name="P", prompt=PHYS)
R = Model(llm, name="P", prompt=AUTH)

torch.manual_seed(0)
t0 = time.time()
sampler = MARS(intersect(P, R), max_new_tokens=6, temperature=1.0)
out = sampler.sample(n_samples=8, max_attempts=2000)
el = time.time() - t0

check("produced the requested samples", len(out) == 8, f"{len(out)} of 8")
print(f"         {el:.1f}s, {sampler.stats.descents} descents, "
      f"{sampler.stats.expansions} expansions, {sampler.stats.model_calls} model calls, "
      f"acceptance {sampler.acceptance_rate:.3f}, root mass {sampler.root_mass:.4f}")
for r in out[:8]:
    print(f"           {PHYS!r} + {AUTH!r} -> {r.text!r}")

check("root mass fell below one", sampler.root_mass < 1.0, f"{sampler.root_mass:.4f}")
check("every sample carries a finite weight",
      all(math.isfinite(r.raw_logprob) for r in out))
check("caching held: model calls stay near two per expansion",
      sampler.stats.model_calls <= 2 * sampler.stats.expansions + 4,
      f"{sampler.stats.model_calls} calls / {sampler.stats.expansions} expansions")

print("\nThe envelope guard must not fire on a real vocabulary")
# This is the blocker an outside review caught: a float32 log-softmax over a real vocabulary sums
# to 1 only to about 1e-5, and a tight tolerance rejected every ordinary model on its first
# expansion. Measure it here on actual GPT-2 logits rather than synthetic ones.
from casa import envelope_runtime as rt  # noqa: E402

plan = rt.plan(Model(llm, name="P", prompt=PHYS).envelope(), {}, 1.0)
ratios = plan.fresh().log_ratios()
err = abs(float(torch.exp(ratios.double()).sum()) - 1.0)
print(f"         |sum(exp(ratios)) - 1| = {err:.3e} on real logits")
check("a single real model samples without tripping the guard",
      len(MARS(Model(llm, name="P", prompt=PHYS), max_new_tokens=6).sample(
          n_samples=3, max_attempts=50)) == 3)
check("a single model needs no rejection at all",
      MARS(Model(llm, name="P", prompt=PHYS), max_new_tokens=6).sample(
          n_samples=3, max_attempts=5)[0].attempts == 1)

print("\nAgreement and coverage on the same pair")
for label, target in [("agree, min", P & R), ("max, leaf rejection", mean([P, R], tau=math.inf))]:
    torch.manual_seed(1)
    s = MARS(target, max_new_tokens=5)
    got = s.sample(n_samples=4, max_attempts=2000)
    ok = len(got) == 4
    check(f"{label}", ok, f"{len(got)}/4, acceptance {s.acceptance_rate:.3f}, "
                          f"{s.stats.expansions} expansions")
    for r in got[:2]:
        print(f"           -> {r.text!r}")

print("\nSanity: the intersection should prefer completions plausible under both prompts")
# Not an assertion about quality, just a visible check that conditioning is wired up: sampling the
# two prompts separately should look different from sampling their intersection.
torch.manual_seed(2)
for name, target in [("physicist only", Model(llm, name="P", prompt=PHYS)),
                     ("author only", Model(llm, name="P", prompt=AUTH)),
                     ("both", intersect(P, R))]:
    s = MARS(target, max_new_tokens=5)
    got = s.sample(n_samples=4, max_attempts=2000)
    print(f"  {name:16s} {[r.text for r in got]}")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all real-model checks passed")
