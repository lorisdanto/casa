"""The runtime guards fire when an expression is not really an envelope.

The algebra checks what it can see: which operations are combined and with what exponents. It
cannot see the numbers a potential actually produces, so a potential that claims to be bounded and
is not will pass validation and reach the sampler. At that point the only honest behaviour is to
stop, because sampling on would return results that look fine and are not exact.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from casa.algebra import Model
from casa.samplers.mars import MARS

VOCAB, EOS, BOS = 4, 2, 3
failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(label)
    print(f"  {'ok    ' if cond else 'FAIL  '}{label}{('  ' + detail) if detail else ''}")


class Tok:
    eos_token_id = EOS

    def __len__(self):
        return VOCAB

    def get_vocab(self):
        return {"a": 0, "b": 1, "$": EOS, "<s>": BOS}

    def encode(self, text, add_special_tokens=False):
        return [BOS]

    def decode(self, ids):
        return "".join("ab$<s>"[i] for i in ids)


class Flat:
    """Every position returns the same row of weights."""

    device = torch.device("cpu")

    def __init__(self, weights):
        self.w = weights

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        n = input_ids.shape[-1]
        rows = torch.tensor([self.w] * n, dtype=torch.float32)
        out = type("Out", (), {})()
        out.logits = torch.log(rows.clamp_min(1e-30)).unsqueeze(0)
        out.past_key_values = None
        return out


class LLM:
    def __init__(self, weights, tokenizer):
        self.model = Flat(weights)
        self.tokenizer = tokenizer
        self.model_id = "fake"

    def format_prompt(self, prompt):
        return ""


tok = Tok()

print("\nA potential whose weights exceed one is caught at the first expansion")
# Declared not sub-stochastic, so it is passed through without normalization, and every entry is
# 2.0. The algebra accepts it under an exponent of one; the runtime must not.
good = Model(LLM([0.3, 0.3, 0.4, 0.0], tok), name="P")
bad = Model(LLM([2.0, 2.0, 2.0, 0.0], tok), name="B", sub_stochastic=False)
sampler = MARS(good * bad, max_new_tokens=4)
try:
    sampler.sample("", n_samples=1, max_attempts=5)
    check("condition (ii) violation raises", False, "sampled without complaint")
except ValueError as e:
    check("condition (ii) violation raises", "condition (ii) violated" in str(e))
    check("the message quantifies the violation", "carry" in str(e) and "exceeds one" in str(e))

print("\nThe same expression with a genuinely bounded potential is fine")
ok_pot = Model(LLM([0.5, 0.5, 0.5, 0.0], tok), name="B", sub_stochastic=False)
sampler = MARS(good * ok_pot, max_new_tokens=4)
got = sampler.sample("", n_samples=5, max_attempts=200)
check("bounded potential samples", len(got) == 5, f"got {len(got)}")

print("\nA model with no prompt is refused rather than silently mis-scored")


class NoPromptTok(Tok):
    def encode(self, text, add_special_tokens=False):
        return []


bare = Model(LLM([0.3, 0.3, 0.4, 0.0], NoPromptTok()), name="P")
try:
    MARS(bare, max_new_tokens=4).sample("", n_samples=1)
    check("empty prompt raises", False, "accepted")
except ValueError as e:
    check("empty prompt raises", "empty prompt" in str(e))

print("\nMixing token spaces is refused")
other = Tok()
other.get_vocab = lambda: {"x": 0, "y": 1, "$": EOS, "<s>": BOS}
a = Model(LLM([0.3, 0.3, 0.4, 0.0], tok), name="P")
b = Model(LLM([0.3, 0.3, 0.4, 0.0], other), name="R")
try:
    MARS(a * b, max_new_tokens=4).sample("", n_samples=1)
    check("tokenizer mismatch raises", False, "accepted")
except Exception as e:  # noqa: BLE001
    check("tokenizer mismatch raises", "token space" in str(e), type(e).__name__)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all guard checks passed")
