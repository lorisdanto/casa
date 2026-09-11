"""Regressions for bugs an independent review found that the other suites could not.

Each of these passed every existing test. They are here because the toy setups elsewhere use a
five-token vocabulary, a single prompt, and targets whose envelope is exact at the leaf, and each
bug below hides in exactly one of those blind spots.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from casa.algebra import Model, mean
from casa.envelope_runtime import _RecognizerMask
from casa.samplers.mars import MARS

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(label)
    print(f"  {'ok    ' if cond else 'FAIL  '}{label}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------------------------
# 1. A real vocabulary. Summing a float32 log-softmax over 150k entries is off by about 5e-5, so
#    a tolerance of 1e-6 on the envelope condition rejected every ordinary model on its very first
#    expansion. Nothing with a five-token alphabet can see this.
# ---------------------------------------------------------------------------------------------

BIG = 152064
EOS_BIG = 3


class BigTok:
    eos_token_id = EOS_BIG

    def __len__(self):
        return BIG

    def get_vocab(self):
        return {"t%d" % i: i for i in range(8)}  # only compared against itself here

    def encode(self, text, add_special_tokens=False):
        return [7]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class BigModel:
    device = torch.device("cpu")

    def __init__(self, seed, eos_boost=0.0, scale=2.0):
        self.seed = seed
        self.eos_boost = eos_boost
        self.scale = scale
        self.calls = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        self.calls += 1
        n = input_ids.shape[-1]
        g = torch.Generator().manual_seed(self.seed + n)
        logits = torch.randn(n, BIG, generator=g) * self.scale
        logits[:, EOS_BIG] += self.eos_boost  # terminate promptly
        out = type("Out", (), {})()
        out.logits = logits.unsqueeze(0)
        out.past_key_values = None
        return out


class BigLLM:
    def __init__(self, seed, tokenizer, eos_boost=0.0, scale=2.0):
        self.model = BigModel(seed, eos_boost, scale)
        self.tokenizer = tokenizer
        self.model_id = "big"

    def format_prompt(self, prompt):
        return prompt


print("\nA production-sized vocabulary must not trip the envelope guard")
btok = BigTok()
P = Model(BigLLM(1, btok, eos_boost=14.0), name="P")
R = Model(BigLLM(2, btok, eos_boost=14.0), name="R")
try:
    sampler = MARS(P ** 0.5 * R ** 0.5, max_new_tokens=4)
    got = sampler.sample("hi", n_samples=5, max_attempts=200)
    check(f"intersect over a {BIG}-token vocabulary", len(got) == 5,
          f"{len(got)} sample(s), {sampler.stats.expansions} expansions")
except ValueError as e:
    check(f"intersect over a {BIG}-token vocabulary", False, f"raised: {str(e)[:110]}")

try:
    solo = MARS(Model(BigLLM(3, btok, eos_boost=14.0), name="S"), max_new_tokens=4)
    got = solo.sample("hi", n_samples=5, max_attempts=200)
    check("a single model over the same vocabulary", len(got) == 5, f"{len(got)} sample(s)")
except ValueError as e:
    check("a single model over the same vocabulary", False, f"raised: {str(e)[:110]}")

# Measure the arithmetic directly, so this fails loudly rather than half the time. A flat-ish
# distribution is the honest case: a peaked one has few entries contributing to the sum and so
# understates the accumulation error.
from casa import envelope_runtime as _rt              # noqa: E402
from casa.samplers.mars import _LOG_SLACK             # noqa: E402

_slack = math.expm1(_LOG_SLACK)
_worst = 0.0
for _seed in range(12):
    _flat = Model(BigLLM(100 + _seed, btok, scale=0.5), name="M")
    _ratios = _rt.plan(_flat.envelope(), {}, 1.0).fresh().log_ratios()
    _worst = max(_worst, abs(float(torch.exp(_ratios.double()).sum()) - 1.0))
check("float32 error over this vocabulary exceeds a 1e-6 tolerance", _worst > 1e-6,
      f"worst |sum-1| = {_worst:.2e}, which is what used to raise on the first expansion")
check("and stays well inside the slack actually used", _worst < _slack / 4,
      f"slack = {_slack:.1e}, margin {_slack / max(_worst, 1e-30):.0f}x")


# ---------------------------------------------------------------------------------------------
# 2. Overriding one model's prompt used to replace the whole mapping, leaving every other model
#    on the empty prompt with no error.
# ---------------------------------------------------------------------------------------------

print("\nA partial prompt override must not blank the other models")


class SpyTok(BigTok):
    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=False):
        self.seen.append(text)
        return [7]


stok = SpyTok()
A = Model(BigLLM(4, stok, eos_boost=14.0), name="A")
B = Model(BigLLM(5, stok, eos_boost=14.0), name="B")
m = MARS(A ** 0.5 * B ** 0.5, max_new_tokens=3)
stok.seen.clear()
m.sample("shared prompt", n_samples=1, max_attempts=50, prompts={id(A): "special"})
check("the overridden model got its override", "special" in stok.seen, str(stok.seen))
check("the other model still got the shared prompt", "shared prompt" in stok.seen, str(stok.seen))


# ---------------------------------------------------------------------------------------------
# 3. In the coverage regime the trie holds a dominating envelope, so the weight that telescopes
#    out of it is not the target's. The reported figure must be the target's.
# ---------------------------------------------------------------------------------------------

print("\nThe reported weight must be the target's, not the dominating envelope's")

V, EOS = 4, 2
PT = {"a": [0.3, 0.2, 0.5, 0.0], "b": [0.1, 0.4, 0.5, 0.0]}
RT = {"a": [0.6, 0.1, 0.3, 0.0], "b": [0.2, 0.2, 0.6, 0.0]}


def state(ctx):
    return "a" if not ctx or ctx[-1] == 0 else "b"


class SmallTok:
    eos_token_id = EOS

    def __len__(self):
        return V

    def get_vocab(self):
        return {"0": 0, "1": 1, "$": EOS, "<s>": 3}

    def encode(self, text, add_special_tokens=False):
        return [3]

    def decode(self, ids):
        return "".join("01$s"[i] for i in ids)


class SmallModel:
    device = torch.device("cpu")

    def __init__(self, table):
        self.table = table

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        ids = input_ids[0].tolist()
        rows = [self.table[state(ids[1:j + 1])] for j in range(len(ids))]
        out = type("Out", (), {})()
        out.logits = torch.log(torch.tensor(rows).clamp_min(1e-30)).unsqueeze(0)
        out.past_key_values = None
        return out


class SmallLLM:
    def __init__(self, table, tokenizer):
        self.model = SmallModel(table)
        self.tokenizer = tokenizer
        self.model_id = "small"

    def format_prompt(self, prompt):
        return ""


def weight(table, seq):
    w, ctx = 1.0, []
    for t in list(seq) + [EOS]:
        w *= table[state(ctx)][t]
        ctx.append(t)
    return w


stk = SmallTok()
p = Model(SmallLLM(PT, stk), name="P")
r = Model(SmallLLM(RT, stk), name="R")
torch.manual_seed(0)
cov = MARS(mean([p, r], tau=math.inf), max_new_tokens=4)
res = cov.sample("", n_samples=250, max_attempts=500)

worst, shown = 0.0, []
for s in res:
    seq = tuple(s.token_ids[:-1])
    want = math.log(max(weight(PT, seq), weight(RT, seq)))   # tau = +inf is the maximum
    worst = max(worst, abs(s.raw_logprob - want))
    if len(shown) < 3 and abs(s.raw_logprob - want) > 1e-6:
        shown.append((seq, s.raw_logprob, want))
check("reported weight matches max(P, R) exactly", worst < 1e-4, f"worst error {worst:.2e}")
for seq, got, want in shown:
    print(f"           {seq} reported {got:.5f} true {want:.5f}")


# ---------------------------------------------------------------------------------------------
# 4. A CASA Grammar holds one recognizer and shares it. Whoever used it last leaves it mid-path.
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# 5. The leaf gap used to be recomputed at every yielded sequence, which forced a forward pass per
#    model on a cached node and defeated the caching the whole design rests on.
# ---------------------------------------------------------------------------------------------

print("\nThe coverage regime must not pay a forward pass per yielded sequence")
torch.manual_seed(1)
p2 = Model(SmallLLM(PT, stk), name="P")
r2 = Model(SmallLLM(RT, stk), name="R")
perf = MARS(mean([p2, r2], tau=math.inf), max_new_tokens=4)
perf.sample("", n_samples=400, max_attempts=500)
budget = 2 * perf.stats.expansions + 8
check("model calls stay proportional to expansions, not to leaves",
      perf.stats.model_calls <= budget,
      f"{perf.stats.model_calls} calls, {perf.stats.expansions} expansions, "
      f"{perf.stats.leaf_trials} leaf trials, budget {budget}")

print("\nA recognizer left mid-path by a previous run must be rewound")


class Rec:
    vocab_size = 5

    def __init__(self):
        self.current_index = 0
        self.path = []

    def reset(self):
        self.current_index = 0
        self.path = []

    def try_advance_token_ids(self, ids):
        new = ids[self.current_index:].tolist()
        self.path += new
        self.current_index += len(new)
        return True

    def filter_vocab(self):
        return [t for t in range(5) if t % 2 == len(self.path) % 2]

    def apply_token_bitmask(self, logits, bitmask):
        allowed = set(bitmask)
        for t in range(logits.shape[-1]):
            if t not in allowed:
                logits[0, t] = float("-inf")


rec = Rec()
rec.try_advance_token_ids(torch.tensor([0]))       # someone else used it and walked one step
check("recognizer starts dirty", rec.path == [0])
mask = _RecognizerMask(rec, "L")
m0 = [float(x) for x in mask([], 5)]
check("the empty prefix still gets the root mask", m0 == [0.0, -math.inf, 0.0, -math.inf, 0.0],
      str(m0))

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all regression checks passed")
