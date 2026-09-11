"""MARS draws exactly from the target, checked against a hand-computable example.

Uses the arithmetic-expression example from the MARS paper: two tiny models over the alphabet
{0, 1, +} whose state depends only on the last token, intersected as a geometric mean. Because the
support is finite and enumerable, the exact target can be computed in closed form and compared with
what the sampler actually produces. No language model and no GPU are involved.

Also checks the claim that MARS reduces to CARS on a hard constraint, and that the acceptance rate
never decreases.
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from casa.algebra import Model, intersect, mean
from casa.samplers.mars import MARS

ZERO, ONE, PLUS, EOS = 0, 1, 2, 3
VOCAB = 4
MAX_BODY = 5  # after this many tokens both models emit end-of-sequence with probability one

# State depends only on the last token, exactly as in the paper's table.
P_TABLE = {
    "start": [0.4, 0.3, 0.3, 0.0],
    "digit": [0.1, 0.1, 0.5, 0.3],
    "plus": [0.5, 0.4, 0.1, 0.0],
}
R_TABLE = {
    "start": [0.5, 0.5, 0.0, 0.0],
    "digit": [0.0, 0.0, 0.4, 0.6],
    "plus": [0.5, 0.5, 0.0, 0.0],
}
STOP = [0.0, 0.0, 0.0, 1.0]


def state_of(ctx):
    if not ctx:
        return "start"
    return "plus" if ctx[-1] == PLUS else "digit"


def row(table, ctx):
    return STOP if len(ctx) >= MAX_BODY else table[state_of(ctx)]


class FakeTokenizer:
    eos_token_id = EOS
    _vocab = {"0": ZERO, "1": ONE, "+": PLUS, "$": EOS}

    def __len__(self):
        return VOCAB

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        return []

    def decode(self, ids):
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv[i] for i in ids if i != EOS)


class FakeModel:
    """Returns table logits. The cache carries the full id list so the table can be indexed."""

    device = torch.device("cpu")

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        self.calls += 1
        prior = list(past_key_values) if past_key_values else []
        new = input_ids[0].tolist()
        ids = prior + new
        probs = torch.tensor(row(self.table, ids), dtype=torch.float32)
        logits = torch.log(probs.clamp_min(1e-30)).view(1, 1, VOCAB)

        class Out:
            pass

        out = Out()
        out.logits = logits
        out.past_key_values = ids
        return out


class FakeLLM:
    def __init__(self, table, tokenizer):
        self.model = FakeModel(table)
        self.tokenizer = tokenizer
        self.model_id = "fake"

    def format_prompt(self, prompt):
        return ""


def enumerate_target(op, table_a=None, table_b=None):
    """Exact weights of every complete sequence, as a dict from token tuple to weight."""
    table_a = P_TABLE if table_a is None else table_a
    table_b = R_TABLE if table_b is None else table_b
    out = {}

    def walk(ctx, lp, lr):
        pr, rr = row(table_a, ctx), row(table_b, ctx)
        for tok in range(VOCAB):
            p, r = pr[tok], rr[tok]
            if p <= 0 and r <= 0:
                continue
            np_, nr = lp * p, lr * r
            if tok == EOS:
                w = op(np_, nr)
                if w > 0:
                    out[tuple(ctx)] = out.get(tuple(ctx), 0.0) + w
            elif len(ctx) < MAX_BODY:
                if np_ > 0 or nr > 0:
                    walk(ctx + [tok], np_, nr)

    walk([], 1.0, 1.0)
    return out


def total_variation(emp, exact):
    keys = set(emp) | set(exact)
    return 0.5 * sum(abs(emp.get(k, 0.0) - exact.get(k, 0.0)) for k in keys)


def noise_floor(exact, n):
    """Expected total variation between the exact law and an n-sample draw *from it*.

    For each outcome the absolute deviation has mean about sqrt(2 p (1-p) / (pi n)), so the
    expected total variation is half their sum. Comparing against a fixed constant instead would
    either pass a broken sampler on a small support or fail a correct one on a large support.
    """
    return 0.5 * sum(math.sqrt(2 * p * (1 - p) / (math.pi * n)) for p in exact.values())


def run(label, target, op, n=20_000, seed=0, slack=3.0, tables=None):
    torch.manual_seed(seed)
    sampler = MARS(target, max_new_tokens=MAX_BODY + 1)
    results = sampler.sample("", n_samples=n, max_attempts=10_000)

    counts = Counter(tuple(r.token_ids[:-1]) for r in results)
    emp = {k: v / len(results) for k, v in counts.items()}

    weights = enumerate_target(op, *(tables or (None, None)))
    z = sum(weights.values())
    exact = {k: v / z for k, v in weights.items()}

    tv = total_variation(emp, exact)
    tol = slack * noise_floor(exact, n)
    ok = tv < tol and len(results) == n
    print(f"  {'ok    ' if ok else 'FAIL  '}{label:28s} "
          f"support={len(exact):3d} TV={tv:.4f} tol={tol:.4f} "
          f"accept={sampler.acceptance_rate:.3f} exp={sampler.stats.expansions}")
    if not ok:
        rows = sorted(exact, key=lambda k: -exact[k])[:8]
        for k in rows:
            print(f"           {''.join('01+'[t] for t in k) or '(empty)':10s} "
                  f"exact={exact[k]:.4f} emp={emp.get(k, 0.0):.4f}")
    return ok, sampler


def main():
    tok = FakeTokenizer()
    passed = True

    print("\nExactness: empirical distribution against the closed form")

    # Geometric mean, the paper's running example.
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    r = Model(FakeLLM(R_TABLE, tok), name="R")
    ok, s = run("intersect, sqrt(P*R)", intersect(p, r), lambda a, b: math.sqrt(a * b))
    passed &= ok

    # Product of experts: exponents sum to 2, still an envelope.
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    r = Model(FakeLLM(R_TABLE, tok), name="R")
    ok, _ = run("product, P*R", p * r, lambda a, b: a * b)
    passed &= ok

    # Agreement.
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    r = Model(FakeLLM(R_TABLE, tok), name="R")
    ok, _ = run("agree, min(P,R)", p & r, min)
    passed &= ok

    # Coverage, which needs the dominating mixture and a leaf acceptance step.
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    r = Model(FakeLLM(R_TABLE, tok), name="R")
    ok, s_max = run("max(P,R), leaf rejection", mean([p, r], tau=math.inf), max)
    passed &= ok
    if s_max.stats.leaf_trials == 0:
        print("  FAIL   coverage regime never exercised the leaf acceptance step")
        passed = False
    else:
        print(f"         leaf acceptance exercised: {s_max.stats.leaf_rejections}"
              f"/{s_max.stats.leaf_trials} rejected")

    # Mixture, tau = 1, an envelope with no leaf step.
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    r = Model(FakeLLM(R_TABLE, tok), name="R")
    ok, _ = run("union, mixture", p | r, lambda a, b: 0.5 * a + 0.5 * b)
    passed &= ok

    # A single model: MARS must reproduce it exactly, with no rejections at all.
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    ok, solo = run("single model P", p, lambda a, b: a)
    passed &= ok
    if solo.stats.rejections:
        print(f"  FAIL   a single model needs no rejection, saw {solo.stats.rejections}")
        passed = False
    else:
        print("         no rejections, as expected for one sub-stochastic model")

    print("\nMonotonicity: the root mass never increases")
    # The root mass is the quantity the theorem says is monotone; the acceptance rate is Z over it,
    # so a non-increasing root mass is exactly a non-decreasing acceptance rate.
    torch.manual_seed(2)
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    r = Model(FakeLLM(R_TABLE, tok), name="R")
    sampler = MARS(intersect(p, r), max_new_tokens=MAX_BODY + 1)
    masses = [sampler.root_mass]  # 1.0 before anything is learned
    for i in range(30):
        sampler.sample("", n_samples=1 if i < 10 else 20, max_attempts=10_000)
        masses.append(sampler.root_mass)
    non_increasing = all(b <= a + 1e-9 for a, b in zip(masses, masses[1:]))
    decreased = masses[-1] < masses[0] - 1e-9
    ok_mono = non_increasing and decreased
    print(f"  {'ok    ' if ok_mono else 'FAIL  '}root mass non-increasing   "
          f"{masses[0]:.5f} -> {masses[-1]:.5f} over {len(masses)} checkpoints")
    if not non_increasing:
        bad = [(a, b) for a, b in zip(masses, masses[1:]) if b > a + 1e-9]
        print(f"         increased at {len(bad)} step(s), first {bad[0]}")
    if not decreased:
        print("         never decreased, so the test witnessed nothing")
    passed &= ok_mono

    print("\nReduction to CARS: a hard constraint gives the constrained LM distribution")
    # R is already an indicator-like model in structure; build a genuine mask instead.
    torch.manual_seed(3)
    mask_table = {"start": [1.0, 0.0, 0.0, 0.0], "digit": [0.0, 0.0, 1.0, 1.0],
                  "plus": [1.0, 0.0, 0.0, 0.0]}
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    m = Model(FakeLLM(mask_table, tok), name="L", sub_stochastic=False)
    ok, cars_like = run("P constrained to 0(+0)*", p * m, lambda a, b: a * b,
                        tables=(P_TABLE, mask_table))
    passed &= ok

    print()
    print("all exactness checks passed" if passed else "FAILURES")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
