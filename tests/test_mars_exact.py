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

from casa.algebra import Model, constrain, intersect, mean, reweight
from casa.samplers.mars import MARS

ZERO, ONE, PLUS, EOS, BOS = 0, 1, 2, 3, 4
VOCAB = 5
MAX_BODY = 5  # after this many tokens both models emit end-of-sequence with probability one

# State depends only on the last token, exactly as in the paper's table.
P_TABLE = {
    "start": [0.4, 0.3, 0.3, 0.0, 0.0],
    "digit": [0.1, 0.1, 0.5, 0.3, 0.0],
    "plus": [0.5, 0.4, 0.1, 0.0, 0.0],
}
R_TABLE = {
    "start": [0.5, 0.5, 0.0, 0.0, 0.0],
    "digit": [0.0, 0.0, 0.4, 0.6, 0.0],
    "plus": [0.5, 0.5, 0.0, 0.0, 0.0],
}
STOP = [0.0, 0.0, 0.0, 1.0, 0.0]

# The same language as lang_mask, as a table, so the exact answer can be enumerated.
mask_table_for_scorer = {
    "start": [1.0, 0.0, 0.0, 0.0, 0.0],
    "digit": [0.0, 0.0, 1.0, 1.0, 0.0],
    "plus": [1.0, 0.0, 0.0, 0.0, 0.0],
}


def state_of(ctx):
    if not ctx:
        return "start"
    return "plus" if ctx[-1] == PLUS else "digit"


def row(table, ctx):
    return STOP if len(ctx) >= MAX_BODY else table[state_of(ctx)]


class FakeTokenizer:
    eos_token_id = EOS
    _vocab = {"0": ZERO, "1": ONE, "+": PLUS, "$": EOS, "<s>": BOS}

    def __len__(self):
        return VOCAB

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        # A one-token prompt. A forward pass gives the conditional for a token only from the
        # position before it, so the first generated token needs something to its left.
        return [BOS]

    def decode(self, ids):
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv[i] for i in ids if i != EOS)


class FakeModel:
    """Table logits at every position, as a real causal model returns."""

    device = torch.device("cpu")

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        self.calls += 1
        ids = input_ids[0].tolist()
        # Position j predicts token j+1 given ids[:j+1]; ids[0] is the prompt token.
        rows = [row(self.table, ids[1:j + 1]) for j in range(len(ids))]
        logits = torch.log(torch.tensor(rows, dtype=torch.float32).clamp_min(1e-30)).unsqueeze(0)

        class Out:
            pass

        out = Out()
        out.logits = logits
        out.past_key_values = None
        return out


class FakeLLM:
    def __init__(self, table, tokenizer):
        self.model = FakeModel(table)
        self.tokenizer = tokenizer
        self.model_id = "fake"

    def format_prompt(self, prompt):
        return ""


def enumerate_target(op, table_a=None, table_b=None, post=None):
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
                if post is not None:
                    w *= post(tuple(ctx))
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


def run(label, target, op, n=12_000, seed=0, slack=3.0, tables=None, post=None):
    torch.manual_seed(seed)
    sampler = MARS(target, max_new_tokens=MAX_BODY + 1)
    results = sampler.sample("", n_samples=n, max_attempts=10_000)

    counts = Counter(tuple(r.token_ids[:-1]) for r in results)
    emp = {k: v / len(results) for k, v in counts.items()}

    weights = enumerate_target(op, *(tables or (None, None)), post=post)
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


def trie_inconsistencies(sampler, tol=1e-6):
    """Every parent's bound on a child must equal that child's own total.

    Precisely: the parent's *correction* term for a child must equal that child's total. The
    parent's full entry, correction plus ratio, is the child's mass relative to the parent, which
    is a different quantity and is not what propagation writes.

    This is the invariant propagation exists to maintain, and it holds globally once any descent
    has finished, because propagation walks all the way to the root. Checking it directly catches
    a missed propagation that a distribution test would only see as a faint bias, if at all.
    """
    bad = []

    def walk(node, path):
        if node.raw_logprob is None:
            return
        for tokid, child in node.children.items():
            if child.raw_logprob is None:
                continue  # created but never expanded; the parent still holds the original bound
            parent_says = float(torch.exp(node.log_theta[0, tokid]))
            child_total = float(torch.exp(child.raw_logprob[0] + child.log_theta[0]).sum())
            if abs(parent_says - child_total) > tol * max(1.0, abs(child_total)):
                bad.append((path + [tokid], parent_says, child_total))
            walk(child, path + [tokid])

    walk(sampler.trie.root, [])
    return bad


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
    mask_table = {"start": [1.0, 0.0, 0.0, 0.0, 0.0], "digit": [0.0, 0.0, 1.0, 1.0, 0.0],
                  "plus": [1.0, 0.0, 0.0, 0.0, 0.0]}
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    m = Model(FakeLLM(mask_table, tok), name="L", sub_stochastic=False)
    ok, cars_like = run("P constrained to 0(+0)*", p * m, lambda a, b: a * b,
                        tables=(P_TABLE, mask_table))
    passed &= ok

    print("\nPrefix-monotone scorer, the path a grammar takes through MARS")

    def lang_mask(ctx, vocab_size):
        # Only "0", then alternating "+" and "0", ending any time after a digit: 0(+0)*
        allow = {PLUS, EOS} if (ctx and ctx[-1] != PLUS) else {ZERO}
        return torch.tensor([0.0 if t in allow else float("-inf") for t in range(vocab_size)])

    p_ = Model(FakeLLM(P_TABLE, tok), name="P")
    ok, _ = run("P * L  via a Scorer", constrain(p_, lang_mask),
                lambda a, b: a * b, tables=(P_TABLE, mask_table_for_scorer))
    passed &= ok

    p_ = Model(FakeLLM(P_TABLE, tok), name="P")
    ok, _ = run("(P * L)**2  sharpened", constrain(p_, lang_mask) ** 2,
                lambda a, b: a * a * b, tables=(P_TABLE, mask_table_for_scorer))
    passed &= ok

    print("\nOutput-only verifier: scored at the leaf, never in the trie")

    def verifier(ctx):
        # Depends on the whole sequence, so it cannot tighten any prefix bound.
        return 0.25 if len(ctx) % 2 else 1.0

    p = Model(FakeLLM(P_TABLE, tok), name="P")
    ok, ver = run("P reweighted by a verifier", reweight(p, verifier, prefix_monotone=False),
                  lambda a, b: a, tables=(P_TABLE, P_TABLE), post=verifier)
    passed &= ok
    if ver.stats.leaf_trials == 0:
        print("  FAIL   the verifier was never consulted")
        passed = False
    else:
        print(f"         consulted at {ver.stats.leaf_trials} leaves, "
              f"{ver.stats.leaf_rejections} rejected")

    print("\nTrie consistency: every parent's bound equals its child's total")
    for label, target, seed in [
        ("intersect", None, 10),
        ("max, leaf rejection", None, 11),
    ]:
        torch.manual_seed(seed)
        a = Model(FakeLLM(P_TABLE, tok), name="P")
        b = Model(FakeLLM(R_TABLE, tok), name="R")
        expr = intersect(a, b) if label == "intersect" else mean([a, b], tau=math.inf)
        sm = MARS(expr, max_new_tokens=MAX_BODY + 1)
        sm.sample("", n_samples=1500, max_attempts=10_000)
        bad = trie_inconsistencies(sm)
        print(f"  {'ok    ' if not bad else 'FAIL  '}{label:28s} "
              f"{len(bad)} inconsistent edge(s) over {sm.stats.expansions} expansions")
        for path, ps, ct in bad[:3]:
            print(f"           path={path} parent says {ps:.9f}, child holds {ct:.9f}")
        passed &= not bad

    print("\nCaching: a revisited prefix must cost no model call")
    torch.manual_seed(4)
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    r = Model(FakeLLM(R_TABLE, tok), name="R")
    sampler = MARS(intersect(p, r), max_new_tokens=MAX_BODY + 1)
    sampler.sample("", n_samples=2000, max_attempts=10_000)
    calls = sampler.stats.model_calls
    raw = p.llm.model.calls + r.llm.model.calls
    if calls != raw:
        print(f"  FAIL   stats.model_calls={calls} disagrees with the models' own count {raw}")
        passed = False
    # Two models, one pass each per expanded node, plus one per leaf-weight query. If advancing
    # through cached nodes were still asking the models, this would scale with total descent
    # length instead, which is orders of magnitude larger.
    budget = 2 * sampler.stats.expansions + 8
    ok_cache = calls <= budget
    print(f"  {'ok    ' if ok_cache else 'FAIL  '}model calls {calls} for "
          f"{sampler.stats.expansions} expansions over {sampler.stats.descents} descents "
          f"(budget {budget})")
    passed &= ok_cache

    print("\nLength cutoff: a descent that never terminates must not crash")
    torch.manual_seed(5)
    # max_new_tokens below the forced-stop depth, so every descent runs out of room.
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    r = Model(FakeLLM(R_TABLE, tok), name="R")
    short = MARS(intersect(p, r), max_new_tokens=2)
    try:
        got = short.sample("", n_samples=3, max_attempts=50)
        ok_cut = short.stats.length_cutoffs > 0
        print(f"  {'ok    ' if ok_cut else 'FAIL  '}survived {short.stats.length_cutoffs} cutoffs, "
              f"{len(got)} sample(s), {short.stats.timeouts} timeout(s)")
        passed &= ok_cut
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  raised {type(e).__name__}: {e}")
        passed = False

    print()
    print("all exactness checks passed" if passed else "FAILURES")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
