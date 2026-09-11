"""The envelope algebra, checked against the operations table of the MARS paper.

Runs in milliseconds with no model and no GPU: every case here is about which targets admit an
envelope, which is a property of the expression, not of any language model.
"""

import math
import sys

# The algebra is about expressions, not tensors, so `casa.algebra` deliberately imports nothing
# heavier than the standard library. Load it by path rather than through `casa/__init__.py`, which
# pulls in torch and the grammar engines: that keeps this test runnable anywhere and fails loudly
# if the module ever grows a dependency it should not have.
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "casa_algebra", Path(__file__).resolve().parent.parent / "src" / "casa" / "algebra.py"
)
_alg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _alg  # dataclasses resolves annotations via sys.modules
_spec.loader.exec_module(_alg)

Model = _alg.Model
NoEnvelope = _alg.NoEnvelope
Scorer = _alg.Scorer
agree, constrain, intersect = _alg.agree, _alg.constrain, _alg.intersect
mean, reweight, union = _alg.mean, _alg.reweight, _alg.union


class _FakeLLM:
    """Stand-in for a CASA LLM. The algebra never looks inside one."""

    def __init__(self, tag):
        self.tag = tag


P = Model(_FakeLLM("a"), name="P")
R = Model(_FakeLLM("b"), name="R")

failures = []


def check(label, fn, want_ok=True, want_leaf=None):
    try:
        env = fn().envelope()
        ok, detail = True, env.verdict.reason
        leaf = env.leaf_acceptance
    except NoEnvelope as e:
        ok, detail, leaf = False, str(e), None
    except Exception as e:  # noqa: BLE001
        failures.append(f"{label}: unexpected {type(e).__name__}: {e}")
        print(f"  ERROR  {label}: {type(e).__name__}: {e}")
        return

    good = ok == want_ok and (want_leaf is None or (leaf is not None and math.isclose(leaf, want_leaf)))
    if not good:
        failures.append(f"{label}: ok={ok} (wanted {want_ok}), leaf={leaf} (wanted {want_leaf})")
    mark = "ok    " if good else "FAIL  "
    verdict = "envelope" if ok else "refused"
    extra = f", leaf>={leaf:g}" if leaf is not None else ""
    print(f"  {mark}{label:34s} {verdict}{extra}")
    if not good:
        print(f"         -> {detail}")


print("\nOperations that admit an envelope")
check("P                    base", lambda: P)
check("P * R                product", lambda: P * R)
check("P**.5 * R**.5        intersect", lambda: P ** 0.5 * R ** 0.5)
check("intersect(P, R)      sugar", lambda: intersect(P, R))
check("P ** 2               sharpen", lambda: P ** 2)
check("P & R                agree, min", lambda: P & R)
check("P | R                union, mixture", lambda: P | R)
check("mean tau=-1          harmonic", lambda: mean([P, R], tau=-1))
check("mean tau=0           product", lambda: mean([P, R], tau=0))
check("mean tau=1           mixture", lambda: mean([P, R], tau=1))

print("\nOperations that need a final acceptance step")
check("mean tau=2           quadratic", lambda: mean([P, R], tau=2), want_leaf=0.5)
check("mean tau=+inf        maximum", lambda: mean([P, R], tau=math.inf), want_leaf=0.5)
check("mean tau=2, w=(.25,.75)", lambda: mean([P, R], tau=2, weights=[0.25, 0.75]), want_leaf=0.25)
check("reweight by a verifier", lambda: reweight(P, lambda ctx: 0.5, prefix_monotone=False), want_leaf=0.0)

print("\nConstraints, which is where CARS lives")


def _mask(ctx, vocab_size):
    return [0.0] * vocab_size  # everything allowed; the algebra never calls this


check("constrain(P, L)      CARS", lambda: constrain(P, _mask))
check("constrain(P*R, L)    CARS+ensemble", lambda: constrain(P * R, _mask))
check("prefix-monotone scorer",
      lambda: P * Scorer(lambda c: 1.0, prefix_monotone=True, log_mask=_mask))

print("\nA leaf requirement must reach the root, however deeply it is nested")
S = _alg.Model(_FakeLLM("c"), name="S")
check("min(max(P,R), S)", lambda: mean([mean([P, R], tau=math.inf), S], tau=-math.inf),
      want_leaf=0.5)
check("max(P,R) * S", lambda: mean([P, R], tau=math.inf) * S, want_leaf=0.5)
check("mix(quadratic(P,R), S)", lambda: mean([mean([P, R], tau=2), S], tau=1), want_leaf=0.5)
check("plain min(P,S) needs no leaf step", lambda: P & S)

print("\nOperations with no envelope, by theorem")
check("P ** 0.5             temper", lambda: P ** 0.5, want_ok=False)
check("P**.3 * R**.3        temper in product", lambda: P ** 0.3 * R ** 0.3, want_ok=False)
check("mean tau=0, temper-y", lambda: (P ** 0.5) ** 0.5, want_ok=False)


def _contrast():
    return P / R


try:
    _contrast()
    failures.append("P / R should refuse at construction")
    print("  FAIL  P / R                contrast        built, expected refusal")
except NoEnvelope as e:
    print(f"  ok    {'P / R                contrast':34s} refused at construction")

print("\nNormalization: exponents are collected before anything is decided")
check("P**.5 * P**.7  -> P**1.2", lambda: P ** 0.5 * P ** 0.7)
check("P**.3 * P**.3  -> P**0.6", lambda: P ** 0.3 * P ** 0.3, want_ok=False)
check("(P**.5 * R**.5)**2", lambda: (P ** 0.5 * R ** 0.5) ** 2)
check("(P * L)**2  sharpen a constrained model", lambda: constrain(P, _mask) ** 2)
check("(P * L)**0.5 still tempering", lambda: constrain(P, _mask) ** 0.5, want_ok=False)
check("constrain(P, L) * R  ensemble", lambda: constrain(P, _mask) * R)

print("\nWithin-model ensembles: one LM, two prompts")
llm = _FakeLLM("shared")
P1 = Model(llm, name="P", prompt="My favorite physicist is")
P2 = Model(llm, name="P", prompt="My favorite author is")
check("P[phys] & P[auth]", lambda: P1 & P2)
check("P[phys]**.5 * P[auth]**.5", lambda: P1 ** 0.5 * P2 ** 0.5)

print("\nExplanations")
for label, expr in [
    ("intersect", intersect(P, R)),
    ("agree", agree(P, R)),
    ("coverage", union(P, R)),
    ("max", mean([P, R], tau=math.inf)),
]:
    env = expr.envelope()
    print(f"\n  {label}:")
    for line in env.explain().splitlines():
        print(f"    {line}")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all algebra checks passed")
