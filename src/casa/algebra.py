"""An algebra of envelopes.

MARS samples exactly from any target that admits an *envelope*: a function on prefixes that is
exact on complete sequences and never promises less for a prefix than for its children combined,

    (i)  E(w$) = phi(w$)                     for every complete sequence
    (ii) E(u) >= sum_a E(ua)                 for every prefix

This module lets you *write* a target as an expression and get an envelope for it, or a refusal
explaining that no envelope exists. Expressions are built with operators that mirror the maths:

    P, R = Model(llm_a), Model(llm_b)

    P * R                 product of experts          E = E_p * E_r
    P ** 0.5 * R ** 0.5   geometric mean, intersect   E = sqrt(E_p * E_r)
    P & R                 agreement, minimum          E = min(E_p, E_r)
    P | R                 union, mixture              E = (E_p + E_r) / 2
    P ** 2                sharpening                  E = E_p ** 2
    P * mask              hard constraint             E = E_p * 1[u in prefix(L)]
    mean([P, R], tau)     generalized mean            the whole family at once

Building an expression never checks anything. Validation happens once, at the root, in
``envelope()``. Two operations are refused there because no envelope exists for them at all, and
no exact sampler with next-token access can handle them without exponentially many model calls:

    P ** 0.5              tempering, on its own       refused
    P / R                 contrasting                 refused

Tempering is refused only *on its own*. Inside a product whose exponents sum to at least one it is
perfectly fine, because the product is what restores condition (ii). That is why a fractional power
does not raise when you write it: the expression tree is normalized first, and only the collected
exponents decide. ``(P ** 0.5) * (R ** 0.5)`` is an envelope; ``P ** 0.5`` alone is not.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------


class NoEnvelope(Exception):
    """Raised when an expression admits no envelope.

    This is not a limitation of the implementation. For tempering and contrasting it is a theorem:
    every exact sampler with next-token access needs exponentially many model calls.
    """


@dataclass(frozen=True)
class Verdict:
    """The outcome of validating an expression.

    Attributes:
        ok: Whether an envelope exists.
        reason: Human-readable justification, citing the closure item or the obstruction.
        leaf_acceptance: If an envelope exists only with a final acceptance step, the guaranteed
            lower bound on that acceptance probability. ``None`` when no leaf step is needed.
    """

    ok: bool
    reason: str
    leaf_acceptance: Optional[float] = None

    def raise_if_bad(self) -> "Verdict":
        if not self.ok:
            raise NoEnvelope(self.reason)
        return self


# --------------------------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------------------------


class Potential(ABC):
    """A node in an envelope expression.

    A ``Potential`` is syntax, not a promise: it may or may not admit an envelope. Ask
    ``.verdict()`` for the answer and ``.envelope()`` for the thing MARS can actually sample.
    """

    # -- algebra ------------------------------------------------------------------------------

    def __mul__(self, other: "Potential | Scorer") -> "Potential":
        return Product.of(self, _coerce(other))

    def __rmul__(self, other: "Potential | Scorer") -> "Potential":
        return Product.of(_coerce(other), self)

    def __pow__(self, gamma: float) -> "Potential":
        if gamma <= 0:
            raise ValueError(f"exponent must be positive, got {gamma}")
        return Power(self, float(gamma))

    def __and__(self, other: "Potential") -> "Potential":
        """Agreement. ``P & R`` is the minimum, the generalized mean at tau = -infinity."""
        return Mean.of([self, _coerce(other)], tau=-math.inf)

    def __or__(self, other: "Potential") -> "Potential":
        """Union. ``P | R`` is the equal-weight mixture, the generalized mean at tau = 1."""
        return Mean.of([self, _coerce(other)], tau=1.0)

    def __truediv__(self, other: "Potential") -> "Potential":
        raise NoEnvelope(
            "contrasting (a quotient of models) admits no envelope: the ratio is unbounded "
            "wherever the denominator is small, so condition (ii) fails and the target need not "
            "be normalizable. No exact sampler with next-token access can do better."
        )

    # -- validation ---------------------------------------------------------------------------

    @abstractmethod
    def verdict(self) -> Verdict:
        """Decide whether this expression admits an envelope, and say why."""

    def envelope(self) -> "Envelope":
        """Validate and wrap as an envelope MARS can sample. Raises ``NoEnvelope`` if it cannot."""
        v = self.verdict().raise_if_bad()
        return Envelope(self, v)

    @abstractmethod
    def leaves(self) -> List["Model"]:
        """Every base model appearing in this expression, in order, with duplicates."""


def _coerce(x) -> Potential:
    if isinstance(x, Potential):
        return x
    raise TypeError(f"expected a Potential, got {type(x).__name__}")


@dataclass(frozen=True)
class Model(Potential):
    """A base potential: a language model, or any sub-stochastic next-token weighting.

    The prefix weight of a sub-stochastic potential is an envelope for it, since condition (ii)
    reads ``p(u) >= p(u) * sum_a p(a|u)``. For a language model this holds with equality.

    Attributes:
        llm: The CASA ``LLM`` this wraps.
        name: Label used in diagnostics and in the canonical form.
        prompt: Optional per-model prompt. Two ``Model`` nodes over the same ``llm`` with different
            prompts are the within-model ensemble setting: one model, two conditionings.
        sub_stochastic: Whether the weights at a prefix are guaranteed to sum to at most one.
            True for language models. A potential that is not sub-stochastic can still be used,
            but only under an exponent of at least one.
    """

    llm: object
    name: str = "p"
    prompt: Optional[str] = None
    sub_stochastic: bool = True

    def verdict(self) -> Verdict:
        return Verdict(True, f"{self.name}: prefix weight of a sub-stochastic potential")

    def leaves(self) -> List["Model"]:
        return [self]


@dataclass(frozen=True)
class Power(Potential):
    """``base ** gamma``.

    An envelope on its own only when ``gamma >= 1`` (sharpening). A fractional exponent is
    tempering, which has no envelope alone, but survives inside a product whose exponents sum to
    at least one. Normalization collects exponents before anything is decided, so this node never
    refuses by itself.
    """

    base: Potential
    gamma: float

    def verdict(self) -> Verdict:
        return Product.of(self).verdict()

    def leaves(self) -> List[Model]:
        return self.base.leaves()


@dataclass(frozen=True)
class Scorer(Potential):
    """A weighting of sequences by values in [0, 1].

    Two kinds, distinguished by ``prefix_monotone``:

    * prefix-monotone (``g(ua) <= g(u)``): an envelope factor outright, closure item (3). Hard
      constraints are the special case where ``g`` is an indicator of the prefix language.
    * output-only: reveals its weight only at complete sequences. MARS still samples exactly, by
      running on the envelope that ignores the scorer and accepting a yielded sequence with
      probability equal to its score. That is leaf rejection.

    Attributes:
        fn: Maps a prefix (token id sequence) to a weight in [0, 1]. For output-only scorers this
            is called only on complete sequences.
        name: Label used in diagnostics.
        prefix_monotone: Whether ``fn`` is non-increasing along prefixes.
    """

    fn: Callable[[Sequence[int]], float]
    name: str = "g"
    prefix_monotone: bool = True

    def verdict(self) -> Verdict:
        if self.prefix_monotone:
            return Verdict(True, f"{self.name}: prefix-monotone scorer, closure item (3)")
        return Verdict(True, f"{self.name}: output-only scorer, sampled with leaf rejection", 0.0)

    def leaves(self) -> List[Model]:
        return []


@dataclass(frozen=True)
class Product(Potential):
    """A weighted product ``prod_k base_k ** gamma_k``.

    An envelope when every exponent is positive and they sum to at least one, which is the
    generalized Hoelder inequality when they sum to exactly one and reduces to it otherwise.
    Prefix-monotone scorer factors ride along freely and do not count toward the exponent sum.

    Build these with ``*`` and ``**`` rather than directly; ``of`` flattens and collects so that
    ``(P ** 0.5) * (P ** 0.7)`` is recognized as ``P ** 1.2`` and therefore an envelope.
    """

    terms: Tuple[Tuple[Potential, float], ...]
    scorers: Tuple[Scorer, ...] = ()

    @staticmethod
    def of(*factors: Potential) -> "Product":
        terms: List[Tuple[Potential, float]] = []
        scorers: List[Scorer] = []

        def absorb(node: Potential, gamma: float) -> None:
            if isinstance(node, Power):
                absorb(node.base, gamma * node.gamma)
            elif isinstance(node, Product):
                for sub, g in node.terms:
                    absorb(sub, gamma * g)
                for s in node.scorers:
                    if gamma != 1.0:
                        raise NoEnvelope(
                            f"scorer {s.name!r} raised to the power {gamma}: a scorer under an "
                            "exponent is not covered by the closure proposition"
                        )
                    scorers.append(s)
            elif isinstance(node, Scorer):
                if gamma != 1.0:
                    raise NoEnvelope(
                        f"scorer {node.name!r} raised to the power {gamma}: a scorer under an "
                        "exponent is not covered by the closure proposition"
                    )
                scorers.append(node)
            else:
                terms.append((node, gamma))

        for f in factors:
            absorb(f, 1.0)

        # Collect repeated bases so that P**0.5 * P**0.7 becomes P**1.2.
        collected: Dict[int, Tuple[Potential, float]] = {}
        order: List[int] = []
        for node, gamma in terms:
            key = id(node) if not isinstance(node, Model) else hash((node.name, id(node.llm), node.prompt))
            if key in collected:
                base, g = collected[key]
                collected[key] = (base, g + gamma)
            else:
                collected[key] = (node, gamma)
                order.append(key)
        return Product(tuple(collected[k] for k in order), tuple(scorers))

    @property
    def exponent_sum(self) -> float:
        return sum(g for _, g in self.terms)

    def verdict(self) -> Verdict:
        if not self.terms:
            return Verdict(True, "empty product, constant envelope")

        for base, gamma in self.terms:
            if gamma <= 0:
                return Verdict(
                    False,
                    f"exponent {gamma} on {_label(base)} is not positive; envelopes are not closed "
                    "under quotients",
                )
            sub = base.verdict()
            if not sub.ok:
                return sub

        total = self.exponent_sum
        if total < 1.0 - 1e-12:
            offenders = ", ".join(f"{_label(b)}**{g:g}" for b, g in self.terms)
            return Verdict(
                False,
                f"exponents sum to {total:g} < 1 ({offenders}). This is tempering: "
                "sum_a p(a|u)**gamma can exceed 1, so condition (ii) fails and no envelope of this "
                "form exists. Raise an exponent, or intersect with another model so the exponents "
                "sum to at least one.",
            )

        # A potential that is not sub-stochastic only behaves under an exponent of at least one.
        for base, gamma in self.terms:
            if isinstance(base, Model) and not base.sub_stochastic and gamma < 1.0 - 1e-12:
                return Verdict(
                    False,
                    f"{base.name} is not sub-stochastic, so it needs an exponent of at least 1; "
                    f"got {gamma:g}",
                )

        leaf = None
        for s in self.scorers:
            sv = s.verdict()
            if sv.leaf_acceptance is not None:
                leaf = 0.0
        how = "Hoelder" if abs(total - 1.0) < 1e-12 else f"Hoelder after scaling (exponents sum to {total:g})"
        extra = f" with {len(self.scorers)} scorer factor(s)" if self.scorers else ""
        return Verdict(True, f"weighted product, closure item (1) by {how}{extra}", leaf)

    def leaves(self) -> List[Model]:
        out: List[Model] = []
        for base, _ in self.terms:
            out.extend(base.leaves())
        return out


@dataclass(frozen=True)
class Mean(Potential):
    """The generalized mean ``M_tau`` of several potentials, with weights in the simplex.

    This is the family Chan et al. characterize as the unique minimizers of weighted sums of
    alpha-divergences, with ``tau = 1 - alpha``. The whole family is here:

        tau -> -inf   minimum        agreement
        tau = -1      harmonic       agreement
        tau -> 0      product        agreement, and the same thing as a weighted product
        tau = 1       mixture        coverage
        tau = 2       quadratic      coverage
        tau -> +inf   maximum        coverage

    For ``tau <= 1`` the mean is concave and positively homogeneous, hence superadditive, so the
    mean of envelopes is an envelope and MARS samples exactly with no leaf step. For ``tau > 1`` it
    is subadditive and that fails; the mixture ``c * sum_k w_k E_k`` with ``c = 1 / min_k w_k``
    dominates the target at complete sequences instead, so MARS samples exactly with a leaf
    acceptance of at least ``1 / c``.
    """

    terms: Tuple[Potential, ...]
    weights: Tuple[float, ...]
    tau: float

    @staticmethod
    def of(terms: Sequence[Potential], tau: float, weights: Optional[Sequence[float]] = None) -> Potential:
        terms = tuple(terms)
        if len(terms) < 1:
            raise ValueError("a mean needs at least one term")
        if weights is None:
            weights = tuple(1.0 / len(terms) for _ in terms)
        else:
            weights = tuple(float(w) for w in weights)
            if len(weights) != len(terms):
                raise ValueError("weights and terms must have the same length")
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative")
            total = sum(weights)
            if not math.isclose(total, 1.0, rel_tol=1e-9):
                raise ValueError(f"weights must sum to 1, got {total}")
        # tau = 0 is the weighted geometric mean, which the product node already handles exactly.
        if tau == 0.0:
            return Product.of(*[t ** w for t, w in zip(terms, weights)])
        return Mean(terms, weights, float(tau))

    @property
    def needs_leaf_rejection(self) -> bool:
        return self.tau > 1.0

    @property
    def leaf_acceptance(self) -> float:
        """Guaranteed lower bound on the leaf acceptance probability, ``1 / c``."""
        return min(self.weights)

    def verdict(self) -> Verdict:
        for t in self.terms:
            sub = t.verdict()
            if not sub.ok:
                return sub
        if self.tau <= 1.0:
            return Verdict(
                True,
                f"generalized mean at tau={_tau(self.tau)} is concave and positively homogeneous, "
                "hence superadditive, so the mean of envelopes is an envelope",
            )
        return Verdict(
            True,
            f"generalized mean at tau={_tau(self.tau)} is subadditive, so the mean of envelopes is "
            f"not one; MARS runs on the dominating mixture with leaf acceptance at least "
            f"{self.leaf_acceptance:g}",
            self.leaf_acceptance,
        )

    def leaves(self) -> List[Model]:
        out: List[Model] = []
        for t in self.terms:
            out.extend(t.leaves())
        return out


# --------------------------------------------------------------------------------------------
# The validated result
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Envelope:
    """A target that MARS can sample exactly, together with why it can.

    Obtained from ``Potential.envelope()``. Holding one of these is a certificate: the expression
    was normalized and checked against the closure proposition, so an envelope exists.
    """

    expr: Potential
    verdict: Verdict

    @property
    def needs_leaf_rejection(self) -> bool:
        return self.verdict.leaf_acceptance is not None

    @property
    def leaf_acceptance(self) -> Optional[float]:
        return self.verdict.leaf_acceptance

    def models(self) -> List[Model]:
        return self.expr.leaves()

    def explain(self) -> str:
        lines = [f"target: {_label(self.expr)}", f"envelope: {self.verdict.reason}"]
        if self.needs_leaf_rejection:
            lines.append(f"leaf acceptance: at least {self.leaf_acceptance:g}")
        seen = {id(m.llm) for m in self.models()}
        lines.append(f"models: {len(self.models())} factor(s) over {len(seen)} distinct LM(s)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.explain()


# --------------------------------------------------------------------------------------------
# Sugar
# --------------------------------------------------------------------------------------------


def mean(terms: Sequence[Potential], tau: float, weights: Optional[Sequence[float]] = None) -> Potential:
    """The generalized mean of ``terms`` at power ``tau``. See :class:`Mean`."""
    return Mean.of(terms, tau, weights)


def intersect(*terms: Potential, weights: Optional[Sequence[float]] = None) -> Potential:
    """Weighted geometric mean, the product-of-experts target. ``tau = 0``."""
    if weights is None:
        weights = [1.0 / len(terms)] * len(terms)
    return Product.of(*[t ** w for t, w in zip(terms, weights)])


def agree(*terms: Potential) -> Potential:
    """Minimum, the strictest consensus. ``tau = -infinity``."""
    return Mean.of(terms, tau=-math.inf)


def union(*terms: Potential, weights: Optional[Sequence[float]] = None) -> Potential:
    """Mixture. ``tau = 1``."""
    return Mean.of(terms, tau=1.0, weights=weights)


def constrain(p: Potential, recognizer, name: str = "L") -> Potential:
    """Restrict ``p`` to a prefix-closed language. This is exactly the CARS target.

    Args:
        p: The potential to constrain.
        recognizer: An object exposing ``accepts_prefix(token_ids) -> bool``.
        name: Label for diagnostics.
    """
    return p * Scorer(
        fn=lambda ctx: 1.0 if recognizer.accepts_prefix(ctx) else 0.0,
        name=name,
        prefix_monotone=True,
    )


def reweight(p: Potential, fn: Callable[[Sequence[int]], float], name: str = "g",
             prefix_monotone: bool = False) -> Potential:
    """Reweight ``p`` by a verifier or reward model. See :class:`Scorer`."""
    return p * Scorer(fn=fn, name=name, prefix_monotone=prefix_monotone)


def sharpen(p: Potential, gamma: float) -> Potential:
    """Raise ``p`` to a power. Envelopes exist only for ``gamma >= 1``; below that it is tempering."""
    return p ** gamma


# --------------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------------


def _tau(t: float) -> str:
    if t == -math.inf:
        return "-inf"
    if t == math.inf:
        return "+inf"
    return f"{t:g}"


def _label(node: Potential) -> str:
    if isinstance(node, Model):
        return node.name if node.prompt is None else f"{node.name}[{_short(node.prompt)}]"
    if isinstance(node, Scorer):
        return node.name
    if isinstance(node, Power):
        return f"{_label(node.base)}**{node.gamma:g}"
    if isinstance(node, Product):
        parts = [f"{_label(b)}**{g:g}" if g != 1.0 else _label(b) for b, g in node.terms]
        parts += [_label(s) for s in node.scorers]
        return " * ".join(parts) if parts else "1"
    if isinstance(node, Mean):
        inner = ", ".join(_label(t) for t in node.terms)
        if node.tau == -math.inf:
            return f"min({inner})"
        if node.tau == math.inf:
            return f"max({inner})"
        if node.tau == 1.0:
            return f"mix({inner})"
        return f"M[tau={_tau(node.tau)}]({inner})"
    return node.__class__.__name__.lower()


def _short(s: str, n: int = 18) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"
