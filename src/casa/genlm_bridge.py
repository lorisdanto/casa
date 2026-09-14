"""Expose a CASA envelope as a genlm-control ``Potential``.

The MARS paper is measured against Chan et al., "Ensembling Language Models with Sequential Monte
Carlo". Their own repository is a 17-byte empty README, but Tim Vieira is a co-author and his lab
ships the inference machinery as ``genlm-control``, which is almost certainly what they ran. So the
baseline runs *their* SMC rather than a rewrite of it, and the only thing we supply is the target.

That matters for the comparison. Reimplementing someone's method invites the obvious objection that
any gap is a defect in the rewrite. Here the sampler is theirs, unmodified, and the potential it
samples is the identical envelope MARS uses, built by the same algebra from the same models. The
two methods differ in how they use the bounds and in nothing else.

The interfaces line up almost exactly, which is not a coincidence: a prefix weight and a next-token
weight vector are what any sequential method needs.

    genlm Potential          CASA envelope
    prefix(context)          log E(u)
    complete(context)        log E(w$), which condition (i) makes the target's own weight
    logw_next(context)       log E(ua) - log E(u) for every token, plus EOS

Usage::

    from casa.algebra import Model, intersect
    from casa.genlm_bridge import EnvelopePotential

    target = intersect(Model(llm, prompt=a), Model(llm, prompt=b)).envelope()
    particles = await smc_sample(target, prompts={...}, n_particles=10, max_tokens=64)

``smc_sample`` is a thin wrapper: it wraps the envelope as a potential, hands that to their
``DirectTokenSampler``, and runs their ``SMC``. Nothing of the method is reimplemented.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import torch

from casa import envelope_runtime as rt
from casa.algebra import Envelope, Potential as AlgebraPotential

NEG_INF = float("-inf")

# Stand-in for -inf on the end-of-sequence entry of a dead-end prefix; see `vocab_eos_weights`.
DEAD_END_LOG_WEIGHT = -1e300


def vocab_eos_weights(ratios: torch.Tensor, eos: int) -> torch.Tensor:
    """A ratio vector in genlm's ``vocab_eos`` order, safe for its sampler.

    genlm orders every token except EOS by id, then EOS last. A particle can reach a prefix where
    nothing may follow: every grammar-allowed continuation has zero target weight, and so does
    stopping. All entries are then -inf, genlm's normalization turns them into NaN, and its
    Gumbel-max draw over NaN returns index 0 every time. The particle is extended by token 0,
    which the grammar rejects, and the run dies with "rejected a prefix MARS had already
    accepted" -- four CARS grammar rows were lost this way.

    The particle's weight is zero either way; what matters is that it ends rather than walking on.
    So such a vector puts a finite but negligible weight on EOS alone: genlm stops the particle
    with log weight ``DEAD_END_LOG_WEIGHT``, which contributes nothing next to any live particle
    and is dropped at resampling. Every other vector is returned unchanged.
    """
    ws = torch.cat([ratios[:eos], ratios[eos + 1:], ratios[eos: eos + 1]]).to(torch.float64)
    if not torch.isfinite(ws).any():
        ws[-1] = DEAD_END_LOG_WEIGHT
    return ws


def _require_genlm():
    try:
        from genlm.control.potential.base import Potential  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment, not logic
        raise SystemExit(
            f"the SMC baseline needs genlm-control ({e}). Install it with\n"
            "  pip install genlm-control\n"
            "It is the inference code from the lab that wrote the paper we compare against."
        )
    return Potential


class EnvelopePotential:
    """A CASA envelope, presented as the potential genlm-control's SMC expects.

    Instances are cheap to build but hold a plan, so build one per task instance and reuse it
    across the particles of a sweep.

    Args:
        envelope: A validated :class:`casa.algebra.Envelope`.
        prompts: Per-model prompt override, keyed by ``id`` of the algebra's ``Model`` node.
        temperature: Applied to every model's logits before the envelope is formed.
    """

    def __new__(cls, envelope, prompts=None, temperature: float = 1.0,
                cache_max: int = 64):
        # genlm's Potential is an abstract base with a required constructor, so the concrete class
        # is built at first use rather than at import. That keeps `import casa` working on a
        # machine without genlm-control, which is every machine that only runs MARS.
        base = _require_genlm()
        impl = _make_impl(base)
        return impl(envelope, prompts, temperature, cache_max)


_IMPL = None


def _make_impl(base):
    global _IMPL
    if _IMPL is not None:
        return _IMPL

    class _EnvelopePotential(base):
        def __init__(self, envelope, prompts=None, temperature=1.0, cache_max=64):
            if isinstance(envelope, AlgebraPotential):
                envelope = envelope.envelope()
            if not isinstance(envelope, Envelope):
                raise TypeError(f"expected an Envelope, got {type(envelope).__name__}")
            self.envelope = envelope
            self._plan = rt.plan(envelope, prompts=prompts, temperature=temperature)
            self._eos = self._plan.eos_token_id
            self._V = self._plan.vocab_size
            # genlm holds EOS out of the vocabulary and appends its own sentinel, so vocab_eos is
            # vocab + [eos] with eos last. Their own LLM potential passes no eos argument and takes
            # the default sentinel; passing a raw token id here instead produces a vocabulary with
            # two different notions of "end" and weights that land on the wrong entries.
            vocab = [i for i in range(self._V) if i != self._eos]
            super().__init__(vocabulary=vocab)
            # Bounded. Each cached runtime holds a materialized next-token vector per model, which
            # is about 600 KB at a 150k vocabulary, so an unbounded cache over particles times
            # sequence length runs into hundreds of megabytes and climbs with both. Particles share
            # prefixes, so a small cache still catches nearly all the reuse.
            self._cache: "OrderedDict[tuple, object]" = OrderedDict()
            # ``cache_max=0`` removes the bound, which is the variant the paper needs in order
            # to separate SMC's own cost from the cap we imposed on it.
            self._cache_max = cache_max
            # The bound on this cache is a choice we made on the baseline's behalf, and it counts
            # against SMC: every eviction is a prefix that has to be walked again from scratch.
            # The paper has to say how much that costs rather than assert it is small, so the
            # hits, misses and evictions are counted and reported with the run.
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            # Particles stopped at a prefix with no valid continuation (see vocab_eos_weights).
            self._dead_ends = 0
            # One counter across every runtime this potential builds, so the cost reported for the
            # baseline is what it actually spent rather than what its last prefix spent.
            self._counters: Dict[int, Dict[str, int]] = {}

        # -- positioning -------------------------------------------------------------------

        def _at(self, context: Sequence[int]):
            """The runtime positioned at ``context``, cached so particles sharing a prefix share
            the work, which is the same saving the MARS trie gets.

            Their samplers append an end-of-sequence sentinel to a finished sequence and may then
            ask about it. The sentinel is not a token our models can be advanced by, so stop at it:
            the weight of a sequence and the weight of that sequence plus the marker are the same
            thing to us, since condition (i) already accounts for termination.
            """
            key = tuple(t for t in context if isinstance(t, int))
            got = self._cache.get(key)
            if got is not None:
                self._hits += 1
                self._cache.move_to_end(key)
                return got
            self._misses += 1
            got = self._plan.fresh(counters=self._counters)
            for t in key:
                got = got.advance(t)
            self._cache[key] = got
            if self._cache_max and len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
                self._evictions += 1
            return got

        # -- the interface SMC consumes ----------------------------------------------------

        async def prefix(self, context) -> float:
            return float(self._at(context).log_weight())

        async def complete(self, context) -> float:
            """Weight of a finished sequence.

            Condition (i) makes the envelope exact here, so this is the target's own unnormalized
            weight, not a bound on it.
            """
            st = self._at(context)
            w = st.log_weight() + float(st.log_ratios()[self._eos])
            if self.envelope.needs_leaf_rejection:
                w += st.log_leaf_acceptance(self._eos)
            return float(w)

        async def logw_next(self, context):
            """Log weight of every next token, and of stopping here.

            This is the ratio vector a MARS trie node stores. Condition (ii) says it exponentiates
            to at most one, and the shortfall is the rejection mass MARS pays; SMC folds the same
            quantity into an importance weight instead.
            """
            ws = vocab_eos_weights(self._at(context).log_ratios(), self._eos)
            if ws[-1] == DEAD_END_LOG_WEIGHT:
                self._dead_ends += 1
            return self.make_lazy_weights(ws.cpu().numpy())

        @property
        def cache_stats(self) -> Dict[str, float]:
            """How well the bounded prefix cache served this run.

            ``eviction_rate`` is evictions per lookup. A rate near zero means the bound never
            bound, and the cost reported for SMC is what an unbounded cache would have given; a
            high rate means part of SMC's measured cost is our cap rather than its method.
            """
            looks = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "lookups": looks,
                "hit_rate": (self._hits / looks) if looks else float("nan"),
                "eviction_rate": (self._evictions / looks) if looks else float("nan"),
                "capacity": self._cache_max or None,
                "high_water": len(self._cache),
                # Not a cache figure, but reported with the run through the same dict: how many
                # particles were stopped at a dead end rather than left to crash the sweep.
                "dead_ends": self._dead_ends,
            }

        @property
        def model_calls(self) -> int:
            """Total forward passes across every prefix this potential has been asked about."""
            return sum(c["passes"] for c in self._counters.values())

        def __repr__(self):
            return f"EnvelopePotential({self.envelope.expr!r})"

    _IMPL = _EnvelopePotential
    return _IMPL


# --------------------------------------------------------------------------------------------
# Running their SMC over our envelope
# --------------------------------------------------------------------------------------------


async def smc_sample(envelope, prompts=None, temperature: float = 1.0, n_particles: int = 10,
                     ess_threshold: float = 0.9, max_tokens: int = 64, cache_max: int = 64,
                     **kwargs):
    """Run genlm-control's SMC over a CASA envelope, unmodified.

    ``DirectTokenSampler`` with no proposal is precisely the paper's locally optimal proposal: it
    samples the next token proportionally to the potential's next-token weights and returns the
    log normalizer as the incremental weight. That normalizer is the same quantity MARS rejects
    with, so the two methods are reading one number two different ways.

    Args:
        envelope: A validated :class:`casa.algebra.Envelope`, or a ``Potential`` to validate.
        prompts: Per-model prompt override, keyed by ``id`` of the algebra's ``Model`` node.
        temperature: Applied to every model's logits.
        n_particles: Chan et al. use 10.
        ess_threshold: Resampling threshold as a fraction of the particle count; they use 0.9.
        max_tokens: Length bound. Their SMC stops a particle here and corrects its weight by the
            end-of-sequence entry, which is the same truncated target MARS samples.

    Returns:
        The particle set their SMC returns, carrying sequences and log weights.
    """
    from genlm.control.sampler.token import DirectTokenSampler

    potential = EnvelopePotential(envelope, prompts=prompts, temperature=temperature,
                                  cache_max=cache_max)
    sampler = DirectTokenSampler(potential)
    try:
        return await sampler.smc(n_particles=n_particles, ess_threshold=ess_threshold,
                                 max_tokens=max_tokens, **kwargs)
    finally:
        await sampler.cleanup()
