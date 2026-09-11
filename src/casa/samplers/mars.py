"""Multi-model Adaptive Rejection Sampling.

MARS draws exact samples from any target that admits an envelope: combinations of language models
built with :mod:`casa.algebra`, including intersections, agreement and coverage ensembles,
sharpening, reweighting, and hard constraints.

It is CARS with one thing generalized. CARS stores, at each visited prefix, the model's next-token
distribution masked by grammar validity; the mask is 1 on valid tokens and 0 on invalid ones, and a
rejection sets an entry to 0 and propagates the shrunken mass to the parent. MARS stores the
*envelope ratio* ``log E(ua) - log E(u)`` instead of that masked distribution. Everything else, the
trie, the propagation, the telescoping argument, is unchanged, and a hard constraint makes the
ratio factor back into "model conditional times validity mask", which is CARS exactly.

Two consequences worth knowing. A rejection can now be partial: where CARS discovers an invalid
continuation by drawing it, MARS flips one coin at the moment a prefix is first expanded, with
survival probability equal to the total child mass over the parent's bound, and never draws the
token at all. And no rejection ever happens at a leaf, because the envelope is exact on complete
sequences, so once a descent reaches the end-of-sequence token the sample is accepted.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import torch

from casa import envelope_runtime as rt
from casa.algebra import Envelope, Potential
from casa.samplers.base import SamplingResult
from casa.utils.helpers import print_progress
from casa.utils.oracle_trie import Trie


class MARS:
    """Exact sampling from a combination of language models.

    Args:
        target: A :class:`casa.algebra.Envelope`, or a :class:`casa.algebra.Potential` which is
            validated on the spot. Passing a potential with no envelope raises ``NoEnvelope``,
            naming the obstruction.
        max_new_tokens: Length bound. Reaching it without an end-of-sequence token is a failure,
            not a sample, so the distribution is not truncated silently.
        verbose: Print a progress line per sample.
        temperature: Applied to every model's logits before the envelope is formed.
    """

    def __init__(self, target, max_new_tokens: int = 512, verbose: bool = False,
                 temperature: float = 1.0):
        self.envelope: Envelope = target.envelope() if isinstance(target, Potential) else target
        if not isinstance(self.envelope, Envelope):
            raise TypeError(f"expected an Envelope or Potential, got {type(target).__name__}")
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.temperature = temperature
        self.trie = Trie()
        self.stats = _Stats()
        self._key = None

    def reset(self) -> None:
        """Forget every bound learned so far and start from the original envelope."""
        self.trie = Trie()
        self.stats = _Stats()

    @property
    def root_mass(self) -> float:
        """The bound currently held at the root.

        Starts at the envelope's own value and never increases. The probability that a descent
        yields a sample is the normalizer over this, so a falling root mass is a rising acceptance
        rate; that is the monotonicity claim, in the form you can actually watch.
        """
        root = self.trie.root
        if root.raw_logprob is None:
            return 1.0
        return float(torch.exp(root.raw_logprob[0] + root.log_theta[0]).sum())

    # -- public ------------------------------------------------------------------------------

    def sample(self, prompt: str = "", n_samples: int = 1, max_attempts: int = 100,
               prompts: Optional[Dict[int, str]] = None) -> List[SamplingResult]:
        """Draw ``n_samples`` exact samples.

        Args:
            prompt: Prompt for every model that does not carry its own.
            n_samples: How many samples to return.
            max_attempts: Give up on a sample after this many rejected descents. Samples that are
                given up on are omitted, and ``stats.timeouts`` records how often that happened.
            prompts: Optional per-model prompt override, keyed by ``id`` of the algebra's ``Model``
                node. Use this to run one task instance at a time without rebuilding the target.

        Returns:
            The successful samples, in order.
        """
        resolved = prompts or self._default_prompts(prompt)

        # A fresh runtime per descent, exactly as CARS rebuilds its KV cache per attempt. Key-value
        # caches mutate in place, and a descent branches, so a runtime reused across descents would
        # hand the second descent a cache holding the first one's tokens. Nothing is lost by this:
        # the trie caches the envelope ratios themselves, so a revisited prefix costs no forward
        # pass at all, which is where the saving actually comes from.
        def fresh():
            return rt.build(self.envelope, prompts=resolved, temperature=self.temperature)

        # The trie belongs to the target, not to one call. Everything MARS learns about a prefix
        # stays valid for every later sample from the same target, and throwing it away between
        # calls would discard exactly the thing that makes the acceptance rate climb. It is reset
        # only when the conditioning changes, which makes the stored bounds meaningless.
        key = tuple(sorted(resolved.items()))
        if key != self._key:
            self.reset()
            self._key = key
        results: List[SamplingResult] = []

        for i in range(n_samples):
            attempts = 0
            got = None
            for _ in range(max_attempts):
                attempts += 1
                got = self._descend(fresh)
                if got is not None:
                    break
            if got is not None:
                got.attempts = attempts
                results.append(got)
                self.stats.accepted += 1
            else:
                self.stats.timeouts += 1
            print_progress(i + 1, n_samples, attempts, max_attempts, self.verbose,
                           timeout=got is None)
        return results

    def _default_prompts(self, prompt: str) -> Dict[int, str]:
        return {id(m): (m.prompt if m.prompt is not None else prompt)
                for m in self.envelope.models()}

    @property
    def acceptance_rate(self) -> float:
        """Fraction of descents that yielded a sample. Monotone non-decreasing over a run."""
        total = self.stats.descents
        return self.stats.accepted / total if total else 0.0

    # -- one descent -------------------------------------------------------------------------

    def _descend(self, fresh) -> Optional[SamplingResult]:
        t0 = time.time()
        self.stats.descents += 1

        base = fresh()
        state = base
        node = self.trie.root
        depth = 0
        context: List[int] = []
        pending = False

        for _ in range(self.max_new_tokens):
            if node.raw_logprob is None:
                # First visit: expand. The node's bound E(u) is replaced by the total mass of its
                # children, which can only be smaller, and the descent survives with exactly the
                # ratio between them. This is the only place a rejection can happen.
                ratios = state.log_ratios()
                node.raw_logprob = ratios.unsqueeze(0).cpu()
                node.log_theta = torch.zeros(1, ratios.shape[-1])
                self.stats.expansions += 1
                pending = True

                survival = float(torch.exp(ratios).sum())
                if not math.isfinite(survival) or survival <= 0.0:
                    self._propagate(node, depth, context)
                    self.stats.dead_ends += 1
                    self.stats.seconds += time.time() - t0
                    return None
                if survival < 1.0 and torch.rand(()).item() > survival:
                    self._propagate(node, depth, context)
                    self.stats.rejections += 1
                    self.stats.rejected_mass += 1.0 - survival
                    self.stats.seconds += time.time() - t0
                    return None

            bounds = node.raw_logprob[0] + node.log_theta[0]
            finite = torch.isfinite(bounds)
            if not finite.any():
                self._propagate(node, depth, context)
                self.stats.dead_ends += 1
                self.stats.seconds += time.time() - t0
                return None

            token = int(torch.multinomial(torch.softmax(bounds, dim=-1), 1))

            if token == state.eos_token_id:
                # The envelope is exact on complete sequences, so there is nothing left to reject
                # unless the target needed a dominating envelope in the first place.
                if self.envelope.needs_leaf_rejection:
                    log_acc = state.log_leaf_acceptance(token)
                    self.stats.leaf_trials += 1
                    if log_acc < 0.0 and math.log(max(torch.rand(()).item(), 1e-300)) > log_acc:
                        self.stats.leaf_rejections += 1
                        self.stats.seconds += time.time() - t0
                        return None
                if pending:
                    self._propagate(node, depth, context)
                self.stats.seconds += time.time() - t0
                return self._result(base, context, state)

            context.append(token)
            state = state.advance(token)
            if token not in node.children:
                node.create_child(token)
            node = node.children[token]
            depth += 1

        # Ran out of length without terminating. Not a sample.
        if pending:
            self._propagate(node, depth, context)
        self.stats.length_cutoffs += 1
        self.stats.seconds += time.time() - t0
        return None

    def _propagate(self, node, depth: int, context: List[int]) -> None:
        """Replace each ancestor's bound on this path by the total mass of its children.

        Identical to the CARS update, and correct for the same reason: the overwrite is valid
        because ``raw_logprob`` already holds the full envelope ratio, so the parent's entry for
        this child is its bound and nothing else.
        """
        while depth > 0:
            total = torch.log(torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum())
            depth -= 1
            node = node.parent
            node.log_theta[0, context[depth]] = total

    def _result(self, base: "rt.EnvelopeRT", context: List[int],
                final: "rt.EnvelopeRT") -> SamplingResult:
        tok = base.models[0].llm.tokenizer
        ids = context + [base.eos_token_id]
        return SamplingResult(
            tokens=[tok.decode([t]) for t in ids],
            token_ids=ids,
            text=tok.decode(context),
            raw_logprob=float(final.log_weight()),
            constrained_logprob=float(final.log_weight() - base.log_weight()),
            success=True,
        )


class _Stats:
    """What a run cost, in the quantities the theory talks about."""

    __slots__ = ("descents", "accepted", "expansions", "rejections", "rejected_mass",
                 "leaf_trials", "leaf_rejections", "dead_ends", "length_cutoffs",
                 "timeouts", "seconds")

    def __init__(self):
        self.descents = 0
        self.accepted = 0
        self.expansions = 0
        self.rejections = 0
        self.rejected_mass = 0.0
        self.leaf_trials = 0
        self.leaf_rejections = 0
        self.dead_ends = 0
        self.length_cutoffs = 0
        self.timeouts = 0
        self.seconds = 0.0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:
        return f"_Stats({self.as_dict()})"
