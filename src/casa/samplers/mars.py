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

NEG_INF = float("-inf")

#: How far above one a prefix's child mass may sit before it counts as a real violation rather
#: than arithmetic. Summing a float32 log-softmax over a 150k vocabulary is off by around 5e-5 in
#: either direction, fifty times a 1e-6 tolerance, so a tight threshold rejects every ordinary
#: model on its first expansion. A genuine violation is a factor, not a fifth of a permille.
_LOG_SLACK = math.log1p(1e-3)


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
        return float(torch.exp(torch.logsumexp(root.raw_logprob[0] + root.log_theta[0], dim=0)))

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
        # Merge, do not replace. A caller overriding one model's prompt must not silently blank
        # every other model's, which is what taking `prompts` wholesale used to do.
        resolved = {**self._default_prompts(prompt), **(prompts or {})}

        # Validation and prompt tokenization happen once, in the plan. Each descent then gets a
        # fresh runtime, exactly as CARS rebuilds its cache per attempt: a descent branches and
        # key-value caches mutate in place, so reusing one across descents would hand the second
        # descent a cache holding the first one's tokens. Nothing is lost, because the trie caches
        # the envelope ratios themselves and a revisited prefix costs no forward pass at all.
        plan = rt.plan(self.envelope, prompts=resolved, temperature=self.temperature)
        fresh = plan.fresh

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
        """Fraction of descents so far that yielded a sample.

        A cumulative ratio, so it is not itself monotone; the quantity the theorem is about is
        :attr:`root_mass`, which never increases.
        """
        total = self.stats.descents
        return self.stats.accepted / total if total else 0.0

    # -- one descent -------------------------------------------------------------------------

    def _descend(self, fresh) -> Optional[SamplingResult]:
        t0 = time.time()
        self.stats.descents += 1
        base = fresh()
        try:
            return self._walk(base)
        finally:
            # Recorded once, here, rather than at each of the seven ways a descent can end.
            self.stats.seconds += time.time() - t0
            self.stats.model_calls += base.forward_passes()

    def _walk(self, base) -> Optional[SamplingResult]:
        state = base
        node = self.trie.root
        depth = 0
        context: List[int] = []
        pending = False
        # raw_logprob holds the pristine envelope ratio and is never rewritten (only log_theta is),
        # so summing the chosen entries telescopes to log E(w$) - log E(root) for free.
        log_ratio_sum = 0.0

        for _ in range(self.max_new_tokens):
            if node.raw_logprob is None:
                # First visit: expand. The node's bound E(u) is replaced by the total mass of its
                # children, which can only be smaller, and the descent survives with exactly the
                # ratio between them. This is the only place a rejection can happen.
                ratios = state.log_ratios()

                # In log space throughout. Exponentiating first underflows to exactly zero below
                # about -104 in float32, and a zero total is indistinguishable from an impossible
                # prefix, so a live subtree would be deleted from the support without a word.
                log_survival = float(torch.logsumexp(ratios.double(), dim=0))
                # Checked before anything is written. Raising after the node holds a bound but
                # before that bound reaches its parent would leave a trie that is quietly wrong,
                # so a caller who caught this and carried on would get biased samples forever.
                if log_survival > _LOG_SLACK:
                    raise ValueError(
                        f"envelope condition (ii) violated at depth {depth}: the children of this "
                        f"prefix carry {math.exp(log_survival):.9f} of its bound, which exceeds "
                        "one. The expression is not an envelope for the target and sampling from "
                        "it would not be exact."
                    )

                node.raw_logprob = ratios.unsqueeze(0).cpu()
                node.log_theta = torch.zeros(1, ratios.shape[-1])
                # The gap between envelope and target at this prefix is a function of the prefix,
                # so compute it now, while the models are materialized, rather than forcing a
                # forward pass every time a cached descent happens to end here.
                node.leaf_gap = (state.log_leaf_acceptance(state.eos_token_id)
                                 if self.envelope.needs_leaf_rejection else 0.0)
                self.stats.expansions += 1
                pending = True
                if math.isnan(log_survival) or log_survival == NEG_INF:
                    self._propagate(node, depth, context)
                    self.stats.dead_ends += 1
                    return None
                if log_survival < 0.0 and math.log(max(torch.rand(()).item(), 1e-300)) > log_survival:
                    self._propagate(node, depth, context)
                    self.stats.rejections += 1
                    self.stats.rejected_mass += max(0.0, 1.0 - math.exp(log_survival))
                    return None

            bounds = node.raw_logprob[0] + node.log_theta[0]
            finite = torch.isfinite(bounds)
            if not finite.any():
                self._propagate(node, depth, context)
                self.stats.dead_ends += 1
                return None

            token = int(torch.multinomial(torch.softmax(bounds, dim=-1), 1))

            if token == state.eos_token_id:
                # The envelope is exact on complete sequences, so there is nothing left to reject
                # unless the target needed a dominating envelope in the first place.
                log_acc = 0.0
                if self.envelope.needs_leaf_rejection:
                    log_acc = getattr(node, "leaf_gap", None)
                    if log_acc is None:  # node predates the cache; pay for it once
                        log_acc = state.log_leaf_acceptance(token)
                    self.stats.leaf_trials += 1
                    if log_acc < 0.0 and math.log(max(torch.rand(()).item(), 1e-300)) > log_acc:
                        # Propagate before giving up. An expansion made during this descent has
                        # tightened a bound, and leaving that out of the ancestors would let the
                        # parent keep over-weighting this subtree with no rejection to pay for it,
                        # which biases every later descent.
                        if pending:
                            self._propagate(node, depth, context)
                        self.stats.leaf_rejections += 1
                        return None
                log_ratio_sum += float(node.raw_logprob[0, token])
                if pending:
                    self._propagate(node, depth, context)
                return self._result(base, context, log_ratio_sum, log_acc)

            log_ratio_sum += float(node.raw_logprob[0, token])
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
        return None

    def _propagate(self, node, depth: int, context: List[int]) -> None:
        """Replace each ancestor's bound on this path by the total mass of its children.

        Identical to the CARS update, and correct for the same reason: the overwrite is valid
        because ``raw_logprob`` already holds the full envelope ratio, so the parent's entry for
        this child is its bound and nothing else.
        """
        # A node created but never expanded holds no bound of its own; its parent's entry for it
        # is still the original envelope value, so there is nothing to push up from it. This
        # happens whenever a descent stops on the length bound.
        while node.raw_logprob is None and depth > 0:
            depth -= 1
            node = node.parent
        while depth > 0:
            total = torch.logsumexp(node.raw_logprob[0] + node.log_theta[0], dim=0)
            depth -= 1
            node = node.parent
            node.log_theta[0, context[depth]] = total

    def _result(self, base: "rt.EnvelopeRT", context: List[int],
                log_ratio_sum: float, log_leaf_gap: float = 0.0) -> SamplingResult:
        """Package a yielded sequence.

        ``raw_logprob`` is ``log phi(w$)``, the target's own unnormalized weight. The ratios stored
        in the trie telescope to ``log E(w$)``, which equals it by condition (i) for every target
        except the coverage regime, where MARS runs on a dominating envelope; there the difference
        is the leaf gap, which varies from sequence to sequence and is added back here. Getting
        that wrong makes every reweighting or divergence computed from these numbers wrong too.

        ``constrained_logprob`` is the same quantity relative to the envelope at the root. Note
        that CARS puts the descent's own log-probability in this field, which is a different thing:
        it differs from this by the current root mass.

        Neither costs a model call; both come from values already in hand.
        """
        tok = base.models[0].llm.tokenizer
        ids = context + [base.eos_token_id]
        root_w = base.log_weight()
        log_phi = root_w + log_ratio_sum + log_leaf_gap
        return SamplingResult(
            tokens=[tok.decode([t]) for t in ids],
            token_ids=ids,
            text=tok.decode(context),
            raw_logprob=float(log_phi),
            constrained_logprob=float(log_phi - root_w),
            success=True,
        )


class _Stats:
    """What a run cost, in the quantities the theory talks about."""

    __slots__ = ("descents", "accepted", "expansions", "rejections", "rejected_mass",
                 "leaf_trials", "leaf_rejections", "dead_ends", "length_cutoffs",
                 "timeouts", "seconds", "model_calls")

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
        self.model_calls = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:
        return f"_Stats({self.as_dict()})"
