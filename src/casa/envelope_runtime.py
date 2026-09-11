"""Evaluating an envelope expression on a prefix.

:mod:`casa.algebra` decides *whether* a target admits an envelope. This module computes it. The two
are kept apart so that the algebra stays free of torch and of any model.

The runtime mirrors the expression one node at a time. Every node answers two questions about the
prefix it currently sits on:

    log_weight()   log E(u)              a scalar
    log_next()     log E(ua) for all a   a vector over the vocabulary

and can be advanced by one token. Because every node answers in *absolute* log-weight rather than
in conditionals, operations like the minimum compose correctly: ``min(E_p, E_r)`` at a prefix is not
determined by the two next-token distributions alone, it needs the two prefix weights as well.

All models in one expression must share a tokenizer. That is checked when the runtime is built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from casa.algebra import Envelope, Mean, Model, Potential, Power, Product, Scorer

NEG_INF = float("-inf")


class TokenizerMismatch(Exception):
    """Raised when an expression mixes models that do not share a token space.

    MARS evaluates every model on the same token context, so a shared vocabulary is required. Two
    models with different tokenizers can still be ensembled, but only after both are mapped to a
    common byte alphabet, which this runtime does not do.
    """


# --------------------------------------------------------------------------------------------
# Node runtimes
# --------------------------------------------------------------------------------------------


class NodeRT:
    """Runtime for one expression node, positioned at some prefix."""

    def log_weight(self) -> float:
        raise NotImplementedError

    def log_next(self) -> torch.Tensor:
        raise NotImplementedError

    def advance(self, token: int) -> "NodeRT":
        raise NotImplementedError

    def forward_passes(self) -> int:
        return 0


class ModelRT(NodeRT):
    """A base language model, with its own KV cache and its own prompt.

    Two ``ModelRT`` over the same ``LLM`` with different prompts is the within-model ensemble: one
    set of weights, two conditionings, two caches.
    """

    def __init__(self, node: Model, prompt_ids: Sequence[int], device, vocab_size: int,
                 temperature: float = 1.0):
        self.node = node
        self.llm = node.llm
        self.prompt_ids = list(prompt_ids)
        self.device = device
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.context: List[int] = []
        self._log_weight = 0.0
        self._past = None
        self._cached_len = 0
        self._log_cond: Optional[torch.Tensor] = None
        self._passes = 0

    def _ensure(self) -> torch.Tensor:
        if self._log_cond is not None:
            return self._log_cond
        ids = self.prompt_ids + self.context
        if self._past is not None and self._cached_len > 0:
            new_ids = torch.tensor([ids[self._cached_len:]], device=self.device)
            past = self._past
        else:
            new_ids = torch.tensor([ids], device=self.device)
            past = None
        with torch.no_grad():
            out = self.llm.model(new_ids, past_key_values=past, use_cache=True)
        self._past = out.past_key_values
        self._cached_len = len(ids)
        self._passes += 1
        logits = out.logits[0, -1, :].float()
        if logits.shape[-1] > self.vocab_size:
            logits = logits[: self.vocab_size]
        elif logits.shape[-1] < self.vocab_size:
            pad = torch.full((self.vocab_size - logits.shape[-1],), NEG_INF, device=self.device)
            logits = torch.cat([logits, pad])
        if self.temperature != 1.0:
            logits = logits / self.temperature
        # A language model's logits become a distribution; a general potential's are already
        # log-weights and must not be renormalized, or a 0/1 mask would come back as a uniform
        # choice among the tokens it permits.
        self._log_cond = torch.log_softmax(logits, dim=-1) if self.node.sub_stochastic else logits
        return self._log_cond

    def log_weight(self) -> float:
        return self._log_weight

    def log_next(self) -> torch.Tensor:
        return self._ensure() + self._log_weight

    def advance(self, token: int) -> "ModelRT":
        cond = self._ensure()
        child = ModelRT.__new__(ModelRT)
        child.__dict__.update(self.__dict__)
        child.context = self.context + [token]
        child._log_weight = self._log_weight + float(cond[token])
        child._log_cond = None
        # The KV cache is shared by reference: CASA's CARS does the same, and a descent only ever
        # extends the context, so the cache stays a prefix of what the child needs.
        return child

    def forward_passes(self) -> int:
        return self._passes


class PowerRT(NodeRT):
    def __init__(self, child: NodeRT, gamma: float):
        self.child, self.gamma = child, gamma

    def log_weight(self) -> float:
        return self.gamma * self.child.log_weight()

    def log_next(self) -> torch.Tensor:
        return self.gamma * self.child.log_next()

    def advance(self, token: int) -> "PowerRT":
        return PowerRT(self.child.advance(token), self.gamma)

    def forward_passes(self) -> int:
        return self.child.forward_passes()


class ProductRT(NodeRT):
    """``prod_k E_k ** gamma_k``, times any prefix-monotone scorer factors."""

    def __init__(self, terms: List[Tuple[NodeRT, float]], scorers: List["ScorerRT"]):
        self.terms, self.scorers = terms, scorers

    def log_weight(self) -> float:
        w = sum(g * c.log_weight() for c, g in self.terms)
        return w + sum(s.log_weight() for s in self.scorers)

    def log_next(self) -> torch.Tensor:
        out = None
        for c, g in self.terms:
            v = c.log_next() * g
            out = v if out is None else out + v
        for s in self.scorers:
            out = out + s.log_next_mask(out.shape[-1], out.device)
        return out

    def advance(self, token: int) -> "ProductRT":
        return ProductRT(
            [(c.advance(token), g) for c, g in self.terms],
            [s.advance(token) for s in self.scorers],
        )

    def forward_passes(self) -> int:
        return sum(c.forward_passes() for c, _ in self.terms)


class MeanRT(NodeRT):
    """The generalized mean at power ``tau``, in log space.

        log M_tau = (1 / tau) * logsumexp_k ( log w_k + tau * log E_k )

    with the limits handled directly: ``tau -> -inf`` is the minimum, ``tau -> +inf`` the maximum,
    and ``tau = 0`` never reaches here because the algebra rewrites it as a weighted product.
    """

    def __init__(self, children: List[NodeRT], weights: Sequence[float], tau: float):
        self.children = children
        self.weights = list(weights)
        self.tau = tau
        self._logw = [math.log(w) if w > 0 else NEG_INF for w in self.weights]

    def _combine_scalar(self, vals: List[float]) -> float:
        if self.tau == -math.inf:
            return min(vals)
        if self.tau == math.inf:
            return max(vals)
        t = torch.tensor([lw + self.tau * v for lw, v in zip(self._logw, vals)])
        return float(torch.logsumexp(t, dim=0)) / self.tau

    def _combine_vec(self, vecs: List[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(vecs, dim=0)
        if self.tau == -math.inf:
            return stack.min(dim=0).values
        if self.tau == math.inf:
            return stack.max(dim=0).values
        lw = torch.tensor(self._logw, device=stack.device).unsqueeze(1)
        return torch.logsumexp(lw + self.tau * stack, dim=0) / self.tau

    @property
    def subadditive(self) -> bool:
        """``tau > 1`` is the coverage regime, where the mean of envelopes is not an envelope."""
        return self.tau > 1.0

    def _dominating_vec(self, vecs: List[torch.Tensor]) -> torch.Tensor:
        """``c * sum_k w_k E_k`` with ``c = 1 / min_k w_k``, which dominates ``M_tau``."""
        stack = torch.stack(vecs, dim=0)
        lw = torch.tensor(self._logw, device=stack.device).unsqueeze(1)
        return torch.logsumexp(lw + stack, dim=0) - math.log(min(self.weights))

    def _dominating_scalar(self, vals: List[float]) -> float:
        t = torch.tensor([lw + v for lw, v in zip(self._logw, vals)])
        return float(torch.logsumexp(t, dim=0)) - math.log(min(self.weights))

    def log_weight(self) -> float:
        vals = [c.log_weight() for c in self.children]
        if self.subadditive:
            return self._dominating_scalar(vals)
        return self._combine_scalar(vals)

    def log_next(self) -> torch.Tensor:
        """The envelope. In the coverage regime this is the dominating mixture, not the mean."""
        vecs = [c.log_next() for c in self.children]
        if self.subadditive:
            return self._dominating_vec(vecs)
        return self._combine_vec(vecs)

    def log_next_true(self) -> torch.Tensor:
        """The target itself, which differs from the envelope only in the coverage regime."""
        return self._combine_vec([_true_next(c) for c in self.children])

    def advance(self, token: int) -> "MeanRT":
        return MeanRT([c.advance(token) for c in self.children], self.weights, self.tau)

    def forward_passes(self) -> int:
        return sum(c.forward_passes() for c in self.children)


class ScorerRT:
    """A prefix-monotone scorer factor.

    Needs the weight of every continuation, not just of the current prefix, so it must supply a
    vectorized mask. A grammar recognizer provides one cheaply as a token bitmask; a general
    Python scorer would have to be called once per vocabulary entry, which is refused rather than
    done silently.
    """

    def __init__(self, node: Scorer, context: Optional[List[int]] = None):
        self.node = node
        self.context = context or []
        if getattr(node, "log_mask", None) is None:
            raise NotImplementedError(
                f"scorer {node.name!r} is prefix-monotone but supplies no vectorized mask. "
                "Give it a `log_mask(context, vocab_size)` returning log g(ua) for every token, "
                "or mark it output-only so MARS uses leaf rejection instead."
            )

    def log_weight(self) -> float:
        v = self.node.fn(self.context)
        return math.log(v) if v > 0 else NEG_INF

    def log_next_mask(self, vocab_size: int, device) -> torch.Tensor:
        return self.node.log_mask(self.context, vocab_size).to(device)

    def advance(self, token: int) -> "ScorerRT":
        return ScorerRT(self.node, self.context + [token])


# --------------------------------------------------------------------------------------------
# Building a runtime from a validated envelope
# --------------------------------------------------------------------------------------------


@dataclass
class EnvelopeRT:
    """The root of an envelope runtime, positioned at the empty prefix."""

    root: NodeRT
    vocab_size: int
    eos_token_id: int
    models: List[ModelRT]

    def log_weight(self) -> float:
        return self.root.log_weight()

    def log_next(self) -> torch.Tensor:
        return self.root.log_next()

    def log_ratios(self) -> torch.Tensor:
        """``log E(ua) - log E(u)`` for every token.

        This is what a MARS trie node stores. Condition (ii) says it exponentiates to at most one
        in total, and the shortfall is exactly the rejection mass paid at this expansion.
        """
        w = self.root.log_weight()
        if w == NEG_INF:
            return torch.full((self.vocab_size,), NEG_INF)
        return self.root.log_next() - w

    def log_leaf_acceptance(self, token: int) -> float:
        """``log phi(w$) - log E(w$)`` for a yielded complete sequence.

        Zero whenever the envelope is exact at leaves, which is every target except the coverage
        regime of the generalized mean. There MARS runs on a dominating mixture and pays the
        difference once, at the leaf, with acceptance bounded below by ``min_k w_k``.
        """
        true = _true_next(self.root)
        env = self.root.log_next()
        gap = float(true[token]) - float(env[token])
        return min(gap, 0.0)

    def advance(self, token: int) -> "EnvelopeRT":
        return EnvelopeRT(self.root.advance(token), self.vocab_size, self.eos_token_id, self.models)

    def forward_passes(self) -> int:
        return self.root.forward_passes()


def _true_next(node: NodeRT) -> torch.Tensor:
    """The target's next-token weights, which coincide with the envelope's except under a
    subadditive mean."""
    fn = getattr(node, "log_next_true", None)
    if fn is not None:
        return fn()
    if isinstance(node, PowerRT):
        return node.gamma * _true_next(node.child)
    if isinstance(node, ProductRT):
        out = None
        for c, g in node.terms:
            v = _true_next(c) * g
            out = v if out is None else out + v
        for s in node.scorers:
            out = out + s.log_next_mask(out.shape[-1], out.device)
        return out
    return node.log_next()


def build(envelope: Envelope, prompts: Optional[Dict[int, str]] = None,
          temperature: float = 1.0) -> EnvelopeRT:
    """Instantiate a runtime for a validated envelope.

    Args:
        envelope: The result of ``Potential.envelope()``.
        prompts: Optional override mapping ``id(Model node)`` to a prompt string. Models carrying
            their own ``prompt`` use it; this is for supplying one task instance at a time.
        temperature: Sampling temperature applied to every model's logits.

    Raises:
        TokenizerMismatch: If the expression mixes models with different token spaces.
    """
    models = envelope.models()
    if not models:
        raise ValueError("expression contains no model")

    ref = models[0].llm
    ref_tok = ref.tokenizer
    for m in models[1:]:
        if not _same_token_space(ref_tok, m.llm.tokenizer):
            raise TokenizerMismatch(
                f"{models[0].name} and {m.name} do not share a token space "
                f"({ref_tok.__class__.__name__}/{len(ref_tok)} vs "
                f"{m.llm.tokenizer.__class__.__name__}/{len(m.llm.tokenizer)}). "
                "MARS evaluates every model on the same token context, so all models in one "
                "expression must share a tokenizer. Map both to a byte alphabet first, or pair "
                "models from the same family."
            )

    vocab_size = len(ref_tok)
    device = ref.model.device
    built: List[ModelRT] = []

    def go(node: Potential) -> NodeRT:
        if isinstance(node, Model):
            prompt = (prompts or {}).get(id(node), node.prompt) or ""
            ids = node.llm.tokenizer.encode(node.llm.format_prompt(prompt), add_special_tokens=False)
            rt = ModelRT(node, ids, device, vocab_size, temperature)
            built.append(rt)
            return rt
        if isinstance(node, Power):
            return PowerRT(go(node.base), node.gamma)
        if isinstance(node, Product):
            return ProductRT([(go(b), g) for b, g in node.terms],
                             [ScorerRT(s) for s in node.scorers])
        if isinstance(node, Mean):
            return MeanRT([go(t) for t in node.terms], node.weights, node.tau)
        if isinstance(node, Scorer):
            raise ValueError("a scorer cannot be the whole target; multiply it into a model")
        raise TypeError(f"unhandled expression node {type(node).__name__}")

    root = go(envelope.expr)
    return EnvelopeRT(root, vocab_size, ref_tok.eos_token_id, built)


def _same_token_space(a, b) -> bool:
    if a is b:
        return True
    if len(a) != len(b):
        return False
    va, vb = a.get_vocab(), b.get_vocab()
    return va == vb
