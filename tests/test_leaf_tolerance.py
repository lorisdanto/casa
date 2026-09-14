"""A leaf where the envelope sits below the target by float32 rounding must accept, not raise.

Twenty of fifty maximum-ensemble shards died on gaps of 4e-6 to 3e-5 in log space, which is the
error of a float32 log-softmax over a 128k vocabulary, not a violation of condition (i).
"""

import math

import pytest
import torch

from casa import envelope_runtime as rt


class _Root(rt.NodeRT):
    def __init__(self, lp):
        self.lp = lp

    def log_weight(self):
        return 0.0

    def log_next(self):
        return self.lp


def _leaf(monkeypatch, gap):
    """An EnvelopeRT on a dominating mixture whose true value exceeds the envelope by ``gap``."""
    V, tok = 64, 7
    env = torch.log_softmax(torch.randn(V), dim=-1)
    true = env.clone()
    true[tok] += gap
    monkeypatch.setattr(rt, "_true_next", lambda root: true)
    state = rt.EnvelopeRT(root=_Root(env), vocab_size=V, eos_token_id=tok, models=[],
                          counters={}, leaf_scorers=(), context=(), dominating=True)
    return state, tok


def test_rounding_sized_gap_accepts(monkeypatch):
    for gap in (3.8e-6, 1.5e-5, 3.1e-5, 5e-4):
        state, tok = _leaf(monkeypatch, gap)
        assert state.log_leaf_acceptance(tok) == 0.0


def test_real_violation_still_raises(monkeypatch):
    state, tok = _leaf(monkeypatch, 0.05)
    with pytest.raises(ValueError, match="condition \\(i\\)"):
        state.log_leaf_acceptance(tok)


def test_envelope_above_target_pays_the_gap(monkeypatch):
    state, tok = _leaf(monkeypatch, -0.7)
    assert math.isclose(state.log_leaf_acceptance(tok), -0.7, rel_tol=1e-5)
