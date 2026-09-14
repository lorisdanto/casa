"""A dead-end prefix must stop the SMC particle, not hand genlm a vector of NaN.

When every entry of a ratio vector is -inf, genlm normalizes it to NaN and its Gumbel-max draw
returns index 0 every time. The particle is then extended by token 0, the grammar rejects it on the
next call, and the whole sweep dies. Four CARS grammar rows (json-product, sql-min, xml-product x2)
were lost that way after 1-2 hours each.
"""

import numpy as np
import pytest
import torch

from casa.genlm_bridge import DEAD_END_LOG_WEIGHT, vocab_eos_weights

NEG_INF = float("-inf")


def test_live_vector_is_only_reordered():
    V, eos = 10, 4
    ratios = torch.log_softmax(torch.randn(V), dim=-1)
    ratios[2] = NEG_INF
    ws = vocab_eos_weights(ratios, eos)
    assert ws.dtype == torch.float64
    expected = torch.cat([ratios[:eos], ratios[eos + 1:], ratios[eos:eos + 1]]).double()
    assert torch.equal(ws, expected)


def test_only_eos_alive_is_untouched():
    V, eos = 10, 9
    ratios = torch.full((V,), NEG_INF)
    ratios[eos] = -3.0
    ws = vocab_eos_weights(ratios, eos)
    assert ws[-1] == -3.0 and torch.isinf(ws[:-1]).all()


def test_dead_end_puts_negligible_weight_on_eos_alone():
    V, eos = 10, 4
    ws = vocab_eos_weights(torch.full((V,), NEG_INF), eos)
    assert ws[-1] == DEAD_END_LOG_WEIGHT
    assert torch.isneginf(ws[:-1]).all()


def test_genlm_draws_eos_at_a_dead_end():
    util = pytest.importorskip("genlm.control.util")
    V, eos = 1000, 500
    ws = vocab_eos_weights(torch.full((V,), NEG_INF), eos).numpy()

    # What used to happen: all -inf normalizes to NaN and the draw is always index 0.
    with np.errstate(invalid="ignore"):
        dead = np.full(V, NEG_INF)
        assert all(util.fast_sample_logprobs(dead - np.logaddexp.reduce(dead), 1)[0] == 0
                   for _ in range(5))

    # Now: the normalized vector is finite at EOS and the draw is EOS, every time.
    with np.errstate(invalid="ignore"):
        logps = ws - np.logaddexp.reduce(ws)
    assert logps[-1] == 0.0
    assert all(util.fast_sample_logprobs(logps, 1)[0] == V - 1 for _ in range(20))


def test_dead_particle_vanishes_next_to_a_live_one():
    # Resampling weights: a stopped dead-end particle against a live particle of tiny weight.
    logw = np.array([DEAD_END_LOG_WEIGHT, -5000.0])
    w = np.exp(logw - np.logaddexp.reduce(logw))
    assert w[0] == 0.0 and w[1] == 1.0
