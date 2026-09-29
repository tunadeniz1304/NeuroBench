"""The event-driven integer hardware model must match the training-time rule bit for bit."""
import random

import pytest
import torch

from hw_model.prototype_rule import PrototypeLearnerHW, isqrt as hw_isqrt
from nbfscil.hebbian import ThreeFactorPrototypes


def _spike_trains(n, dim, T, rate, seed):
    g = torch.Generator().manual_seed(seed)
    rates = torch.rand(dim, generator=g) * rate
    return (torch.rand(n, T, dim, generator=g) < rates).to(torch.int8)


def _events(train):
    t_idx, i_idx = torch.nonzero(train, as_tuple=True)
    return list(zip(t_idx.tolist(), i_idx.tolist()))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 15, 16, 17, 10**6, 2**40 + 12345, 99999999977])
def test_isqrt(n):
    import math
    assert hw_isqrt(n) == math.isqrt(n)


@pytest.mark.parametrize("bits,stoch,center,normalize", [
    (8, True, True, True), (4, True, True, True), (8, False, True, True),
    (6, True, False, True), (8, False, True, False)])
def test_learning_and_inference_bit_exact(bits, stoch, center, normalize):
    dim, T, n_cls, shots = 64, 40, 6, 5
    base = _spike_trains(200, dim, T, 0.3, seed=0)
    ref = ThreeFactorPrototypes(n_cls, dim, weight_bits=bits, acc_bits=16, center=center,
                                normalize=normalize, stochastic_round=stoch, seed=12345)
    ref.set_center(base.sum(1))
    hw = PrototypeLearnerHW(n_cls, dim, weight_bits=bits, acc_bits=16, frac_bits=ref.frac_bits,
                            G=ref.G, mu_q=ref.mu_q.tolist(), seed=12345, center=center,
                            normalize=normalize, stochastic_round=stoch)
    for c in range(n_cls):
        support = _spike_trains(shots, dim, T, 0.4, seed=100 + c)
        ref.learn_class(c, support.sum(1))
        for k in range(shots):
            hw.present_sample(_events(support[k]))
        hw.consolidate(c)
        assert ref.W[c].tolist() == hw.W[c], f"weights differ for class {c}"
        assert int(ref.b[c]) == hw.B[c], f"bias differs for class {c}"

    queries = _spike_trains(30, dim, T, 0.35, seed=999)
    ref_scores = ref.scores(queries.sum(1))
    for q in range(len(queries)):
        pred, score = hw.infer(_events(queries[q]), list(range(n_cls)))
        assert [score[c] for c in range(n_cls)] == ref_scores[q].tolist()
        assert pred == int(ref_scores[q].argmax())


def test_float_readout_equals_integer_scores():
    """What the harness runs (float W, b summed over time) equals the integer scores."""
    dim, T = 128, 50
    ref = ThreeFactorPrototypes(4, dim, weight_bits=8, seed=7)
    ref.set_center(_spike_trains(100, dim, T, 0.2, 1).sum(1))
    for c in range(4):
        ref.learn_class(c, _spike_trains(5, dim, T, 0.3, 10 + c).sum(1))
    x = _spike_trains(20, dim, T, 0.25, 3)
    W, b = ref.readout()
    lin = torch.nn.Linear(dim, 4)
    lin.weight.data, lin.bias.data = W, b / T
    harness = lin(x.float()).sum(1)
    assert torch.equal(torch.round(harness).long(), ref.scores(x.sum(1)))
