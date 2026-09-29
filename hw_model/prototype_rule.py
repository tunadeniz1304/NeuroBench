"""Bit-exact, event-driven integer reference of the three-factor prototype learner + readout.

Written the way an RTL block would process data: one hidden spike event at a time, fixed-width
integer registers, no floating point anywhere, a sequential integer square root and divider used
once per class at consolidation, and a xorshift32 LFSR for stochastic rounding.

Blocks
  AccumulatorBank  A[i]  (acc_bits, unsigned)   += pre_i(t) when post_c(t) & M(t)
  Consolidation    centring, sum of squares, isqrt, per-synapse divide with stochastic rounding
  WeightMemory     W[c][i] (weight_bits, signed), B[c] (bias register)
  Readout          score[c] += W[c][i] on every hidden spike i; + B[c] at the end; argmax

Must agree exactly with nbfscil.hebbian.ThreeFactorPrototypes (tests/test_hw_model.py).
"""
from nbfscil.prng import xorshift32


def isqrt(n):
    """Digit-by-digit (restoring) integer square root, as a small sequential RTL unit would do."""
    if n < 0:
        raise ValueError
    res, bit = 0, 1 << ((n.bit_length() + 1) & ~1)
    while bit > n:
        bit >>= 2
    while bit:
        if n >= res + bit:
            n -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    return res


def floordiv(a, b):
    return a // b  # python floor division == arithmetic floor for signed operands


class PrototypeLearnerHW:
    def __init__(self, n_classes, dim, weight_bits, acc_bits, frac_bits, G, mu_q, seed,
                 center=True, normalize=True, stochastic_round=True):
        self.n_classes, self.dim = n_classes, dim
        self.wmax = (1 << (weight_bits - 1)) - 1
        self.acc_max = (1 << (acc_bits - 1)) - 1
        self.frac_bits, self.G = frac_bits, G
        self.mu_q = [int(v) for v in mu_q] if center else [0] * dim
        self.normalize, self.stochastic_round = normalize, stochastic_round
        self.lfsr = seed & 0xFFFFFFFF or 0x9E3779B9
        self.W = [[0] * dim for _ in range(n_classes)]
        self.B = [0] * n_classes
        self._reset_bank()

    # ---------------------------------------------------------------- learning
    def _reset_bank(self):
        self.A = [0] * self.dim
        self.n = 0

    def present_sample(self, spike_events, post=True, M=True):
        """spike_events: iterable of (t, i) hidden spikes for one support sample of the current class."""
        if not (post and M):
            return
        for _, i in spike_events:
            self.A[i] += 1
            if self.A[i] > self.acc_max:
                raise OverflowError("accumulator overflow")
        self.n += 1

    def consolidate(self, c):
        D = [(self.A[i] << self.frac_bits) - self.n * self.mu_q[i] for i in range(self.dim)]
        w = [0] * self.dim
        if self.normalize:
            ss = 0
            for d in D:
                ss += d * d
            N = max(isqrt(ss), 1)
            for i in range(self.dim):
                if self.stochastic_round:
                    self.lfsr = xorshift32(self.lfsr)
                    r = self.lfsr % N
                else:
                    r = N >> 1
                w[i] = floordiv(D[i] * self.G + r, N)
        else:
            for i in range(self.dim):
                w[i] = floordiv(D[i], self.n << self.frac_bits)
        for i in range(self.dim):
            w[i] = max(-self.wmax, min(self.wmax, w[i]))
        acc = 0
        for i in range(self.dim):
            acc += self.mu_q[i] * w[i]
        self.W[c] = w
        self.B[c] = -(acc >> self.frac_bits)  # arithmetic shift == floor division by 2^f
        self._reset_bank()

    # ---------------------------------------------------------------- inference
    def infer(self, spike_events, active_classes):
        score = {c: 0 for c in active_classes}
        for _, i in spike_events:
            for c in active_classes:
                score[c] += self.W[c][i]
        for c in active_classes:
            score[c] += self.B[c]
        best = max(active_classes, key=lambda c: (score[c], -c))  # ties -> lowest index, like argmax
        return best, score
