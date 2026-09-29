"""Three-factor Hebbian prototype rule — training-time (vectorised, integer-exact) version.

Local rule, per readout synapse (class neuron c, hidden neuron i):

    dA_ci(t) = pre_i(t) * post_c(t) * M(t)

  pre_i(t)  in {0,1}  hidden spike
  post_c(t) in {0,1}  teacher/label spike on class neuron c while a support sample of c is presented
  M(t)      in {0,1}  neuromodulatory learning-enable (novelty) signal, high during a learning session

Integrated over a session, A_c is the summed spike count of the n support samples of class c.
When the session for class c ends (consolidation), the class neuron converts its accumulator into
k-bit weights and a bias using only its own synapses plus two shared constants (the base centre mu_q
and the global scale G):

    D_ci  = A_ci * 2^f - n * mu_q,i                       (centring; f fractional bits of mu_q)
    N_c   = isqrt(sum_i D_ci^2)                           (integer sqrt, once per class)
    w_ci  = clip(floor((D_ci * G + r_ci) / N_c))          (stochastic rounding, r_ci ~ U[0, N_c) from xorshift32)
    b_c   = -floor(sum_i mu_q,i * w_ci / 2^f)

Class score for an input with spike counts S: score_c = sum_i S_i w_ci + b_c
(= G * cosine(S - mu, P_c - mu) * |S - mu| up to quantisation; the |S - mu| factor is shared by all classes).

The bit-exact event-driven reference lives in hw_model/prototype_rule.py; tests/test_hw_model.py checks equality.
"""
import math

import torch

from nbfscil.prng import xorshift32_stream


def isqrt(n: int) -> int:
    return math.isqrt(int(n))


def floor_div(a: torch.Tensor, b: int) -> torch.Tensor:
    return torch.div(a, b, rounding_mode="floor")


class ThreeFactorPrototypes:
    def __init__(self, n_classes, dim, weight_bits=8, acc_bits=24, center=True, normalize=True,
                 stochastic_round=True, frac_bits=4, scale=None, seed=1):
        self.n_classes, self.dim = n_classes, dim
        self.weight_bits, self.acc_bits, self.frac_bits = weight_bits, acc_bits, frac_bits
        self.center, self.normalize, self.stochastic_round = center, normalize, stochastic_round
        self.wmax = 2 ** (weight_bits - 1) - 1
        # global scale G: a unit vector in R^dim has typical entries 1/sqrt(dim); map ~4 sigma to wmax
        self.G = int(scale) if scale is not None else int(round(self.wmax * math.sqrt(dim) / 4))
        self.mu_q = torch.zeros(dim, dtype=torch.int64)
        self.W = torch.zeros(n_classes, dim, dtype=torch.int64)
        self.b = torch.zeros(n_classes, dtype=torch.int64)
        self.prng_state = seed

    # ------------------------------------------------------------------ offline constants
    def set_center(self, base_counts):
        if self.center:
            mu = base_counts.double().mean(0)
            self.mu_q = torch.round(mu * 2 ** self.frac_bits).to(torch.int64)

    # ------------------------------------------------------------------ learning
    def accumulate(self, counts):
        """Integrated three-factor updates for one class: A = sum over shots of spike counts."""
        A = counts.to(torch.int64).sum(0)
        assert A.max() < 2 ** (self.acc_bits - 1), "accumulator overflow"
        return A, counts.shape[0]

    def consolidate(self, c, A, n):
        D = A * 2 ** self.frac_bits - n * self.mu_q
        if self.normalize:
            N = isqrt(int((D * D).sum()))
            N = max(N, 1)
            if self.stochastic_round:
                rnd, self.prng_state = xorshift32_stream(self.prng_state, self.dim)
                r = torch.tensor(rnd, dtype=torch.int64) % N
            else:
                r = torch.full((self.dim,), N // 2, dtype=torch.int64)  # round half up
            w = floor_div(D * self.G + r, N)
        else:
            w = floor_div(D, n * 2 ** self.frac_bits)
        w = w.clamp(-self.wmax, self.wmax)
        self.W[c] = w
        self.b[c] = -floor_div((self.mu_q * w).sum(), 2 ** self.frac_bits)

    def learn_class(self, c, counts):
        A, n = self.accumulate(counts)
        self.consolidate(c, A, n)

    # ------------------------------------------------------------------ deployment
    def readout(self):
        return self.W.float(), self.b.float()

    def scores(self, counts):
        return counts.to(torch.int64) @ self.W.t() + self.b

    def state_bytes(self):
        # persistent beyond the readout weights: mu_q (dim x 16 bit) + one transient accumulator bank
        return self.dim * 2 + self.dim * self.acc_bits // 8 + 4
