"""Incremental readout learners on spike-count features.

Every learner produces a linear readout (W, b) over the per-sample spike-count vector S of the last
hidden layer, so that class scores are `W S + b`. Inside the SNN this is realised by the sum-over-time
readout `sum_t (W s_t + b/T)` (see nbfscil.snn.SumReadout), i.e. every hidden spike adds one weight
column to the class accumulators (event-driven ACs).

Learners only ever see the few-shot support set of the current session plus their own stored state
(e.g. per-class integer accumulators); no raw data are replayed.
"""
import torch

from nbfscil.hebbian import ThreeFactorPrototypes


class EuclidProto:
    """Upstream baseline: w_c = 2 m_c, b_c = -|m_c|^2  (float NCM, squared Euclidean)."""

    def __init__(self, n_classes=200, dim=1024, **_):
        self.W = torch.zeros(n_classes, dim)
        self.b = torch.zeros(n_classes)

    def fit_base(self, feats, labels, classes):
        self.learn(feats, labels, classes)

    def learn(self, feats, labels, classes):
        for c in classes:
            m = feats[labels == c].float().mean(0)
            self.W[c] = 2 * m
            self.b[c] = -(m @ m)

    def readout(self):
        return self.W, self.b

    def state_bytes(self):
        return 0  # the prototype *is* the readout weight


class HebbianCL2N:
    """Centered, L2-normalized, k-bit integer prototypes learned with a three-factor rule.

    score_c = S . w_c + b_c,  w_c = q((P_c - mu) / |P_c - mu|),  b_c = -mu . w_c
    (the sample-norm term of the cosine similarity is common to all classes and drops out of argmax).
    """

    def __init__(self, n_classes=200, dim=1024, weight_bits=8, center=True, normalize=True,
                 acc_bits=16, stochastic_round=True, seed=0, **_):
        self.rule = ThreeFactorPrototypes(n_classes, dim, weight_bits=weight_bits, acc_bits=acc_bits,
                                          center=center, normalize=normalize,
                                          stochastic_round=stochastic_round, seed=seed)

    def fit_base(self, feats, labels, classes):
        self.rule.set_center(feats)  # mean base spike count, computed once offline
        self.learn(feats, labels, classes)

    def learn(self, feats, labels, classes):
        for c in classes:
            self.rule.learn_class(c, feats[labels == c])

    def readout(self):
        return self.rule.readout()

    def state_bytes(self):
        return self.rule.state_bytes()


LEARNERS = {"euclid_proto": EuclidProto, "hebbian_cl2n": HebbianCL2N}


def build_learner(cfg, n_classes, dim):
    cfg = dict(cfg)
    return LEARNERS[cfg.pop("name")](n_classes=n_classes, dim=dim, **cfg)
