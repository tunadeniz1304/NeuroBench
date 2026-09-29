"""Base-session training losses (offline, backprop allowed)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CE(nn.Module):
    """Upstream recipe: cross-entropy on the sum-over-time linear readout."""

    def __init__(self, model, label_smoothing=0.0):
        super().__init__()
        self.ls = label_smoothing

    def forward(self, model, x, y):
        return F.cross_entropy(model(x), y, label_smoothing=self.ls)

    def logits(self, model, x):
        return model(x)


class CosineClassifier(nn.Module):
    """Cosine classifier on (optionally centred) last-layer spike counts.

    Matches the deployment readout geometry: score_c ∝ cos(S - mu, w_c). mu is an EMA of the
    batch mean spike count, i.e. the same centre the incremental learner later stores.
    Optional additive margin (CosFace) tightens class clusters for few-shot prototypes.
    """

    def __init__(self, model, n_classes=200, dim=None, scale=16.0, margin=0.0, center=True,
                 label_smoothing=0.0, ema=0.01):
        super().__init__()
        dim = dim or model.snn[-1].W.in_features
        self.weight = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.scale, self.margin, self.center, self.ls, self.ema = scale, margin, center, label_smoothing, ema
        self.register_buffer("mu", torch.zeros(dim))

    def forward(self, model, x, y):
        S = model.hidden(x).sum(1)
        if self.center:
            with torch.no_grad():
                self.mu.mul_(1 - self.ema).add_(self.ema * S.mean(0))
            S = S - self.mu
        cos = F.normalize(S, dim=1, eps=1e-6) @ F.normalize(self.weight, dim=1).t()
        if self.margin:
            cos = cos - self.margin * F.one_hot(y, cos.shape[1])
        return F.cross_entropy(self.scale * cos, y, label_smoothing=self.ls)

    def logits(self, model, x):
        S = model.hidden(x).sum(1)
        if self.center:
            S = S - self.mu
        return F.normalize(S, dim=1, eps=1e-6) @ F.normalize(self.weight, dim=1).t()


class SupConPlusCosine(CosineClassifier):
    """Cosine classifier + supervised contrastive term on the normalised spike-count embedding."""

    def __init__(self, model, supcon_weight=0.5, temperature=0.1, **kw):
        super().__init__(model, **kw)
        self.w, self.t = supcon_weight, temperature

    def forward(self, model, x, y):
        S = model.hidden(x).sum(1)
        if self.center:
            with torch.no_grad():
                self.mu.mul_(1 - self.ema).add_(self.ema * S.mean(0))
            S = S - self.mu
        z = F.normalize(S, dim=1, eps=1e-6)
        cos = z @ F.normalize(self.weight, dim=1).t()
        if self.margin:
            cos = cos - self.margin * F.one_hot(y, cos.shape[1])
        ce = F.cross_entropy(self.scale * cos, y, label_smoothing=self.ls)
        sim = z @ z.t() / self.t
        eye = torch.eye(len(y), device=y.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)
        pos = (y[:, None] == y[None, :]) & ~eye
        logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        npos = pos.sum(1).clamp(min=1)
        supcon = -(logprob * pos).sum(1) / npos
        return ce + self.w * supcon.mean()


LOSSES = {"ce": CE, "cosine": CosineClassifier, "supcon_cosine": SupConPlusCosine}


def build_loss(lcfg, model):
    lcfg = dict(lcfg)
    return LOSSES[lcfg.pop("name")](model, **lcfg)
