"""Graph-capturable re-implementation of the sparch RadLIF RSNN used by the NeuroBench baseline.

Same equations, parameter names and state_dict keys as
third_party/neurobench/examples/mswc_fscil/sparchSNNs.py (RadLIFLayer, Simple_ReadoutLayer),
so upstream checkpoints load unchanged. Differences are purely technical:
  * random initial states are drawn on-device (upstream draws on CPU and copies, which
    blocks CUDA-graph capture);
  * optional `init_state="zeros"` for deterministic, hardware-friendly inference.
tests/test_snn_equivalence.py checks the forward pass against the upstream module.
"""
import numpy as np
import torch
import torch.nn as nn


class SpikeFunctionBoxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x


spike_fct = SpikeFunctionBoxcar.apply


class RadLIFLayer(nn.Module):
    def __init__(self, input_size, hidden_size, threshold=1.0, dropout=0.0,
                 normalization="batchnorm", use_bias=False, init_state="random"):
        super().__init__()
        self.input_size, self.hidden_size = int(input_size), int(hidden_size)
        self.threshold = threshold
        self.init_state = init_state
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.empty(self.hidden_size).uniform_(*self.alpha_lim))
        self.beta = nn.Parameter(torch.empty(self.hidden_size).uniform_(*self.beta_lim))
        self.a = nn.Parameter(torch.empty(self.hidden_size).uniform_(*self.a_lim))
        self.b = nn.Parameter(torch.empty(self.hidden_size).uniform_(*self.b_lim))
        nn.init.orthogonal_(self.V.weight)
        self.normalize = normalization == "batchnorm"
        if self.normalize:
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        Wx = self.W(x)
        if self.normalize:
            B, T, H = Wx.shape
            Wx = self.norm(Wx.reshape(B * T, H)).reshape(B, T, H)
        s = self._cell(Wx)
        return self.drop(s)

    def _cell(self, Wx):
        B, T, H = Wx.shape
        if self.init_state == "random":
            ut = torch.rand(B, H, device=Wx.device)
            wt = torch.rand(B, H, device=Wx.device)
            torch.rand(B, H, device=Wx.device)  # upstream draws an unused st; keep RNG streams aligned
            st = spike_fct(ut - 0.5)
        else:
            ut = Wx.new_zeros(B, H)
            wt = Wx.new_zeros(B, H)
            st = Wx.new_zeros(B, H)
        alpha = torch.clamp(self.alpha, *self.alpha_lim)
        beta = torch.clamp(self.beta, *self.beta_lim)
        a = torch.clamp(self.a, *self.a_lim)
        b = torch.clamp(self.b, *self.b_lim)
        with torch.no_grad():
            self.V.weight.fill_diagonal_(0.0)
        s = []
        for t in range(T):
            wt = beta * wt + a * ut + b * st
            # module call (not st @ V.weight.t()) as upstream: the harness counts SynOps through Linear hooks,
            # so a functional matmul would leave the recurrent synapses out of Dense / Eff_ACs
            ut = alpha * (ut - st) + (1 - alpha) * (Wx[:, t, :] + self.V(st) - wt)
            st = spike_fct(ut - self.threshold)
            s.append(st)
        return torch.stack(s, dim=1)


class SumReadout(nn.Module):
    """Upstream Simple_ReadoutLayer: sum over time of W s_t (+ b)."""

    def __init__(self, input_size, n_out, use_bias=False):
        super().__init__()
        self.W = nn.Linear(input_size, n_out, bias=use_bias)

    def forward(self, x):
        return self.W(x).sum(dim=1)


class RSNN(nn.Module):
    """RadLIF RSNN; `self.snn` mirrors the upstream `SNN.snn` ModuleList (same state_dict keys)."""

    def __init__(self, layer_sizes=(1024, 1024), n_in=20, n_out=200, dropout=0.1,
                 normalization="batchnorm", init_state="random", use_readout_bias=False):
        super().__init__()
        layers, d = [], n_in
        for h in layer_sizes:
            layers.append(RadLIFLayer(d, h, dropout=dropout, normalization=normalization,
                                      init_state=init_state))
            d = h
        layers.append(SumReadout(d, n_out, use_bias=use_readout_bias))
        self.snn = nn.ModuleList(layers)
        self.activation_module = RadLIFLayer

    def hidden(self, x):
        for layer in self.snn[:-1]:
            x = layer(x)
        return x

    def forward(self, x):
        return self.snn[-1](self.hidden(x))
