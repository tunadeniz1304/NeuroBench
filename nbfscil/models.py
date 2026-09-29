"""Model zoo. Every model maps (B, T, 20) S2S spikes -> (B, 200) class scores and exposes
`hidden(x)` -> (B, T, H) spikes of the last hidden layer, used by the incremental learners."""
import torch
import torch.nn as nn

from nbfscil.learners import _upstream_snn_module


class UpstreamRSNN(nn.Module):
    """The NeuroBench SNN baseline architecture (sparch RadLIF RSNN + sum-over-time readout)."""

    def __init__(self, layer_sizes=(1024, 1024), dropout=0.1, neuron_type="RadLIF"):
        super().__init__()
        sp = _upstream_snn_module()
        self.net = sp.SNN(input_shape=(256, 201, 20), neuron_type=neuron_type,
                          layer_sizes=list(layer_sizes) + [200], normalization="batchnorm",
                          dropout=dropout, bidirectional=False, use_readout_bias=False,
                          use_readout_layer=True)
        self.activation_module = getattr(sp, neuron_type + "Layer")

    def hidden(self, x):
        for layer in self.net.snn[:-1]:
            x = layer(x)
        return x

    def forward(self, x):
        return self.net(x)


from nbfscil.snn import RSNN

MODELS = {"upstream_rsnn": UpstreamRSNN, "rsnn": RSNN}


def build_model(mcfg):
    mcfg = dict(mcfg)
    return MODELS[mcfg.pop("name")](**mcfg)
