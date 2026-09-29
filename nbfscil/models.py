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


class M5Net(nn.Module):
    """NeuroBench M5 ANN baseline (MFCC input), used only as the non-spiking reference."""
    is_spiking = False
    cache_prefix = "mfcc_"

    def __init__(self, n_channel=256):
        super().__init__()
        from nbfscil.learners import UPSTREAM_FSCIL
        import sys
        if UPSTREAM_FSCIL not in sys.path:
            sys.path.insert(0, UPSTREAM_FSCIL)
        from M5 import M5
        self.net = M5(n_input=20, stride=2, n_channel=n_channel, n_output=200,
                      input_kernel=4, pool_kernel=2, drop=True)

    def hidden(self, x):
        return self.net(x, features_out=True).unsqueeze(1)  # (B, 1, 512): "counts" = features

    def forward(self, x):
        return self.net(x)

MODELS = {"upstream_rsnn": UpstreamRSNN, "rsnn": RSNN, "m5": M5Net}


def cache_prefix(mcfg):
    return getattr(MODELS[mcfg["name"]], "cache_prefix", "")


def build_model(mcfg):
    mcfg = dict(mcfg)
    return MODELS[mcfg.pop("name")](**mcfg)
