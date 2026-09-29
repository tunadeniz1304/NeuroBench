"""Backbones + incremental learners behind one interface used by all evaluation protocols.

A "system" exposes:
  harness_model        neurobench TorchModel whose net maps (B, T, 20) spikes -> (B, 200) scores
  learn_base()         set base-class readout from the base train split (or a subset of classes)
  learn_session(x, y)  incremental session update from the few-shot support set
  extra_state_bytes()  memory held by the learner beyond the model parameters (counted in footprint)
"""
import os
import sys

import torch
import torch.nn as nn

from neurobench.models import TorchModel

from nbfscil.sessions import load_cache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_FSCIL = os.path.join(REPO, "third_party", "neurobench", "examples", "mswc_fscil")
N_CLASSES = 200


def _upstream_snn_module():
    if UPSTREAM_FSCIL not in sys.path:
        sys.path.insert(0, UPSTREAM_FSCIL)
    import sparchSNNs
    return sparchSNNs


@torch.no_grad()
def spike_counts(layers, x, batch_size=500):
    """Per-sample spike count vector of the last hidden layer, summed over time."""
    out = []
    for i in range(0, len(x), batch_size):
        h = x[i:i + batch_size]
        for layer in layers:
            h = layer(h)
        out.append(h.sum(dim=1))
    return torch.cat(out), h.shape[1]


class UpstreamProtoSNN:
    """Reference SNN baseline: pretrained RadLIF RSNN + prototype readout (float).

    Identical to the upstream example: w_c = 2 m_c, b_c = -|m_c|^2 / T, where m_c is the
    mean per-sample spike count of class c in the last hidden layer.
    """

    def __init__(self, cfg, device):
        sp = _upstream_snn_module()
        net = sp.SNN(input_shape=(256, 201, 20), neuron_type="RadLIF",
                     layer_sizes=[1024, 1024, 200], normalization="batchnorm",
                     dropout=0.1, bidirectional=False, use_readout_layer=True).to(device)
        ckpt = cfg.get("checkpoint", os.path.join(UPSTREAM_FSCIL, "model_data", "mswc_rsnn_proto"))
        net.load_state_dict(torch.load(ckpt, map_location=device))
        old = net.snn[-1].W
        proto = nn.Linear(old.weight.shape[1], N_CLASSES, bias=True).to(device)
        proto.weight.data = old.weight.data.clone()
        proto.bias.data.zero_()
        net.snn[-1].W = proto
        self.net, self.device = net, device
        self.harness_model = TorchModel(net)  # sets eval()
        self.harness_model.add_activation_module(sp.RadLIFLayer)
        self.base_classes = cfg.get("base_classes", list(range(100)))

    def _set_protos(self, x, y, classes):
        feats, T = spike_counts(self.net.snn[:-1], x)
        W = self.net.snn[-1].W
        for c in classes:
            m = feats[y == c].mean(dim=0)
            W.weight.data[c] = 2 * m
            W.bias.data[c] = -(m @ m) / T

    def learn_base(self):
        train = load_cache("base_train")
        keep = torch.isin(train["y"], torch.as_tensor(self.base_classes))
        x, y = train["x"][keep].to(self.device).float(), train["y"][keep].to(self.device)
        self._set_protos(x, y, self.base_classes)

    def learn_session(self, x, y):
        self._set_protos(x, y, sorted(set(y.tolist())))

    def extra_state_bytes(self):
        return 0


SYSTEMS = {"upstream_proto_snn": UpstreamProtoSNN}


def build_system(cfg, device):
    return SYSTEMS[cfg["system"]](cfg, device)
