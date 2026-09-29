"""nbfscil.snn.RSNN must compute exactly what the upstream sparch RSNN computes."""
import os

import pytest
import torch

from nbfscil.learners import UPSTREAM_FSCIL, _upstream_snn_module
from nbfscil.snn import RSNN

_TORCH_RAND = torch.rand

pytestmark = pytest.mark.skipif(not os.path.isdir(UPSTREAM_FSCIL), reason="upstream repo not cloned")


def _pair(sizes=(64, 48)):
    sp = _upstream_snn_module()
    ref = sp.SNN(input_shape=(8, 30, 20), neuron_type="RadLIF", layer_sizes=list(sizes) + [200],
                 normalization="batchnorm", dropout=0.1, bidirectional=False,
                 use_readout_bias=False, use_readout_layer=True)
    ours = RSNN(layer_sizes=sizes)
    ours.load_state_dict(ref.state_dict())
    return ref.eval(), ours.eval()


def _patch_rand_to_cpu_seeded(monkeypatch, seed):
    """Make both implementations draw identical initial states."""
    gen = torch.Generator().manual_seed(seed)
    def rand(*shape, device=None, **kw):
        return _TORCH_RAND(*shape, generator=gen).to(device if device is not None else "cpu")
    monkeypatch.setattr(torch, "rand", rand)


def test_forward_identical(monkeypatch):
    ref, ours = _pair()
    x = torch.randint(-1, 2, (8, 30, 20)).float()
    _patch_rand_to_cpu_seeded(monkeypatch, 0)
    out_ref = ref(x)
    _patch_rand_to_cpu_seeded(monkeypatch, 0)
    out_ours = ours(x)
    assert torch.allclose(out_ref, out_ours, atol=1e-5)


def test_upstream_checkpoint_loads():
    ckpt = os.path.join(UPSTREAM_FSCIL, "model_data", "mswc_rsnn_proto")
    ours = RSNN(layer_sizes=(1024, 1024))
    missing, unexpected = ours.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    assert not missing and not unexpected
