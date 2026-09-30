"""A training run interrupted after an epoch and resumed must end bit-identical to an uninterrupted run
(CPU / no-graph path; the CUDA-graph path restores the same state in place)."""
import numpy as np
import pytest
import torch

import nbfscil.train as tr


def _fake_cache(split):
    g = torch.Generator().manual_seed(0 if split.endswith("train") else 1)
    n = 400 if split.endswith("train") else 200
    return {"x": torch.randint(-1, 2, (n, 12, 20), generator=g).to(torch.int8), "y": torch.arange(n) % 100}


def _cfg(loss):
    return {"model": {"name": "rsnn", "layer_sizes": [16, 16], "dropout": 0.1, "init_state": "random"},
            "loss": loss,
            "train": {"lr": 0.01, "weight_decay": 0.0001, "epochs": 3, "schedule": "step", "step_size": 2,
                      "batch_size": 64, "eval_every": 1}}


class _Crash(Exception):
    pass


@pytest.mark.parametrize("loss,extra", [({"name": "ce"}, {}),
                                        ({"name": "cosine", "scale": 16, "center": True}, {}),
                                        ({"name": "cosine", "scale": 16, "center": True}, {"grad_clip_rel": 1.2})])
def test_resume_is_bit_identical(tmp_path, monkeypatch, loss, extra):
    monkeypatch.setattr(tr, "load_cache", _fake_cache)
    cfg, dev = _cfg(loss), torch.device("cpu")
    cfg["train"].update(extra)

    full = str(tmp_path / "full.pt")
    hist_full = tr.train(cfg, 3, None, full, dev, use_graph=False)

    # crash while logging epoch 2, i.e. before its checkpoint and resume state are written
    part = str(tmp_path / "part.pt")
    real_print = print
    def crashing_print(*a, **k):
        if a and '"epoch": 2' in str(a[0]):
            raise _Crash
        real_print(*a, **k)
    monkeypatch.setattr(tr, "print", crashing_print, raising=False)
    with pytest.raises(_Crash):
        tr.train(cfg, 3, None, part, dev, use_graph=False)
    assert torch.load(part + ".resume", weights_only=False)["epoch"] == 1

    # a fresh process would reseed at start; resuming must still reproduce the uninterrupted run
    monkeypatch.setattr(tr, "print", real_print, raising=False)
    torch.manual_seed(12345); np.random.seed(12345)
    hist_part = tr.train(cfg, 3, None, part, dev, use_graph=False)

    a, b = torch.load(full, weights_only=False), torch.load(part, weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    strip = lambda h: [{k: v for k, v in r.items() if k != "sec"} for r in h]
    assert strip(hist_full) == strip(hist_part)
    assert not (tmp_path / "part.pt.resume").exists() and not (tmp_path / "full.pt.resume").exists()


def test_resume_ignored_for_other_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "load_cache", _fake_cache)
    cfg, dev = _cfg({"name": "ce"}), torch.device("cpu")
    out = str(tmp_path / "x.pt")
    tr.train(cfg, 0, None, out, dev, use_graph=False)
    ref = torch.load(out, weights_only=False)["model"]
    # a stale resume file from another seed must not be picked up
    tr.save_resume(out + ".resume", cfg, 7, None, 2, [], tr.build_model(cfg["model"]),
                   tr.build_loss(cfg["loss"], tr.build_model(cfg["model"])),
                   torch.optim.Adam(tr.build_model(cfg["model"]).parameters()))
    tr.train(cfg, 0, None, out, dev, use_graph=False)
    again = torch.load(out, weights_only=False)["model"]
    assert all(torch.equal(ref[k], again[k]) for k in ref)
