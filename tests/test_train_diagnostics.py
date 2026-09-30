"""Per-epoch gradient diagnostics and relative (spike) gradient clipping in nbfscil.train (CPU / no-graph path)."""
import math

import torch

import nbfscil.train as tr
from test_train_resume import _cfg, _fake_cache

COS = {"name": "cosine", "scale": 16, "center": True}


def _run(tmp_path, monkeypatch, name, **train):
    monkeypatch.setattr(tr, "load_cache", _fake_cache)
    cfg = _cfg(COS)
    cfg["train"].update(train)
    out = str(tmp_path / f"{name}.pt")
    hist = tr.train(cfg, 0, None, out, torch.device("cpu"), use_graph=False)
    return hist, torch.load(out, weights_only=False)["model"]


def test_history_has_gradient_diagnostics(tmp_path, monkeypatch):
    hist, _ = _run(tmp_path, monkeypatch, "plain")
    for r in hist:
        assert math.isfinite(r["gnorm_mean"]) and r["gnorm_mean"] > 0
        assert r["gnorm_max"] >= r["gnorm_mean"]
        assert r["nonfinite_steps"] == 0 and r["clipped_steps"] == 0


def test_loose_relative_clip_changes_nothing(tmp_path, monkeypatch):
    hist_a, a = _run(tmp_path, monkeypatch, "plain")
    hist_b, b = _run(tmp_path, monkeypatch, "loose", grad_clip_rel=1e6)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert all(r["clipped_steps"] == 0 for r in hist_b)
    strip = lambda h: [{k: v for k, v in r.items() if k != "sec"} for r in h]
    assert strip(hist_a) == strip(hist_b)


def test_tight_relative_clip_clips_and_changes_training(tmp_path, monkeypatch):
    _, a = _run(tmp_path, monkeypatch, "plain")
    hist, b = _run(tmp_path, monkeypatch, "tight", grad_clip_rel=1e-3)
    steps = len(_fake_cache("base_train")["y"]) // 64
    # every step but the first (which only seeds the EMA) is capped
    assert hist[0]["clipped_steps"] == steps - 1
    assert all(r["clipped_steps"] == steps for r in hist[1:])
    # logged norms are measured before clipping, so they stay on the unclipped scale
    assert all(r["gnorm_mean"] > 0 for r in hist)
    assert not all(torch.equal(a[k], b[k]) for k in a)
