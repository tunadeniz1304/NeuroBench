"""Data-split integrity: no test or future-session data can reach training."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from nbfscil.paths import DATA_ROOT, CACHE_DIR
from nbfscil.sessions import (BASE_LANG_CLASSES, K_SHOT, N_WAY, PSEUDO_FOLDS, SUPPORT_POOL,
                              SAMPLES_PER_CLASS, official_sessions, pseudo_sessions, pseudo_split)

MSWC_DIR = os.path.join(DATA_ROOT, "MSWC")
needs_data = pytest.mark.skipif(not os.path.exists(os.path.join(MSWC_DIR, "base_train.csv")),
                                reason="MSWC data not available")
needs_cache = pytest.mark.skipif(not os.path.exists(os.path.join(CACHE_DIR, "evaluation.pt")),
                                 reason="S2S cache not built")


@needs_data
def test_base_splits_disjoint():
    files = {s: set(pd.read_csv(os.path.join(MSWC_DIR, f"base_{s}.csv")).LINK)
             for s in ("train", "val", "test")}
    assert not files["train"] & files["test"]
    assert not files["val"] & files["test"]
    assert not files["train"] & files["val"]


@needs_data
def test_base_languages_match_class_blocks():
    d = pd.read_csv(os.path.join(MSWC_DIR, "base_train.csv"))
    from neurobench.datasets.MSWC_dataset import FSCIL_KEYWORDS
    idx = {w: i for i, w in enumerate(FSCIL_KEYWORDS)}
    for word, lang in d.groupby("WORD").LANGUAGE.first().items():
        assert idx[word] in BASE_LANG_CLASSES[lang]


def test_pseudo_split_holds_out_whole_languages():
    for fold, held in PSEUDO_FOLDS.items():
        base, novel = pseudo_split(fold)
        flat = [c for g in novel for c in g]
        assert len(base) == 60 and len(flat) == 40
        assert not set(base) & set(flat)
        assert set(flat) == {c for l in held for c in BASE_LANG_CLASSES[l]}
        assert all(c < 100 for c in base + flat)


def _fake_split(classes, per_class):
    y = torch.as_tensor(np.repeat(classes, per_class))
    x = torch.arange(len(y))[:, None, None].expand(-1, 1, 1).clone()  # x encodes sample id
    return {"x": x, "y": y}


def test_pseudo_sessions_structure_and_sources():
    base, novel = pseudo_split(0)
    train = _fake_split(list(range(100)), 500)
    val = _fake_split(list(range(100)), 100)
    val["x"] = val["x"] + 10**7  # val sample ids live in a disjoint range
    seen_all = []
    for s, (sx, sy, qx, qy, seen) in enumerate(pseudo_sessions(3, novel, train, val)):
        new = sorted(set(sy.tolist()))
        assert len(new) == N_WAY and len(sy) == N_WAY * K_SHOT
        assert all((sy == c).sum() == K_SHOT for c in new)
        assert not set(new) & set(base)          # pseudo-base classes never in support
        assert not set(new) & set(seen_all)      # every session brings new classes
        assert (sx < 10**7).all()                # support from base train only
        assert (qx >= 10**7).all()               # queries from base val only
        seen_all += new
        assert set(qy.tolist()) == set(seen_all)  # cumulative query of seen classes only
        assert len(qy) == 100 * len(seen_all)
    assert s == 3 and sorted(seen_all) == sorted(c for g in novel for c in g)


def _fake_evaluation():
    from nbfscil.cache import EVAL_LANGUAGES
    ev = {}
    for li, lang in enumerate(EVAL_LANGUAGES):
        classes = list(range(100 + 10 * li, 110 + 10 * li))
        y = torch.as_tensor(np.repeat(classes, SAMPLES_PER_CLASS))
        within = torch.as_tensor(np.tile(np.arange(SAMPLES_PER_CLASS), len(classes)))
        ev[lang] = {"x": within[:, None, None].clone(), "y": y}
    return ev


def test_official_sessions_support_query_disjoint_and_cumulative():
    ev = _fake_evaluation()
    seen_all = []
    for s, (sx, sy, qx, qy, seen) in enumerate(official_sessions(7, ev)):
        new = sorted(set(sy.tolist()))
        assert len(new) == N_WAY and len(sy) == N_WAY * K_SHOT
        assert (sx < SUPPORT_POOL).all()     # support pool: samples 0..99 of each class
        assert (qx >= SUPPORT_POOL).all()    # query pool: samples 100..199
        assert not set(new) & set(seen_all)
        seen_all += new
        assert sorted(seen) == sorted(seen_all)
        assert len(qy) == 100 * len(seen_all)
        assert all(c >= 100 for c in new)    # only evaluation classes
    assert s == 9


def test_official_sessions_seeded():
    ev = _fake_evaluation()
    a = [sx.clone() for sx, *_ in official_sessions(1, ev)]
    b = [sx.clone() for sx, *_ in official_sessions(1, ev)]
    c = [sx.clone() for sx, *_ in official_sessions(2, ev)]
    assert all(torch.equal(i, j) for i, j in zip(a, b))
    assert not all(torch.equal(i, j) for i, j in zip(a, c))


@needs_cache
def test_cache_shapes_and_labels():
    from nbfscil.sessions import load_cache
    tr, te = load_cache("base_train"), load_cache("base_test")
    assert tr["x"].shape[1:] == (200, 20) and tr["x"].dtype == torch.int8
    assert len(tr["y"]) == 50000 and len(te["y"]) == 10000
    assert set(tr["y"].tolist()) == set(range(100))
    ev = load_cache("evaluation")
    labels = torch.cat([v["y"] for v in ev.values()])
    assert set(labels.tolist()) == set(range(100, 200))
