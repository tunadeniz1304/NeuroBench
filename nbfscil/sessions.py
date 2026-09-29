"""Session samplers on top of the S2S cache.

`official_sessions` reproduces neurobench's IncrementalFewShot(k_shot=5, query_shots=100,
support_query_split=(100, 100)) exactly, but with an explicit seed:
  - session order: random permutation of the 10 evaluation languages
  - each language holds 10 classes x 200 samples, ordered class by class;
    samples 0..99 of a class form the support pool, 100..199 the query pool
  - support: 5 shots per class drawn from the support pool (10-way 5-shot)
  - query: all 100 query-pool samples of every class seen so far (cumulative)

`pseudo_sessions` is the Phase 3 validation protocol. It only touches the base
train/val splits and never the base test split or the evaluation languages.
"""
import os

import numpy as np
import torch

from nbfscil.paths import CACHE_DIR
from nbfscil.cache import EVAL_LANGUAGES

N_WAY = 10
K_SHOT = 5
SAMPLES_PER_CLASS = 200
SUPPORT_POOL = 100


def load_cache(split):
    return torch.load(os.path.join(CACHE_DIR, f"{split}.pt"))


def official_sessions(seed, evaluation=None):
    """Yield (support_x, support_y, query_x, query_y, seen_novel_classes) for sessions 1..10."""
    rng = np.random.RandomState(seed)
    evaluation = evaluation if evaluation is not None else load_cache("evaluation")
    order = rng.permutation(len(EVAL_LANGUAGES))
    qx, qy, seen = [], [], []
    for li in order:
        lang = EVAL_LANGUAGES[li]
        x, y = evaluation[lang]["x"], evaluation[lang]["y"]
        n_cls = len(x) // SAMPLES_PER_CLASS
        assert n_cls == N_WAY, (lang, n_cls)
        classes = rng.permutation(n_cls)
        sx, sy = [], []
        for c in classes:
            start = c * SAMPLES_PER_CLASS
            label = int(y[start])
            assert (y[start:start + SAMPLES_PER_CLASS] == label).all()
            shots = start + rng.choice(SUPPORT_POOL, K_SHOT, replace=False)
            sx.append(x[shots]); sy.append(y[shots])
            q = torch.arange(start + SUPPORT_POOL, start + SAMPLES_PER_CLASS)
            qx.append(x[q]); qy.append(y[q])
            seen.append(label)
        yield torch.cat(sx), torch.cat(sy), torch.cat(qx), torch.cat(qy), list(seen)


# ---------------------------------------------------------------- pseudo protocol
# Base classes are 5 languages x 20 classes (en 0-19, fr 20-39, ca 40-59, de 60-79, rw 80-99).
# The official incremental sessions each bring a *new language*, so the pseudo protocol
# holds out whole languages: pseudo-base = 3 languages (60 classes), pseudo-novel = 2
# languages (40 classes) = 4 sessions of 10 classes, one language half per session.
BASE_LANG_CLASSES = {"en": range(0, 20), "fr": range(20, 40), "ca": range(40, 60),
                     "de": range(60, 80), "rw": range(80, 100)}
PSEUDO_FOLDS = {0: ("ca", "de"), 1: ("fr", "rw")}


def pseudo_split(fold=0):
    """Return (pseudo_base_classes, [pseudo-novel classes grouped per language])."""
    held = PSEUDO_FOLDS[fold]
    base = sorted(c for l, r in BASE_LANG_CLASSES.items() if l not in held for c in r)
    novel = [list(BASE_LANG_CLASSES[l]) for l in held]
    return base, novel


def pseudo_sessions(seed, novel_by_lang, train, val):
    """Pseudo-incremental sessions from base data only.

    Each held-out language is cut into two random 10-class sessions; session order is
    random. Support shots come from the base *train* split of pseudo-novel classes,
    queries from the base *val* split (100 per class), accumulated over sessions.
    """
    rng = np.random.RandomState(seed)
    groups = []
    for classes in novel_by_lang:
        perm = [classes[i] for i in rng.permutation(len(classes))]
        groups += [perm[i:i + N_WAY] for i in range(0, len(perm), N_WAY)]
    groups = [groups[i] for i in rng.permutation(len(groups))]
    qx, qy, seen = [], [], []
    for classes in groups:
        sx, sy = [], []
        for c in classes:
            pool = torch.nonzero(train["y"] == c).squeeze(1)
            shots = pool[rng.choice(len(pool), K_SHOT, replace=False)]
            sx.append(train["x"][shots]); sy.append(train["y"][shots])
            q = torch.nonzero(val["y"] == c).squeeze(1)
            qx.append(val["x"][q]); qy.append(val["y"][q])
            seen.append(c)
        yield torch.cat(sx), torch.cat(sy), torch.cat(qx), torch.cat(qy), list(seen)
