"""Phase 3 pseudo-incremental validation protocol — the only protocol used for design decisions.

Backbone trained on the pseudo-base classes only (nbfscil.train --fold k). Session 0 = pseudo-base
val split (60 classes x 100). Sessions 1..4 = 10-way 5-shot on held-out base languages; support from
their base *train* samples, cumulative queries from their base *val* samples. Accuracy uses the same
class masking as the official harness. Never loads base_test or the evaluation languages.

Readouts are linear in the last-hidden-layer spike counts, so counts are extracted once per backbone
and all learners are evaluated on them (identical to running the readout inside the SNN).

Usage: python -m nbfscil.pseudo_eval --ckpt ckpts/x.pt --learner configs/learners/<l>.yaml --seeds 0 1 2 3 4
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml

from nbfscil.models import build_model
from nbfscil.readout import build_learner
from nbfscil.sessions import load_cache, pseudo_sessions, pseudo_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(os.environ.get("NB_DATA", "/content/data"), "feats")


@torch.no_grad()
def extract_counts(model, x, device, bs=500, seed=0):
    torch.manual_seed(seed)
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(model.hidden(x[i:i + bs].to(device).float()).sum(1).to(torch.int16).cpu())
    return torch.cat(out)


def load_backbone(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


def backbone_counts(ckpt_path, device):
    """Spike counts of base train/val for a pseudo-fold backbone, cached on disk."""
    os.makedirs(FEAT_DIR, exist_ok=True)
    key = os.path.splitext(os.path.basename(ckpt_path))[0]
    path = os.path.join(FEAT_DIR, f"{key}.pt")
    if os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(ckpt_path):
        return torch.load(path)
    model, ck = load_backbone(ckpt_path, device)
    assert ck["fold"] is not None, "pseudo protocol needs a backbone trained on a pseudo fold"
    feats = {"fold": ck["fold"]}
    for split in ("base_train", "base_val"):
        d = load_cache(split)
        feats[split] = {"x": extract_counts(model, d["x"], device), "y": d["y"]}
    torch.save(feats, path)
    return feats


def masked_acc(W, b, x, y, classes):
    scores = x.float() @ W.t() + b
    mask = torch.full((W.shape[0],), float("inf"))
    mask[torch.as_tensor(classes)] = 0
    return ((scores - mask).argmax(1) == y).float().mean().item()


def run_protocol(feats, learner_cfg, seed, n_classes=200):
    fold = feats["fold"]
    base_cls, novel = pseudo_split(fold)
    tr, va = feats["base_train"], feats["base_val"]
    learner = build_learner({**learner_cfg, **({"seed": seed + 1} if "seed" not in learner_cfg else {})},
                            n_classes, tr["x"].shape[1])
    keep = torch.isin(tr["y"], torch.as_tensor(base_cls))
    learner.fit_base(tr["x"][keep], tr["y"][keep], base_cls)
    vkeep = torch.isin(va["y"], torch.as_tensor(base_cls))
    bx, by = va["x"][vkeep], va["y"][vkeep]
    W, b = learner.readout()
    accs = [masked_acc(W, b, bx, by, base_cls)]
    novel_accs = []
    for sx, sy, qx, qy, seen in pseudo_sessions(seed, novel, tr, va):
        learner.learn(sx, sy, sorted(set(sy.tolist())))
        W, b = learner.readout()
        classes = base_cls + seen
        accs.append(masked_acc(W, b, torch.cat([bx, qx]), torch.cat([by, qy]), classes))
        novel_accs.append(masked_acc(W, b, qx, qy, classes))
    return {"session_accs": accs, "session_avg": float(np.mean(accs)), "novel_accs": novel_accs}


def evaluate(feats, learner_cfg, seeds):
    runs = [run_protocol(feats, learner_cfg, s) for s in seeds]
    avg = [r["session_avg"] for r in runs]
    return {"session_avg_mean": float(np.mean(avg)), "session_avg_std": float(np.std(avg)),
            "base_mean": float(np.mean([r["session_accs"][0] for r in runs])),
            "last_mean": float(np.mean([r["session_accs"][-1] for r in runs])),
            "novel_mean": float(np.mean([np.mean(r["novel_accs"]) for r in runs])),
            "per_session_mean": np.mean([r["session_accs"] for r in runs], 0).round(4).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--learner", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = backbone_counts(args.ckpt, device)
    for lpath in args.learner:
        res = evaluate(feats, yaml.safe_load(open(lpath)), args.seeds)
        print(json.dumps({"ckpt": args.ckpt, "learner": lpath, **res}))


if __name__ == "__main__":
    main()
