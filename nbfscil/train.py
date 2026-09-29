"""Offline base-session training (backprop allowed) on the S2S cache.

Trains on the base *train* split restricted to `train_classes` (all 100 for the official
backbone, the pseudo-base classes for the Phase 3 protocol) and reports accuracy on the
base *val* split of the same classes. The base test split is never loaded here.

The whole step (augment, forward, loss, backward, Adam) is captured once in a CUDA graph and
replayed; the 200-step recurrent loop otherwise spends most of its time launching tiny kernels.

Usage: python -m nbfscil.train --config configs/<name>.yaml --seed 0 [--fold 0] --out ckpt.pt
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from nbfscil.augment import build_augment
from nbfscil.losses import build_loss
from nbfscil.models import build_model, cache_prefix
from nbfscil.sessions import load_cache, pseudo_split


def subset(split, classes):
    keep = torch.isin(split["y"], torch.as_tensor(classes))
    return split["x"][keep], split["y"][keep]


@torch.no_grad()
def evaluate(model, x, y, classes, device, bs=500, logits=None):
    model.eval()
    logits = logits or model
    mask = torch.full((200,), float("inf"), device=device)
    mask[torch.as_tensor(classes)] = 0
    correct = 0
    for i in range(0, len(x), bs):
        out = logits(x[i:i + bs].to(device).float()) - mask
        correct += (out.argmax(-1).cpu() == y[i:i + bs]).sum().item()
    return correct / len(x)


def lr_at(tcfg, epoch_float):
    base, epochs = tcfg["lr"], tcfg["epochs"]
    warm = tcfg.get("warmup_epochs", 0)
    if epoch_float < warm:
        return base * (epoch_float + 1e-3) / warm
    if tcfg.get("schedule", "step") == "step":
        return base * 0.1 ** int(epoch_float // tcfg.get("step_size", 20))
    p = (epoch_float - warm) / max(epochs - warm, 1e-9)
    return base * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def train(cfg, seed, fold, out_path, device, use_graph=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    classes = list(range(100)) if fold is None else pseudo_split(fold)[0]
    pre = cache_prefix(cfg["model"])
    xtr, ytr = subset(load_cache(pre + "base_train"), classes)
    xva, yva = subset(load_cache(pre + "base_val"), classes)
    tcfg = cfg["train"]
    B = tcfg.get("batch_size", 256)
    epochs = tcfg["epochs"]
    steps_per_epoch = len(xtr) // B

    model = build_model(cfg["model"]).to(device)
    loss_fn = build_loss(cfg.get("loss", {"name": "ce"}), model).to(device)
    augment = build_augment(cfg.get("augment", {}))
    params = list(model.parameters()) + list(loss_fn.parameters())
    lr_t = torch.tensor(lr_at(tcfg, 0), device=device)
    opt = torch.optim.Adam(params, lr=lr_t, weight_decay=tcfg.get("weight_decay", 1e-4),
                           capturable=use_graph)

    X = xtr.to(device)                       # int8 on GPU, ~200 MB
    Y = ytr.to(device)
    sx = torch.zeros((B,) + tuple(X.shape[1:]), device=device)
    sy = torch.zeros(B, dtype=torch.long, device=device)

    def step():
        opt.zero_grad(set_to_none=False)
        xb = augment(sx)
        loss = loss_fn(model, xb, sy)
        loss.backward()
        if tcfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        opt.step()
        return loss

    def load_batch(idx):
        sx.copy_(X[idx].float())
        sy.copy_(Y[idx])

    model.train()
    if use_graph:
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                load_batch(torch.randint(len(X), (B,), device=device))
                step()
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_loss = step()

    history = []
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), torch.zeros((), device=device)
        perm = torch.randperm(len(X), device=device)
        for k in range(steps_per_epoch):
            lr_t.fill_(lr_at(tcfg, ep + k / steps_per_epoch))
            load_batch(perm[k * B:(k + 1) * B])
            if use_graph:
                graph.replay()
                tot += static_loss.detach()
            else:
                tot += step().detach()
        rec = {"epoch": ep + 1, "loss": tot.item() / steps_per_epoch, "sec": round(time.time() - t0, 1),
               "lr": float(lr_t)}
        if (ep + 1) % tcfg.get("eval_every", 5) == 0 or ep + 1 == epochs:
            rec["val_acc"] = evaluate(model, xva, yva, classes, device,
                                      logits=lambda xb: loss_fn.logits(model, xb))
        history.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "cfg": cfg, "seed": seed, "fold": fold,
                    "epoch": ep + 1, "history": history}, out_path)
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=None, help="pseudo-protocol fold; omit for all 100 base classes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke runs)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(cfg, args.seed, args.fold, args.out, device, use_graph=not args.no_graph and device.type == "cuda")


if __name__ == "__main__":
    main()
