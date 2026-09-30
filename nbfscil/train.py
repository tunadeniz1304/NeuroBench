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

    amp = tcfg.get("amp", False)
    clip_rel = tcfg.get("grad_clip_rel")
    gn_ema = torch.zeros((), device=device)   # EMA of accepted gradient norms (grad_clip_rel)

    def step():
        opt.zero_grad(set_to_none=False)
        xb = augment(sx)
        # fp16 matmuls, fp32 neuron state; no loss scaling (initial logits are large enough that
        # scaling overflows), non-finite gradients are zeroed instead (graph-safe, no host sync)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = loss_fn(model, xb, sy)
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        nonfinite = torch.stack([(~torch.isfinite(g)).any() for g in grads]).any().float()
        if amp:
            for g in grads:
                torch.nan_to_num_(g, nan=0.0, posinf=0.0, neginf=0.0)
        gnorm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        clipped = torch.zeros((), device=gnorm.device)
        if clip_rel:
            # Spike clipping, scale-free: the norm is capped at clip_rel x the EMA of recent (capped) norms.
            # Adam is invariant to a constant gradient scale, so only steps far above the recent level change.
            thr = torch.where(gn_ema > 0, clip_rel * gn_ema, gnorm)     # the first step only seeds the EMA
            coef = (thr / (gnorm + 1e-6)).clamp(max=1.0)
            for g in grads:
                g.mul_(coef)
            clipped = (coef < 1).float()
            gn_ema.copy_(torch.where(gn_ema > 0, 0.99 * gn_ema + 0.01 * torch.minimum(gnorm, thr), gnorm))
        if tcfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        opt.step()
        return loss, gnorm, nonfinite, clipped

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
            static_out = step()

    # Resume after a runtime reset. Restored only after graph capture, and in place, so the captured graph keeps
    # pointing at the same parameter / optimizer tensors (the warm-up steps above are overwritten).
    resume_path = out_path + ".resume"
    history, start = [], 0
    state = load_resume(resume_path, cfg, seed, fold)
    if state is not None:
        restore_state(state, model, loss_fn, opt, params, lr_t, gn_ema)
        history, start = state["history"], state["epoch"]
        print(json.dumps({"resumed_from_epoch": start}), flush=True)

    for ep in range(start, epochs):
        model.train()
        t0 = time.time()
        tot, gsum, gmax, nbad, nclip = (torch.zeros((), device=device) for _ in range(5))
        perm = torch.randperm(len(X), device=device)
        for k in range(steps_per_epoch):
            lr_t.fill_(lr_at(tcfg, ep + k / steps_per_epoch))
            load_batch(perm[k * B:(k + 1) * B])
            if use_graph:
                graph.replay()
                loss, gn, bad, clp = static_out
            else:
                loss, gn, bad, clp = step()
            tot += loss.detach()
            gsum += gn
            gmax = torch.maximum(gmax, gn)
            nbad += bad
            nclip += clp
        # gnorm_*: total gradient norm after zeroing non-finite entries, before clipping
        rec = {"epoch": ep + 1, "loss": tot.item() / steps_per_epoch, "sec": round(time.time() - t0, 1),
               "lr": float(lr_t), "gnorm_mean": gsum.item() / steps_per_epoch, "gnorm_max": gmax.item(),
               "nonfinite_steps": int(nbad.item()), "clipped_steps": int(nclip.item())}
        if (ep + 1) % tcfg.get("eval_every", 5) == 0 or ep + 1 == epochs:
            rec["val_acc"] = evaluate(model, xva, yva, classes, device,
                                      logits=lambda xb: loss_fn.logits(model, xb))
        history.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "cfg": cfg, "seed": seed, "fold": fold,
                    "epoch": ep + 1, "history": history}, out_path)
        save_resume(resume_path, cfg, seed, fold, ep + 1, history, model, loss_fn, opt, gn_ema)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    return history


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_resume(path, cfg, seed, fold, epoch, history, model, loss_fn, opt, gn_ema=None):
    """Full training state after `epoch`, written atomically next to the checkpoint."""
    tmp = path + ".tmp"
    torch.save({"cfg": cfg, "seed": seed, "fold": fold, "epoch": epoch, "history": history,
                "model": model.state_dict(), "loss": loss_fn.state_dict(), "opt": opt.state_dict(),
                "gn_ema": None if gn_ema is None else gn_ema.detach().cpu(), "rng": rng_state()}, tmp)
    os.replace(tmp, path)


def load_resume(path, cfg, seed, fold):
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["cfg"], state["seed"], state["fold"]) != (cfg, seed, fold):
        print(json.dumps({"resume_ignored": path, "reason": "different cfg/seed/fold"}), flush=True)
        return None
    return state


def restore_state(state, model, loss_fn, opt, params, lr_t, gn_ema):
    model.load_state_dict(state["model"])        # copies into the existing tensors
    loss_fn.load_state_dict(state["loss"])
    if state.get("gn_ema") is not None:          # resume files written before grad_clip_rel have none
        gn_ema.copy_(state["gn_ema"])
    saved = state["opt"]["state"]
    if all(opt.state.get(p) for i, p in enumerate(params) if i in saved):
        for i, p in enumerate(params):           # graph path: optimizer state exists after warm-up
            for k, v in saved.get(i, {}).items():
                opt.state[p][k].copy_(v)
    else:
        opt.load_state_dict(state["opt"])
        for g in opt.param_groups:               # keep the shared LR tensor driven by lr_at()
            g["lr"] = lr_t
    rng = state["rng"]
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


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
