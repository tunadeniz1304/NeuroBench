"""Official NeuroBench MSWC FSCIL evaluation on the S2S cache.

Mirrors examples/mswc_fscil/mswc_fscil.py (SPIKING branch) step by step, using the
unmodified neurobench Benchmark, metrics and post-processing, with three changes that
do not alter the task: (1) inputs come from the S2S cache instead of being re-encoded,
(2) everything is seeded, (3) several repeats (seeds) are run and aggregated.

Every invocation is one OFFICIAL EVAL RUN and is appended to results/official_runs.jsonl.
Budget: at most 5 official runs in total (see CLAUDE.md).

Usage: python -m nbfscil.official_eval --config configs/<name>.yaml --seeds 0 1 2
"""
import argparse
import datetime
import json
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from neurobench.benchmarks import Benchmark
from neurobench.metrics.static import ConnectionSparsity, Footprint
from neurobench.metrics.workload import ActivationSparsity, ClassificationAccuracy, SynapticOperations
from neurobench.processors.abstract.postprocessor import NeuroBenchPostProcessor
from neurobench.processors.abstract.preprocessor import NeuroBenchPreProcessor

from nbfscil.learners import build_system
from nbfscil.sessions import load_cache, official_sessions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CLASSES = 200
N_BASE = 100
BATCH_SIZE = 256


class ToDeviceFloat(NeuroBenchPreProcessor):
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        return batch[0].to(self.device).float(), batch[1].to(self.device)


class OutMask(NeuroBenchPostProcessor):
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, out):
        return out - self.mask


class Softmax(NeuroBenchPostProcessor):
    def __call__(self, out):
        return F.softmax(out, dim=-1)


class Out2Pred(NeuroBenchPostProcessor):
    def __call__(self, out):
        return torch.argmax(out, dim=-1)


class SqueezeOut(NeuroBenchPostProcessor):
    def __call__(self, out):
        return torch.squeeze(out)


def class_mask(classes, device):
    mask = torch.full((N_CLASSES,), float("inf"), device=device)
    mask[torch.as_tensor(classes, dtype=torch.long)] = 0
    return mask


def post(mask):
    return [OutMask(mask), Softmax(), Out2Pred(), SqueezeOut()]


def masked_accuracy(net, x, y, mask, device, reset_hooks):
    correct = 0
    for i in range(0, len(x), BATCH_SIZE):
        out = net(x[i:i + BATCH_SIZE].to(device).float()) - mask
        reset_hooks()
        correct += (out.argmax(-1).cpu() == y[i:i + BATCH_SIZE]).sum().item()
    return correct / len(x)


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@torch.no_grad()  # the harness does not disable autograd; metrics are unaffected
def run_seed(cfg, seed, device, base_test, evaluation):
    torch.manual_seed(seed)
    np.random.seed(seed)
    system = build_system(cfg, device)  # loads backbone, builds harness model + incremental learner
    system.learn_base()                 # base prototypes / readout from base train split only

    test_ds = TensorDataset(base_test["x"], base_test["y"])
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    pre = [ToDeviceFloat(device)]
    bench = Benchmark(system.harness_model,
                      metric_list=[[Footprint, ConnectionSparsity],
                                   [ClassificationAccuracy, ActivationSparsity, SynapticOperations]],
                      dataloader=loader, preprocessors=pre, postprocessors=[])

    sessions = []
    res = bench.run(postprocessors=post(class_mask(range(N_BASE), device)), device=device)
    sessions.append(res)
    print(f"[seed {seed}] session 0: {res}", flush=True)

    # forward passes outside bench.run() also fire the harness activation hooks; clear them
    reset_hooks = lambda: bench.workload_metric_manager.reset_hooks(system.harness_model)
    query_accs = []
    for s, (sx, sy, qx, qy, seen) in enumerate(official_sessions(seed, evaluation), start=1):
        system.learn_session(sx.to(device).float(), sy.to(device))
        reset_hooks()
        full = ConcatDataset([test_ds, TensorDataset(qx, qy)])
        mask = class_mask(list(range(N_BASE)) + seen, device)
        res = bench.run(dataloader=DataLoader(full, batch_size=BATCH_SIZE), postprocessors=post(mask), device=device)
        sessions.append(res)
        # novel-class accuracy (diagnostic only). Computed directly with the same mask/argmax:
        # a second Benchmark would register a second set of activation hooks on the model.
        query_accs.append(masked_accuracy(system.harness_model.__net__(), qx, qy, mask, device, reset_hooks))
        print(f"[seed {seed}] session {s}: acc {res['ClassificationAccuracy']:.4f} "
              f"novel {query_accs[-1]:.4f}", flush=True)

    accs = [r["ClassificationAccuracy"] for r in sessions]
    return {
        "seed": seed,
        "session_accs": accs,
        "base_acc": accs[0],
        "session_avg": float(np.mean(accs)),
        "query_accs": query_accs,
        "footprint": sessions[0]["Footprint"],
        "connection_sparsity": sessions[0]["ConnectionSparsity"],
        "activation_sparsity": [r["ActivationSparsity"] for r in sessions],
        "dense": [r["SynapticOperations"]["Dense"] for r in sessions],
        "eff_macs": [r["SynapticOperations"]["Effective_MACs"] for r in sessions],
        "eff_acs": [r["SynapticOperations"]["Effective_ACs"] for r in sessions],
        "extra_footprint_bytes": system.extra_state_bytes(),
    }


def _synthetic_test_data(n_test=300, T=200):
    from nbfscil.cache import EVAL_LANGUAGES
    g = torch.Generator().manual_seed(0)
    rnd = lambda n: torch.randint(-1, 2, (n, T, 20), generator=g).to(torch.int8)
    base_test = {"x": rnd(n_test), "y": torch.arange(n_test) % N_BASE}
    evaluation = {}
    for li, lang in enumerate(EVAL_LANGUAGES):
        y = torch.as_tensor(np.repeat(np.arange(100 + 10 * li, 110 + 10 * li), 200))
        evaluation[lang] = {"x": rnd(len(y)), "y": y}
    return base_test, evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--note", default="")
    ap.add_argument("--smoke", action="store_true", help="run on random spikes; not an official run, not logged")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        # pipeline check on random spikes: touches neither the test split nor the evaluation languages
        base_test, evaluation = _synthetic_test_data()
    else:
        base_test = load_cache("base_test")
        evaluation = load_cache("evaluation")
    runs = [run_seed(cfg, s, device, base_test, evaluation) for s in args.seeds]
    if args.smoke:
        print("SMOKE OK", [round(r["session_avg"], 3) for r in runs])
        return

    summary = {
        "time": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "kind": "OFFICIAL_EVAL",
        "config": args.config,
        "git": git_hash(),
        "seeds": args.seeds,
        "note": args.note,
        "base_acc_mean": float(np.mean([r["base_acc"] for r in runs])),
        "base_acc_std": float(np.std([r["base_acc"] for r in runs])),
        "session_avg_mean": float(np.mean([r["session_avg"] for r in runs])),
        "session_avg_std": float(np.std([r["session_avg"] for r in runs])),
        "runs": runs,
    }
    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    with open(os.path.join(REPO, "results", "official_runs.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
