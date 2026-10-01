"""Cost pre-check: harness Footprint / SynOps / sparsity of a system on the base validation split.

Runs the unmodified neurobench Benchmark exactly as nbfscil.official_eval does for session 0 (base classes,
same pre/post-processing and metrics), but on base_val, so the cost of a candidate system can be compared
with the baseline before an official run. Never touches base_test or the evaluation languages; not an official
run and not appended to results/official_runs.jsonl.

Usage: python -m nbfscil.cost_check --config configs/baseline_snn.yaml configs/final/clip_cl2n_8bit.yaml --seeds 0 1 2
"""
import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from neurobench.benchmarks import Benchmark
from neurobench.metrics.static import ConnectionSparsity, Footprint
from neurobench.metrics.workload import ActivationSparsity, ClassificationAccuracy, SynapticOperations

from nbfscil.learners import build_system
from nbfscil.official_eval import BATCH_SIZE, N_BASE, ToDeviceFloat, class_mask, post, seed_config
from nbfscil.sessions import load_cache

ALLOWED_SPLITS = ("base_val",)


def check_split(split):
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"cost_check runs on {ALLOWED_SPLITS} only, not {split!r} (test data is for official runs)")
    return split


@torch.no_grad()
def cost_seed(cfg, seed, device, data):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = seed_config(cfg, seed)
    system = build_system(cfg, device)
    system.learn_base()
    loader = DataLoader(TensorDataset(data["x"], data["y"]), batch_size=BATCH_SIZE)
    bench = Benchmark(system.harness_model,
                      metric_list=[[Footprint, ConnectionSparsity],
                                   [ClassificationAccuracy, ActivationSparsity, SynapticOperations]],
                      dataloader=loader, preprocessors=[ToDeviceFloat(device)], postprocessors=[])
    res = bench.run(postprocessors=post(class_mask(range(N_BASE), device)), device=device)
    return {"seed": seed, "checkpoint": cfg.get("checkpoint"), "acc": res["ClassificationAccuracy"],
            "footprint": res["Footprint"], "extra_footprint_bytes": system.extra_state_bytes(),
            "connection_sparsity": res["ConnectionSparsity"], "activation_sparsity": res["ActivationSparsity"],
            "dense": res["SynapticOperations"]["Dense"], "eff_macs": res["SynapticOperations"]["Effective_MACs"],
            "eff_acs": res["SynapticOperations"]["Effective_ACs"]}


def summarize(config, runs):
    out = {"config": config, "seeds": [r["seed"] for r in runs]}
    for k in ("acc", "footprint", "extra_footprint_bytes", "activation_sparsity", "dense", "eff_macs", "eff_acs"):
        v = [float(r[k]) for r in runs]
        out[k] = [float(np.mean(v)), float(np.std(v))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", default="base_val")
    args = ap.parse_args()
    data = load_cache(check_split(args.split))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in args.config:
        cfg = yaml.safe_load(open(path))
        runs = [cost_seed(cfg, s, device, data) for s in args.seeds]
        print(json.dumps({"split": args.split, **summarize(path, runs), "runs": runs}), flush=True)


if __name__ == "__main__":
    main()
