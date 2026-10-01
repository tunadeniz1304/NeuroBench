"""Report figures from results/official_runs.jsonl only (no other numbers enter the plots).

Usage: python scripts/make_figures.py  ->  docs/figures/sessions.png, docs/figures/pareto.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "figures")
BASELINE = "configs/baseline_snn.yaml"
FINAL = "configs/final/clip_cl2n_8bit.yaml"
LEADERBOARD = {"SNN (leaderboard)": (75.27, 1.36e7), "M5 ANN (leaderboard)": (89.27, 6.03e6)}


def load():
    runs = {}
    for line in open(os.path.join(REPO, "results", "official_runs.jsonl")):
        r = json.loads(line)
        if r["kind"] == "OFFICIAL_EVAL":
            runs[r["config"]] = r  # the last run of a config wins
    return runs[BASELINE], runs[FINAL]


def sessions_figure(base, ours):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for run, label, color in ((base, "SNN baseline (rerun)", "tab:gray"), (ours, "Ours (Hebbian CL2N 8 bit)", "tab:blue")):
        acc = np.array([r["session_accs"] for r in run["runs"]]) * 100
        nov = np.array([r["query_accs"] for r in run["runs"]]) * 100
        s = np.arange(acc.shape[1])
        ax[0].errorbar(s, acc.mean(0), yerr=acc.std(0), marker="o", ms=4, capsize=2, color=color, label=label)
        ax[1].errorbar(s[1:], nov.mean(0), yerr=nov.std(0), marker="o", ms=4, capsize=2, color=color, label=label)
    ax[0].set(xlabel="session", ylabel="accuracy, all seen classes (%)", title="Accuracy per session")
    ax[1].set(xlabel="session", ylabel="accuracy, novel-class queries (%)", title="Novel classes only")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sessions.png"), dpi=150)


def pareto_figure(base, ours):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for run, label, color, marker in ((base, "SNN baseline (rerun)", "tab:gray", "s"),
                                      (ours, "Ours, per backbone seed", "tab:blue", "o")):
        acc = [r["session_avg"] * 100 for r in run["runs"]]
        acs = [np.mean(r["eff_acs"]) / 1e6 for r in run["runs"]]
        fp = [(r["footprint"] + r.get("extra_footprint_bytes", 0)) / 1e6 for r in run["runs"]]
        ax[0].scatter(acs, acc, color=color, marker=marker, label=label)
        ax[1].scatter(fp, acc, color=color, marker=marker, label=label)
        for r, x, y in zip(run["runs"], acs, acc):
            if run is ours:
                ax[0].annotate(f"s{r['seed']}", (x, y), textcoords="offset points", xytext=(5, -3), fontsize=8)
    b_acs = np.mean([np.mean(r["eff_acs"]) for r in base["runs"]]) / 1e6
    ax[0].axvline(b_acs, color="tab:gray", ls="--", lw=0.8)
    for (name, (acc, fp)), m in zip(LEADERBOARD.items(), ("^", "v")):
        ax[1].scatter([fp / 1e6], [acc], color="k", marker=m, facecolors="none", label=name)
    ax[0].set(xlabel="Eff_ACs per sample, mean over sessions (M, neurobench 2.3.0, summed over T=200)",
              ylabel="session average (%)", title="Accuracy vs. synaptic operations")
    ax[1].set(xlabel="footprint incl. learner state (MB)", ylabel="session average (%)",
              title="Accuracy vs. footprint")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].xaxis.label.set_size(8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pareto.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    base, ours = load()
    sessions_figure(base, ours)
    pareto_figure(base, ours)
    print("wrote", OUT)
