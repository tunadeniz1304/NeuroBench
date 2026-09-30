<div align="center">

# Hardware-Plausible SNN for NeuroBench Keyword FSCIL

**Spiking keyword spotting that learns new words on-device with a local, integer, three-factor rule**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](requirements.txt)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](requirements.txt)
[![NeuroBench 2.3.0](https://img.shields.io/badge/NeuroBench-2.3.0-6f42c1.svg)](https://github.com/NeuroBench/neurobench)
[![Status: WIP](https://img.shields.io/badge/status-work%20in%20progress-orange.svg)](EXPERIMENT_LOG.md)

[Overview](#overview) •
[Results](#results) •
[Method](#method) •
[Installation](#installation) •
[Usage](#usage) •
[Reproducibility](#reproducibility) •
[Citation](#citation) •
[License](#license)

</div>

---

## Overview

This repository targets the **NeuroBench v1.0 Keyword Few-Shot Class-Incremental Learning (FSCIL)**
task on the Multilingual Spoken Words Corpus (MSWC): 100 base classes, followed by 10 incremental
sessions of 10-way 5-shot learning.

The goal is to beat the published SNN result with a spiking network whose incremental sessions use
**only a local, integer, low-bit three-factor learning rule**. The rule is designed so it can later be
built as a small digital neuromorphic RTL block.

> [!NOTE]
> **Work in progress.** No result in this repository is final yet. Progress and every run are
> recorded in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

### Key features

- **On-device incremental learning without backprop:** new classes are learned by an integer
  Hebbian three-factor rule `ΔA_ci = pre_i · post_c · M`.
- **Bit-exact hardware reference:** [`hw_model/`](hw_model) is an event-driven integer model of the
  rule, checked bit for bit against the training-time implementation.
- **Upstream-compatible backbone:** a CUDA-graph-capturable RadLIF RSNN that loads the NeuroBench
  baseline checkpoints unchanged and matches the upstream forward pass.
- **Clean evaluation protocol:** every design decision is made on a pseudo-incremental protocol
  built from the base splits. The official test split is used for at most 5 logged runs.
- **Full metric reporting:** all NeuroBench harness metrics are reported, including learner state
  beyond the model parameters.

## Results

| Method | Base acc. | Session avg. | Footprint (B) | Act. sparsity | Dense | Eff_ACs | Source |
|---|---|---|---|---|---|---|---|
| M5 ANN (leaderboard) | 97.09% | 89.27% | 6.03E6 | 0.783 | 2.59E7 | 0 (7.85E6 MACs) | `leaderboard.rst` |
| SNN (leaderboard) | 93.48% | 75.27% | 1.36E7 | 0.916 | 3.39E6 | 3.65E5 | `leaderboard.rst` |
| SNN baseline, our rerun (official run 1/5) | 84.21 ± 0.06% ¹ | 76.09 ± 0.36% | 1.36E7 | 0.917 | 6.74E8 ² | 7.12E7 ² | [`results/official_runs.jsonl`](results/official_runs.jsonl) |
| **Ours** | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

¹ Measured with the prototype readout, i.e. the readout used in the incremental sessions.<br>
² neurobench 2.3.0 counts ops per sample over all T = 200 steps. The leaderboard values are ≈ 200×
smaller (6.74E8 / 200 = 3.37E6, 7.12E7 / 200 = 3.56E5), so ops are compared only against our rerun
on the same harness.

**Reporting conventions**

- Every number from this repository is mean ± std over ≥ 3 seeds, measured with neurobench 2.3.0.
- The leaderboard SNN base accuracy (93.48%) uses the backprop-trained readout. This repository
  always reports base accuracy with the readout used in the incremental sessions
  (see [`docs/protocol.md`](docs/protocol.md)).

## Method

```
 audio ──► S2S spike encoder ──► RadLIF RSNN backbone ──► sum-over-time readout (W, b) ──► class
                                  (offline, backprop)       ▲
                                                            │ integer three-factor rule
                                                            │ (on-device, incremental sessions)
```

1. **Backbone.** A RadLIF recurrent SNN with the same architecture as the NeuroBench baseline,
   trained offline on the 100 base classes. The re-implementation in
   [`nbfscil/snn.py`](nbfscil/snn.py) is graph-capturable and matches upstream at the bit level.
   The final training recipe is TODO (Phase 4).
2. **Incremental learner.** Centered, L2-normalized, k-bit integer class prototypes are learned by a
   three-factor Hebbian rule ([`nbfscil/hebbian.py`](nbfscil/hebbian.py)). Consolidation uses only
   integer add, multiply, integer square root and shift.
3. **Deployed readout.** The learned `(W, b)` is written into the SNN's sum-over-time readout, so
   the harness measures exactly the computation that would run on hardware.

A detailed description is in [`report.md`](report.md) (draft).

## Installation

**Requirements**

- Python 3.13
- A CUDA GPU for training (tested on Tesla T4 and NVIDIA L4). Tests that need no data also run on CPU.

**Setup**

```bash
git clone https://github.com/tunadeniz1304/NeuroBench.git
cd NeuroBench
./reproduce.sh env        # pinned dependencies + upstream NeuroBench @ e521c28 in third_party/
```

`reproduce.sh env` installs the pinned versions from [`requirements.txt`](requirements.txt)
(torch 2.11, CUDA 12.8 wheels) and clones the upstream NeuroBench repository at a pinned commit.

## Usage

### Build the data cache

```bash
export NB_DATA=/content/data   # fast local disk; dataset and caches never go into the repository
./reproduce.sh cache           # download the MSWC FSCIL subset and build the S2S / MFCC caches
```

### Run the tests

```bash
./reproduce.sh test            # tests that need the data cache are skipped if it is missing
```

### Train a backbone

```bash
python -m nbfscil.train --config configs/rsnn_baseline_train_amp.yaml --seed 0 --fold 0 \
    --out $NB_DATA/ckpts/rsnn_baseline_amp_f0_s0.pt
```

- Omit `--fold` to train on all 100 base classes.
- An interrupted run resumes from its last finished epoch when started again with the same arguments.

### Evaluate on the pseudo-incremental protocol

```bash
python -m nbfscil.pseudo_eval --ckpt $NB_DATA/ckpts/rsnn_baseline_amp_f0_s0.pt \
    --learner configs/learners/*.yaml --seeds 0 1 2 3 4
```

To run a sequence of training and evaluation jobs that survives runtime resets (e.g. on Colab), use
[`scripts/colab_queue.sh`](scripts/colab_queue.sh):

```bash
bash scripts/colab_queue.sh configs/exp/cos_1024_amp_coslr.yaml:0:0 configs/exp/cos_1024_amp_coslr.yaml:0:1
```

> [!WARNING]
> `nbfscil.official_eval` runs on the official test split. Each invocation uses one of at most
> **5 official runs**, is appended to [`results/official_runs.jsonl`](results/official_runs.jsonl)
> and must only be used for final configurations.

## Project structure

```
.
├── nbfscil/
│   ├── cache.py            # one-time S2S / MFCC encoding of MSWC (same encoder as upstream)
│   ├── sessions.py         # seeded official session sampler and pseudo-incremental protocol
│   ├── snn.py, models.py   # RSNN re-implementation, upstream RSNN wrapper, M5 reference
│   ├── train.py            # offline base training (CUDA-graph captured step, resumable)
│   ├── losses.py, augment.py
│   ├── hebbian.py          # three-factor integer prototype rule
│   ├── readout.py, learners.py
│   ├── pseudo_eval.py      # validation protocol (the only protocol used for decisions)
│   └── official_eval.py    # official harness evaluation (5-run budget)
├── hw_model/               # bit-exact integer reference of the on-chip learning rule
├── configs/                # training, system, learner and experiment configs
├── scripts/                # resumable Colab job queue
├── tests/                  # equivalence, hardware-model, split and resume tests
├── results/                # logged pseudo-protocol and official results
├── docs/                   # task protocol, literature landscape, idea scoring
├── reproduce.sh            # end-to-end reproduction script
├── EXPERIMENT_LOG.md       # chronological log of every experiment
└── report.md               # technical report (draft)
```

## Reproducibility

The full pipeline is driven by [`reproduce.sh`](reproduce.sh):

| Stage | What it does |
|---|---|
| `env` | installs pinned dependencies and upstream NeuroBench @ `e521c28` |
| `cache` | downloads the MSWC FSCIL subset and builds the spike caches |
| `test` | runs the test suite |
| `baseline` | trains the upstream SNN recipe on our RSNN (3 seeds) |
| `pseudo` | trains pseudo-fold backbones and runs the pseudo-incremental protocol |
| `final` | trains the final backbone(s); needs `FINAL_TRAIN_CFG` (TODO) |
| `official` | official harness evaluation; requires `CONFIRM_OFFICIAL=1` |

```bash
./reproduce.sh env cache test baseline pseudo
```

### Evaluation rules

- Design and hyperparameter decisions are made only on the pseudo-incremental protocol built from
  the base train/val splits ([`docs/protocol.md`](docs/protocol.md)).
- The official test split is used for final configurations only: at most 5 runs in total, every run
  logged.
- The task definition (shots, classes, sessions, splits) and the harness metric code are unchanged.
- All harness metrics are reported. Learner state beyond the model parameters is reported separately
  (`extra_footprint_bytes`) and added to the footprint.

## Roadmap

- [x] Reproduce the NeuroBench SNN baseline on the official harness (tag `v0-baseline-repro`)
- [x] Pseudo-incremental validation protocol and integer prototype learners
- [ ] Final backbone training recipe (Phase 4)
- [ ] Official evaluation of the final configuration
- [ ] Technical report and leaderboard submission (tag `v1-final`)

## Citation

If you use this code, please cite this repository and the NeuroBench paper:

```bibtex
@software{deniz2026nbfscil,
  author = {Deniz, Tuna},
  title  = {Hardware-Plausible SNN for NeuroBench Keyword FSCIL},
  year   = {2026},
  url    = {https://github.com/tunadeniz1304/NeuroBench}
}

@article{yik2025neurobench,
  title   = {The neurobench framework for benchmarking neuromorphic computing algorithms and systems},
  author  = {Yik, Jason and others},
  journal = {Nature Communications},
  volume  = {16},
  pages   = {1545},
  year    = {2025}
}
```

## License

This project is released under the [MIT License](LICENSE).

The upstream NeuroBench code is not part of this repository. It is cloned into `third_party/`
(gitignored) and remains under its own license. [`nbfscil/snn.py`](nbfscil/snn.py) re-implements the
equations and state-dict layout of the upstream `sparchSNNs.py` so that upstream checkpoints load unchanged.

## Acknowledgements

- [NeuroBench](https://github.com/NeuroBench/neurobench) for the benchmark harness, the task
  definition and the baseline models.
- The [Multilingual Spoken Words Corpus](https://mlcommons.org/datasets/multilingual-spoken-words/)
  (MLCommons) for the dataset.
