# Integer three-factor prototype learning for spiking keyword FSCIL on NeuroBench MSWC

> Draft skeleton. Every `TODO` needs a number or text from a logged run (`results/`, `EXPERIMENT_LOG.md`).
> No number goes in here without a source file.

## Abstract

TODO (write last). Setup: NeuroBench v1.0 Keyword FSCIL (MSWC), 100 base classes + 10 sessions of
10-way 5-shot. Previous best SNN: 93.48% base / 75.27% session average. Our result: TODO ± TODO
session average (mean ± std, N = TODO seeds), with incremental learning done only by a local integer
three-factor rule.

## 1. Introduction

- Problem: on-device few-shot keyword learning with an SNN, no backprop after deployment.
- Gap: the SNN baseline loses 9.17 pts at session 0 when its trained readout is replaced by
  prototypes (NeuroBench paper, Nat. Commun. 16:1545, 2025); new-class accuracy 57.23% vs 79.61% (ANN).
- Contributions (to confirm against results):
  1. Centered, L2-normalized integer prototypes learned by a three-factor Hebbian rule, with a
     bit-exact hardware reference model.
  2. Base training aligned with the deployment readout: TODO (depends on Phase 4 outcome).
  3. Full NeuroBench metric report and Pareto comparison with both baselines.

## 2. Task and protocol

- Official task: see `docs/protocol.md` (unchanged task definition, unchanged harness metrics,
  neurobench 2.3.0, upstream commit e521c28).
- Inputs come from a one-time S2S cache produced by the upstream encoder (48 kHz, hop 240, threshold 1).
- Session sampler: same sampling distribution as upstream `IncrementalFewShot`
  (random language order, 5 random shots from samples 0–99, queries = samples 100–199), with an
  explicit seed per repeat.
- Validation protocol used for every design decision: pseudo-incremental sessions on held-out base
  languages (fold 0: ca+de, fold 1: fr+rw; 60 pseudo-base classes, 4 sessions × 10-way 5-shot).
- Base accuracy is always measured with the readout used in the incremental sessions.

## 3. Method

### 3.1 Backbone
RadLIF RSNN, layer sizes TODO, S2S input, T = 200 steps. Training recipe: TODO (loss, augmentation,
epochs, seeds). Graph-capturable re-implementation, bit-level equal to upstream (`tests/test_snn_equivalence.py`).

### 3.2 Three-factor integer prototype rule
- Learning: `ΔA_ci(t) = pre_i(t) · post_c(t) · M(t)`, integer accumulators (11 bits suffice for a 5-shot session).
- Consolidation per class: `D = A·2^f − n·μ_q`, `N = isqrt(ΣD²)`, `w = clip(⌊(D·G + r)/N⌋)` with
  xorshift32 stochastic rounding, `b = −⌊μ_q·w / 2^f⌋`.
- Inference: `score_c = Σ_i S_i w_ci + b_c` as event-driven ACs in the sum-over-time readout.
- Weight bits: TODO. Stored state beyond weights: `μ_q` (dim × 16 bit) + one accumulator bank.

### 3.3 Hardware plausibility
- Operations used after deployment: TODO table (add, compare, multiply at consolidation only, isqrt,
  shift, xorshift32). No gradients, no raw-data replay.
- Bit-exact reference: `hw_model/prototype_rule.py`, verified by `tests/test_hw_model.py`.
- Cost per incremental update (ACs / memory writes): TODO.

## 4. Experiments

### 4.1 Baseline reproduction (Phase 2)
| Config | Base (proto readout) | Session avg | Notes |
|---|---|---|---|
| Upstream SNN checkpoint + upstream prototype readout, official run 1/5 | TODO | TODO | `configs/baseline_snn.yaml`, seeds 0 1 2 |
| RSNN re-trained with upstream recipe (pseudo protocol) | TODO | TODO | `configs/rsnn_baseline_train.yaml` |

Known deviation: ActivationSparsity on the GSC example is 0.9072 vs 0.9669 in the upstream README
(metric refactor in neurobench v2); accuracy and synaptic ops match exactly. TODO: state the effect on MSWC.

### 4.2 Pseudo-protocol ablations (design decisions)
| Learner | Fold 0 session avg | Fold 1 session avg | Novel acc | Last session |
|---|---|---|---|---|
| Euclid prototypes (baseline) | TODO | TODO | TODO | TODO |
| Float CL2N | TODO | TODO | TODO | TODO |
| Hebbian CL2N 8-bit | TODO | TODO | TODO | TODO |
| Hebbian CL2N 4-bit | TODO | TODO | TODO | TODO |
| Centering only, 8-bit | TODO | TODO | TODO | TODO |
| L2 only, 8-bit | TODO | TODO | TODO | TODO |

Backbone ablations (loss, augmentation, size): TODO.

### 4.3 Official evaluation (final configs only)
| Run | Config | Commit | Seeds | Base | Session avg |
|---|---|---|---|---|---|
| 1/5 | `configs/baseline_snn.yaml` | ba7ad52 | 0 1 2 | TODO | TODO |
| 2/5 | TODO | TODO | TODO | TODO | TODO |

All runs are listed, including bad ones.

### 4.4 Full NeuroBench metrics
| Method | Base | Session avg | Footprint (B) | Extra learner state (B) | Exec. rate | Conn. sparsity | Act. sparsity | Dense | Eff_MACs | Eff_ACs |
|---|---|---|---|---|---|---|---|---|---|---|
| M5 ANN (leaderboard) | 97.09% | 89.27% | 6.03E6 | – | 1 | 0.0 | 0.783 | 2.59E7 | 7.85E6 | 0 |
| SNN (leaderboard) | 93.48% | 75.27% | 1.36E7 | – | 200 | 0.0 | 0.916 | 3.39E6 | 0 | 3.65E5 |
| Ours | TODO | TODO | TODO | TODO | 200 | TODO | TODO | TODO | TODO | TODO |

Note: the upstream script copies session-0 synaptic ops to later sessions; we report the harness
output per session as-is (TODO: per-session table in appendix).

### 4.5 Per-session accuracy
TODO: figure with session 0–10 accuracy (ours, SNN baseline rerun), and novel-class-only accuracy.

### 4.6 Accuracy vs. cost (Pareto)
TODO: session avg vs Eff_ACs (or Eff_MACs) and vs footprint, both baselines and ours.

## 5. Discussion and limitations
- Audio backend: upstream SNN was trained with torchaudio 2.0.2 + sox_io; we use torchaudio 2.11. TODO: measured gap.
- Session sampler draws differ from upstream (upstream is unseeded); distribution is the same.
- Backprop is used for offline base training only.
- TODO: failure cases / languages with lowest accuracy.

## 6. Reproducibility
`./reproduce.sh` (stages env, cache, test, baseline, pseudo, final, official). Hardware: 1× Tesla T4,
2 CPU cores. Pinned versions in `requirements.txt`. Config and commit of every official run in
`results/official_runs.jsonl`.

## References
See `docs/landscape.md` (NeuroBench, Nat. Commun. 2025; SimpleShot; C-FSCIL; TEEN; Chameleon; CLP-SNN; …).
