# NeuroBench Keyword FSCIL — hardware-plausible SNN

Goal: beat the published SNN result on the NeuroBench v1.0 Keyword Few-Shot Class-Incremental
Learning task (MSWC) with an SNN whose incremental sessions use a local, integer, low-bit
three-factor learning rule that can later be built as a small digital neuromorphic RTL block.

**Status: work in progress.** No result of this repo is final yet; see `EXPERIMENT_LOG.md`.

## Results

| Method | Base acc. | Session avg. | Footprint (B) | Act. sparsity | Dense | Eff_ACs | Source |
|---|---|---|---|---|---|---|---|
| M5 ANN (leaderboard) | 97.09% | 89.27% | 6.03E6 | 0.783 | 2.59E7 | 0 (7.85E6 MACs) | `leaderboard.rst` |
| SNN (leaderboard) | 93.48% | 75.27% | 1.36E7 | 0.916 | 3.39E6 | 3.65E5 | `leaderboard.rst` |
| SNN baseline, our rerun (official run 1/5) | TODO | TODO | TODO | TODO | TODO | TODO | `results/official_runs.jsonl` |
| **Ours** | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

All numbers of this repo are mean ± std over ≥ 3 seeds, measured with neurobench 2.3.0.
The leaderboard SNN base accuracy (93.48%) uses the backprop-trained readout; this repo always reports
base accuracy with the same readout that is used in the incremental sessions (see `docs/protocol.md`).

## Method (short)

1. **Backbone:** RadLIF recurrent SNN (same architecture as the NeuroBench baseline, graph-capturable
   re-implementation in `nbfscil/snn.py`, bit-level equal to upstream), trained offline on the 100 base classes.
   Final training recipe: TODO (Phase 4).
2. **Incremental learner:** centered, L2-normalized, k-bit integer class prototypes learned by a
   three-factor Hebbian rule `ΔA_ci = pre_i · post_c · M` (`nbfscil/hebbian.py`). Consolidation uses only
   integer add, multiply, integer square root and shift. `hw_model/prototype_rule.py` is an event-driven
   integer reference model; `tests/test_hw_model.py` checks it bit for bit against the training-time rule.
3. **Readout inside the SNN:** the learned `(W, b)` is written into the sum-over-time readout, so the
   harness measures exactly the deployed computation.

## Repository layout

| Path | Content |
|---|---|
| `nbfscil/cache.py` | one-time S2S / MFCC encoding of MSWC into `$NB_DATA/cache` (same encoder as upstream) |
| `nbfscil/sessions.py` | seeded official session sampler and the pseudo-incremental protocol |
| `nbfscil/snn.py`, `nbfscil/models.py` | RSNN re-implementation, upstream RSNN wrapper, M5 reference |
| `nbfscil/train.py`, `losses.py`, `augment.py` | offline base training (CUDA-graph captured step) |
| `nbfscil/hebbian.py`, `readout.py` | three-factor integer prototype rule and readout learners |
| `nbfscil/pseudo_eval.py` | Phase 3 validation protocol (only protocol used for decisions) |
| `nbfscil/official_eval.py` | official harness evaluation (budget: 5 runs, logged in `results/official_runs.jsonl`) |
| `hw_model/` | bit-exact integer reference of the on-chip learning rule |
| `configs/` | training, system and learner configs |
| `docs/` | task protocol, literature landscape, idea scoring |

## Reproduce

```bash
export NB_DATA=/content/data          # fast local disk; dataset and caches never go into the repo
./reproduce.sh env cache test         # dependencies, upstream NeuroBench @ e521c28, S2S cache, tests
./reproduce.sh baseline               # upstream SNN recipe on our RSNN (3 seeds)
./reproduce.sh pseudo                 # pseudo-incremental protocol (design decisions)
./reproduce.sh final                  # final backbone(s)  — needs FINAL_TRAIN_CFG (TODO)
CONFIRM_OFFICIAL=1 ./reproduce.sh official   # official harness eval — uses the 5-run budget
```

Tested on 1× Tesla T4, Python 3.13, torch 2.11 (CUDA 12.8). Tests that need no data also run on CPU.

## Evaluation rules we follow

- Design and hyperparameter decisions only on the pseudo-incremental protocol built from the base
  train/val splits (`docs/protocol.md`). The official test/evaluation split is used for final configs only,
  at most 5 runs in total, every run logged.
- Task definition (shots, classes, sessions, splits) and harness metric code are unchanged.
- All harness metrics are reported; learner state beyond model parameters is reported separately
  (`extra_footprint_bytes`) and added to the footprint.
