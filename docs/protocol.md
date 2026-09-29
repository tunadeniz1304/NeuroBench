# NeuroBench Keyword FSCIL protocol (MSWC)

Source of truth: `third_party/neurobench/examples/mswc_fscil/mswc_fscil.py`,
`neurobench/datasets/MSWC_dataset.py`, `neurobench/datasets/MSWC_IncrementalLoader.py`
(upstream commit `e521c28`, 2026-05-12) and `leaderboard.rst`.

## Data
- Subset of MSWC hosted at `huggingface.co/datasets/NeuroBench/mswc_fscil_subset`
  (`mswc_fscil.tar.gz`, 651 MB). 48 kHz opus audio, 200 keyword classes.
- **Base classes (0–99)**, English-dominant: 500 train / 100 validation / 100 test samples per class
  (`MSWC(subset="base", procedure="training"|"validation"|"testing")`).
- **Evaluation (incremental) classes (100–199)**: 10 languages
  (fa, eo, pt, eu, pl, cy, nl, ru, es, it), 10 classes each, 200 samples per class,
  split into disjoint support pool (first 100) and query pool (last 100)
  via `support_query_split=(100, 100)`.

## Sessions
- **Session 0:** model trained offline on base train set (backprop allowed).
  Evaluated on base test set (100 classes × 100 = 10 000 samples), output masked to classes 0–99.
- **Sessions 1–10:** one language per session, order randomly permuted each repeat.
  - Support: 10-way × **5-shot** drawn from the support pool (50 samples).
  - Query: 100 samples per class drawn from the query pool, accumulated over sessions.
  - Test set for session *s* = base test set ∪ cumulative query set
    (10 000 + 1 000·s samples); output masked to all classes seen so far.
- **Metrics:** Base accuracy = session 0 accuracy. Session average = mean of the
  11 session accuracies (0–10). Also reported: Footprint, ConnectionSparsity,
  ActivationSparsity, SynapticOperations (Dense, Effective_MACs, Effective_ACs).
- Note: in the reference script `syn_ops_acs` for sessions ≥1 is copied from session 0
  (`pre_train_results`), i.e. leaderboard Eff_ACs is effectively the session-0 value.
  We report the harness output as-is and flag this.

## What the reference methods do in incremental sessions
- Readout is replaced with a **prototype (NCM-equivalent) linear layer**:
  `w_c = 2·μ_c`, `b_c = −‖μ_c‖²` (SNN: μ summed over time and divided by T in bias),
  where μ_c is the mean hidden feature of the class. Base prototypes are computed
  from the base train set; novel prototypes from the 5 support shots.
- Backbone is frozen; no gradient updates in sessions 1–10.

## Reference models
- SNN: 2× RadLIF recurrent layers (1024, 1024) + readout, batchnorm, S2S spike encoding
  (48 kHz, hop 240 → 201 timesteps × 20 channels). Checkpoint `mswc_rsnn_proto`.
  Upstream note: trained/evaluated with `torchaudio==2.0.2` + `sox_io` backend; the SNN is
  sensitive to the audio decoding backend → possible reproduction gap.
- M5 ANN: 1-D CNN on MFCC (20 coeffs). Checkpoint `mswc_cnn_proto`.

## Leaderboard (verified in `leaderboard.rst`, 2026-09-29)
| Method | Base / Session avg | Footprint | Exec. rate | Conn. sparsity | Act. sparsity | Dense | Eff_MACs | Eff_ACs |
|---|---|---|---|---|---|---|---|---|
| M5 ANN | 97.09% / 89.27% | 6.03E6 | 1 | 0.0 | 0.783 | 2.59E7 | 7.85E6 | 0 |
| SNN | 93.48% / 75.27% | 1.36E7 | 200 | 0.0 | 0.916 | 3.39E6 | 0 | 3.65E5 |

## Rules we impose on ourselves
- Official eval (base test + evaluation classes) only for final configs, ≤5 runs total.
- Design decisions use the pseudo-incremental protocol built from base train/val only (Phase 3).
- No replay of raw audio; any stored statistics count towards footprint.

## Important: which readout the leaderboard numbers use
Per the NeuroBench paper (Nat. Commun. 16:1545, 2025; see `docs/landscape.md`), the leaderboard
*base accuracy* (SNN 93.48%) was measured with the backprop-trained readout, while the
*session average* uses the prototype readout from session 0 on. The upstream script
(`mswc_fscil.py`) evaluates session 0 **after** the prototype conversion, so a faithful rerun
gives the prototype session-0 accuracy (paper: ~84.3% for the SNN, i.e. −9.17 pts).
We always report base accuracy with the same readout that is used in the incremental sessions.

## Target (Phase 1 gate)
No stronger entry than the 2024 baselines exists (leaderboard all branches, neurobench.ai, literature, 2026-09).
- Minimum: SNN session average **≥ 80%** (+5 pts over the SNN SOTA 75.27%).
- Stretch: **≥ 85%**; ANN parity: **≥ 89.3%**.
- Base accuracy measured with the incremental readout; Eff_ACs and footprint ≤ SNN baseline (3.65E5, 1.36E7 B).

## Measured with neurobench 2.3.0 (official run 1/5, baseline SNN)
- Dense 6.74E8, Eff_ACs 7.12E7 per sample (sum over T = 200 steps) vs leaderboard 3.39E6 / 3.65E5:
  ratio ≈ 200 = T. Footprint (1.36E7) and activation sparsity (0.917) match the leaderboard.
- Connection sparsity 0.0199 (leaderboard 0.0).
- Consequence: win condition 2 (ops, footprint) is checked against this rerun, same harness version.
