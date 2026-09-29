# Experiment log

## 2026-09-29 12:54 UTC — Phase 0: hardware
- GPU: 1× Tesla T4, 15 GB VRAM (driver 580.82, CUDA 13.0). RAM 12 GB, no swap. Disk ~65 GB free. 2 CPU cores.
- Python 3.13.15, torch 2.11.0+cu128, torchaudio 2.11.0+cu128.
- Decision: small GPU + 2 CPU cores → dataloading (opus decode) will be the bottleneck.
  Plan to pre-decode/cache encoded features (S2S spikes / MFCC) of the base set to local disk once,
  and keep models ≤ baseline size. Budget in Phase 4 scaled accordingly.

## 2026-09-29 13:00 UTC — Phase 0: repo + resources
- `origin` = https://github.com/tunadeniz1304/NeuroBench.git (public, empty) cloned to `/content/NeuroBench`.
- BLOCKER: no git identity and no push auth on this machine. Per rules, not worked around;
  asked the user to configure them. No commits made yet.
- Upstream NeuroBench cloned to `third_party/neurobench` (commit e521c28, 2026-05-12). Latest PyPI: neurobench 2.3.0.
- Dataset: `mswc_fscil.tar.gz`, 651 MB, downloading to `/content/data` (local disk, not Drive/repo).
- Leaderboard numbers verified in `leaderboard.rst`: M5 ANN 97.09/89.27, SNN 93.48/75.27.

## 2026-09-29 13:20 UTC — Phase 0: environment
- Push auth fixed by user (fine-grained PAT); first push of scaffolding OK.
- Installed neurobench 2.3.0, snntorch 1.0.0, tonic 1.6.0 (pulled numpy down to 1.26.4). Pinned in requirements.txt.
- torchaudio 2.11 decodes opus at ~17 ms/sample (2 cores) → built an int8 S2S cache (`python -m nbfscil.cache`)
  with the exact upstream encoder config (48 kHz, hop 240, threshold 1). Upstream warns SNN is sensitive to
  the audio backend (trained with torchaudio 2.0.2 + sox_io) → expect some reproduction gap.
- Upstream RSNN training step: ~3.3 s/batch (Python time loop, CPU-bound launches) → ~9 h for the 50-epoch recipe.
  Not acceptable; will reimplement RadLIF graph-capturable (same math) and use CUDA graphs.

## 2026-09-29 13:27 UTC — Phase 1: reconnaissance (gate passed)
- `docs/protocol.md` and `docs/landscape.md` written.
- Key finding: leaderboard SNN base 93.48% uses the trained readout; session average uses prototypes
  (paper: prototype session 0 ≈ 84.3%). Most of the SNN deficit is the representation's poor fit to a
  prototype readout, not forgetting. New-class accuracy: ANN 79.6%, SNN 57.2%.
- No stronger FSCIL entry on the leaderboard / neurobench.ai / literature. Target: session avg ≥ 80% (min),
  ≥ 85% (stretch), ≥ 89.3% (ANN parity).
- Pseudo protocol decision: base has 5 languages × 20 classes; official sessions bring new languages,
  so pseudo-novel = 2 held-out base languages (fold 0: ca+de, fold 1: fr+rw), 4 sessions × 10-way 5-shot.

## 2026-09-29 13:34 UTC — Phase 0 gate: GSC example (passed, with one noted metric drift)
- `examples/gsc/benchmark_ann.py`: Footprint 109228, ConnSparsity 0.0, Acc 0.865334, ActSparsity 0.385446,
  Eff_MACs 1728072, Dense 1880256 → matches README to ≥5 significant digits.
- `examples/gsc/benchmark_snn.py`: Footprint 583900, Acc 0.856338, Eff_ACs 3289835, Dense 29030400 → match.
  ActivationSparsity 0.9072 vs README 0.9669. The README numbers predate the v2 library refactor
  (upstream commits cfb182c "refactor entire library", d895ea8 activation-sparsity refactor) and snntorch 1.0.
  Accuracy and synaptic ops match exactly, so the model computation is identical; the drift is in the metric
  implementation of the installed harness. Decision: gate passed; always re-measure baselines with the same
  harness version (neurobench 2.3.0) and compare like with like; flag this in the final report.
- S2S cache built: base_train 50000, base_val 10000, base_test 10000, evaluation 10 langs × 2000; T = 200 steps
  (S2S drops the first of 201 frames, same as upstream).
- `nbfscil/snn.py`: graph-capturable RadLIF RSNN with identical math/state_dict; `tests/test_snn_equivalence.py`
  confirms bit-level agreement with the upstream module (incl. RNG stream of the random initial state).

## 2026-09-29 13:51 UTC — Official eval budget definition
- One "official run" = one invocation of `nbfscil.official_eval` (base test + evaluation languages),
  covering one config over its seeds. Budget: 5. Every run is appended to `results/official_runs.jsonl`
  and listed here, including bad ones.
- OFFICIAL RUN 1/5 (Phase 2 reproduction): `configs/baseline_snn.yaml` (upstream checkpoint + prototype
  readout), seeds 0 1 2, commit ba7ad52. No decisions are tuned on it; it is the reference.
