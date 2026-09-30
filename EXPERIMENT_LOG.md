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

## 2026-09-29 15:59 UTC — CPU-only housekeeping session (no GPU, no data)
- Environment: no GPU, no MSWC data, no `$NB_DATA` cache. Nothing trained, no pseudo or official eval run;
  official budget unchanged (1/5 used, result of run 1 not yet in `results/`).
- pytest on CPU (Python 3.11, torch 2.14 CPU, neurobench 2.3.0, upstream @ e521c28 in third_party/):
  22 passed, 3 skipped (need MSWC data / S2S cache). SNN equivalence and HW bit-exactness tests pass.
- Checked `official_sessions` against upstream `IncrementalFewShot`: same sampling distribution
  (random language order, 5 shots from samples 0–99, all 100 query samples 100–199); only the RNG differs.
- Added `reproduce.sh` (stages env/cache/test/baseline/pseudo/final/official; `official` refuses to run
  without CONFIRM_OFFICIAL=1 and a final config) and drafts README.md, report.md, PR_DRAFT.md,
  LEADERBOARD_ROW.md with TODOs for every missing number.
- Noted, not changed: `results/runs.csv` referenced by the resume procedure does not exist yet;
  `extra_footprint_bytes` is reported next to, not inside, the harness Footprint and must be added in the report.

## 2026-09-29 16:02 UTC — OFFICIAL RUN 1/5 result: baseline SNN reproduction (Phase 2 gate passed)
- Config `configs/baseline_snn.yaml` (upstream `mswc_rsnn_proto` checkpoint + upstream prototype readout),
  seeds 0 1 2, run started at commit ba7ad52. The `git` field in `results/official_runs.jsonl` reads b65d03e
  because the hash is taken when the run finishes; commits made during the run were not loaded by it.
- Base accuracy (prototype readout, session 0): 84.21% ± 0.06 (paper: ~84.3% = 93.48 − 9.17).
- Session average: 76.09% ± 0.36 (leaderboard SNN: 75.27%). Difference within what the unseeded upstream
  sampler and the audio backend (torchaudio 2.11 vs 2.0.2 + sox_io) can explain.
- Decision: reproduction accepted, Phase 2 gate passed, tag v0-baseline-repro. This run is the reference only;
  no design decision is tuned on it. Official budget used: 1/5.

## 2026-09-29 16:30 UTC — Official run 1/5: full harness metrics, unit check
- From `results/official_runs.jsonl` (seeds 0 1 2): Footprint 13 550 384 B, ConnectionSparsity 0.0199,
  ActivationSparsity 0.917, Dense 6.742E8, Eff_MACs 0, Eff_ACs 7.121E7 (session 0; later sessions within 0.2%).
  Per-session mean acc: 84.21 81.85 79.91 78.34 76.68 75.62 74.06 73.13 72.05 71.09 70.05; novel-class 59.52%.
- Dense and Eff_ACs are ≈ 200× the leaderboard values (6.742E8/200 = 3.37E6 vs 3.39E6; 7.121E7/200 = 3.56E5
  vs 3.65E5): neurobench 2.3.0 sums ops over all T = 200 steps per sample, the 2024 table is per step.
  Footprint and activation sparsity match. Decision: win condition 2 is checked against this rerun
  (same harness), never against the leaderboard numbers; both conventions are reported in the final table.

## 2026-09-29 17:00 UTC — Phase 3 start: training throughput on the pseudo fold
- 1-epoch smoke, fold 0 (60 classes, 117 steps/epoch), seed 0: fp32 476 s/epoch (4.1 s/step),
  loss 7.41, val 5.1%; amp (`configs/rsnn_baseline_train_amp.yaml`) 253 s/epoch (2.2 s/step), loss 7.63, val 5.0%.
- Full pseudo grid (2 folds x 3 seeds x 50 epochs) would take ~21 h with amp, ~40 h fp32: too long for one
  Colab session. Throughput is also far below what the FLOP count suggests for a T4 (to be profiled).
- Decision: use amp (1.9x faster, same first-epoch behaviour). Train fold 0 seed 0 first (~3.5 h),
  run all learners on it (5 protocol seeds), then extend to more backbone seeds / fold 1.
  Checkpoints go to Drive so a runtime reset does not lose them.

## 2026-09-29 17:40 UTC — Phase 3: moved to an L4 runtime
- Colab gave an NVIDIA L4 (23 GB) instead of the requested A100. Fresh runtime: repo, env and S2S cache rebuilt
  (cache ~3 min on this machine), pytest 25 passed (no skips).
- 1-epoch smoke, fold 0 seed 0, amp: 98.3 s/epoch (T4: 253 s) → ~1.4 h per 50-epoch backbone, ~9 h for the
  2 folds x 3 seeds grid.
- Full training fold 0 seed 0 started (checkpoint on Drive).
- INVALID, discarded: a pseudo_eval of the same checkpoint path ran seconds after training started (notebook
  "run all"), i.e. on an untrained/partial checkpoint; all learners at chance level (3-10%). Not used for anything.

## 2026-09-29 19:10 UTC — Phase 3: first pseudo-protocol result (fold 0, baseline recipe, backbone seed 0)
- Backbone `configs/rsnn_baseline_train_amp.yaml`, fold 0, seed 0, 50 epochs on L4 (98 s/epoch).
  Trained CE readout on pseudo-base val (60 classes): 92.23%.
- Learners, 5 protocol seeds (`results/pseudo_eval.jsonl`), session avg / base / novel, %:
  | learner | session avg | base (s0) | last (s4) | novel |
  |---|---|---|---|---|
  | euclid (upstream rule) | 60.04 ± 0.68 | 69.08 | 53.40 | 32.96 |
  | float CL2N | 61.27 ± 0.67 | 70.02 | 54.94 | 36.81 |
  | Hebbian CL2N 8-bit | 60.68 ± 0.28 | 67.68 | 55.23 | 39.32 |
  | Hebbian CL2N 4-bit | 60.47 ± 0.44 | 67.47 | 55.04 | 39.24 |
  | centre only, 8-bit | 36.29 ± 0.36 | 41.88 | 31.85 | 27.11 |
  | L2 only, 8-bit | 3.06 ± 0.43 | 3.48 | 2.62 | 2.60 |
- Reading: (1) The dominant loss is the readout swap: trained readout 92.2% vs prototype base 69-70%
  (23 pts; official split: 93.5 → 84.2). (2) CL2N vs euclid: +1.2 float, +0.6 8-bit, +0.4 4-bit session avg —
  within/near the ±0.5 noise band with one backbone seed; the clearer effect is on novel classes (+3.9 float,
  +6.4 8-bit) at a 1.4-pt base cost. (3) 8/4-bit integer learning costs ≤0.8 pts vs float CL2N.
  (4) Centring is required for low-bit prototypes: uncentred 8-bit L2 collapses to chance. Reproduced on synthetic
  sparse heavy-tailed counts (float uncentred cosine 100%, uncentred 8-bit 20%): a few always-on neurons dominate
  the unit vector, clip at w_max and the class-specific part rounds away. Not a code bug; kept as an ablation.
  Centre-only (no norm term) is not NCM and loses 24 pts, so the normalisation is also needed.
- Decision: the representation/readout mismatch is the main lever → run idea 3 next (cosine-classifier
  pretraining, `configs/exp/cos_1024_amp.yaml`, fold 0 seed 0), then baseline seeds 1-2 for the noise level.
  Learner choice (CL2N 8-bit) is provisional until ≥3 backbone seeds.

## 2026-09-29 20:40 UTC — Phase 4, idea 3: cosine-classifier pretraining (fold 0, seed 0) — large gain
- Backbone `configs/exp/cos_1024_amp.yaml` (centred cosine classifier, scale 16, zero initial state), fold 0,
  seed 0, 50 epochs; cosine-readout val 91.73% (CE baseline: 92.23%).
- Learners, 5 protocol seeds, session avg / base / novel, % (Δ vs baseline backbone, same learner):
  | learner | session avg | base | novel |
  |---|---|---|---|
  | euclid | 83.07 ± 0.35 (+23.0) | 88.03 | 69.08 |
  | float CL2N | 82.61 ± 0.39 (+21.3) | 88.05 | 66.03 |
  | Hebbian CL2N 8-bit | 81.90 ± 0.46 (+21.2) | 87.17 | 67.07 |
  | Hebbian CL2N 4-bit | 81.87 ± 0.40 (+21.4) | 87.10 | 67.21 |
  | centre only, 8-bit | 75.66 ± 0.31 | 83.83 | 49.31 |
  | L2 only, 8-bit | 12.44 ± 1.27 | 15.07 | 7.67 |
- Reading: the readout-swap loss shrinks from 23 pts (92.2 → 69.1) to ~4 pts (91.7 → 88.0); novel accuracy
  doubles (33 → 67-69). Far outside the ±0.5 noise band even with one backbone seed.
  On this backbone float euclid is ahead of CL2N (+0.5 float CL2N, +1.2 8-bit); with one backbone seed this
  ordering is not yet established. 8-bit integer learning costs 0.7 pts vs float CL2N; 4-bit = 8-bit.
- Decision: idea 3 adopted as the backbone recipe. Queue reordered: cosine seeds 1-2 first (confirm the gain and
  the learner ordering), then baseline seeds 1-2. If euclid stays ahead, add an integer three-factor variant of
  the Euclidean prototype rule (w = mean count, b = -|w|²/2) with a bit-exact HW model.

## 2026-09-30 02:00 UTC — Phase 4: cosine pretraining over 3 seeds — gain holds, but training is unstable
- Runtime reset during baseline seed 2 (epoch ~15); cos seeds 1-2 and baseline seed 1 finished before it.
  Their pseudo evals were recomputed from the Drive checkpoints (deterministic).
- Session avg (fold 0, 5 protocol seeds each), %:
  | learner | cos s0 | cos s1 | cos s2 | cos mean ± std | baseline s0, s1 | baseline mean |
  |---|---|---|---|---|---|---|
  | euclid | 83.07 | 70.78 | 80.14 | 78.00 ± 5.24 | 60.04, 61.86 | 60.95 |
  | float CL2N | 82.61 | 70.57 | 79.32 | 77.50 ± 5.08 | 61.27, 62.16 | 61.72 |
  | Hebbian CL2N 8-bit | 81.90 | 68.70 | 78.94 | 76.51 ± 5.66 | 60.68, 63.20 | 61.94 |
  | Hebbian CL2N 4-bit | 81.87 | 68.59 | 78.81 | 76.42 ± 5.68 | 60.47, 63.12 | 61.80 |
  Trained cosine-readout val: s0 91.73, s1 85.62, s2 90.77 (CE baseline: 92.23, 92.08).
- Reading: (1) Cosine pretraining gain is real (+14.6 to +17.1 on the mean; even the worst cosine seed beats
  every baseline seed by ≥5.5 pts). (2) But seed-to-seed std is ~5 pts vs ~1 pt for CE: the recipe is unstable;
  seed 1 also has a weak trained readout (85.6%), so the backbone itself trained worse, not the learner.
  Suspects: fp16 autocast with silently zeroed non-finite gradients (train.amp), no LR warmup at lr 1e-3,
  EMA centre updated inside the loss. (3) Learner ordering is consistent across all 3 cosine seeds:
  euclid > float CL2N > Hebbian 8-bit (euclid − 8-bit: 1.17 / 2.08 / 1.20). On the CE backbone the order flips
  (8-bit ≥ euclid). 4-bit = 8-bit within 0.1.
- Decisions: (a) stabilise cosine training before anything else: inspect per-epoch histories of the three
  cosine checkpoints, then test fp32 (no amp) and LR warmup on the pseudo protocol; (b) add an integer
  three-factor Euclidean prototype rule (with bit-exact HW model) since float euclid is consistently ahead of
  CL2N on cosine backbones; (c) add mid-run resume to training so a runtime reset costs ≤1 epoch.

## 2026-09-30 03:00 UTC — Diagnosis of cosine-training instability
- Per-epoch histories (fold 0): no loss spikes or NaN-like jumps → fp16 is not the primary suspect.
  Cosine-loss runs converge much slower than CE (val@5: 0.58 / 0.36 / 0.44 vs CE 0.69). At the step LR drop
  (epoch 20) the seeds sit at 0.89 / 0.83 / 0.88 and barely move afterwards (final 0.917 / 0.856 / 0.908):
  the spread is fixed by where each run is when the LR is cut.
- Next: `configs/exp/cos_1024_amp_coslr.yaml` — same recipe, cosine LR schedule + 2 warm-up epochs (one change).
  Run the bad seed (1) first, then 0 and 2. Accept if seed std drops to ~1-2 pts without lowering the mean.

## 2026-09-30 16:20 UTC — Phase 4: cosine LR schedule (coslr) rejected; loss spikes are the instability
- `configs/exp/cos_1024_amp_coslr.yaml` (cosine LR + 2 warm-up epochs; only change vs `cos_1024_amp`), fold 0,
  seeds 0-2, 5 protocol seeds each (`results/pseudo_eval.jsonl`). Session avg, %:
  | learner | coslr s0 | coslr s1 | coslr s2 | coslr mean ± std | step (cos_1024_amp) mean ± std |
  |---|---|---|---|---|---|
  | euclid | 53.12 | 77.28 | 77.22 | 69.21 ± 13.93 | 78.00 ± 5.24 |
  | float CL2N | 53.38 | 76.79 | 76.81 | 68.99 ± 13.52 | 77.50 ± 5.08 |
  | Hebbian CL2N 8-bit | 51.09 | 75.25 | 75.40 | 67.25 ± 13.99 | 76.51 ± 5.66 |
  | Hebbian CL2N 4-bit | 51.04 | 75.23 | 75.36 | 67.21 ± 14.00 | 76.42 ± 5.68 |
  | centre only, 8-bit | 32.04 | 66.30 | 67.49 | 55.28 ± 20.14 | — |
  | L2 only, 8-bit | 24.72 | 57.44 | 11.80 | 31.32 ± 23.52 | — |
  Trained cosine-readout val: coslr 76.05 / 88.95 / 88.58 (step: 91.73 / 85.62 / 90.77).
- Reading: (1) coslr is worse on the mean (−8.8 euclid) and the spread grows (std 13.9 vs 5.2). The bad seed moved
  (step: s1; coslr: s0) instead of disappearing. Even the two good coslr seeds (77.3, 77.2) stay below the two
  good step seeds (83.1, 80.1), with lower readout val. (2) Correction of the 03:00 diagnosis ("no loss spikes"):
  the coslr histories do show them. s0: train loss 1.50 → 1.84 → 2.83 at epochs 8-10 (lr ≈ 9.4e-4), val back to
  0.46; it never recovered (final train loss 0.96 vs 0.43-0.48 for s1/s2). s1 has smaller jumps at epochs 10, 21,
  26 that it recovers from; s2 one small jump at 21. The seed outcome is set by how large the early high-lr spike
  is, and the cosine schedule keeps lr ≥ 5e-4 until epoch ~26, i.e. longer exposure than the step drop at 20.
  None of the configs clips gradients. (3) Learner ordering unchanged on all 3 seeds: euclid ≈ float CL2N > 8-bit
  (1.8-2.0 pts); 4-bit = 8-bit. (4) Open anomaly: in all three runs the train loss rises over the last 3-6 epochs
  while lr < 1e-5 (s0 0.955 → 1.030, s1 0.432 → 0.459, s2 0.464 → 0.482) and val drops 0.5-0.6 pts from epoch 45
  to 50. With lr that small the parameters should barely move; cause unknown (to check with the new gradient logs).
- Decisions: (a) coslr rejected; `cos_1024_amp` (step) stays the backbone recipe. (b) Next: the same step recipe
  with relative spike clipping (`configs/exp/cos_1024_amp_clip.yaml`, `train.grad_clip_rel: 2.0`, one change),
  fold 0 seeds 0-2. The earlier `cos_1024_amp_coslr_clip.yaml` (clip on top of coslr) was never run and is
  replaced by it. Per-epoch `gnorm_mean/max`, `nonfinite_steps`, `clipped_steps` are now logged, which separates
  gradient explosion from fp16 overflow. Accept if seed std drops to ~1-2 pts without lowering the mean (78.0).
- Baseline backbone seed 2 (`rsnn_baseline_train_amp`) restarted from epoch 1 at 16:13 (its interrupted run
  predates the resume feature); results pending.
