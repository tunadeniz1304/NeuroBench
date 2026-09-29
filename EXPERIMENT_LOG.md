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

## 2026-09-29 13:40 UTC — Phase 0: environment
- Push auth fixed by user (fine-grained PAT); first push of scaffolding OK.
- Installed neurobench 2.3.0, snntorch 1.0.0, tonic 1.6.0 (pulled numpy down to 1.26.4). Pinned in requirements.txt.
- torchaudio 2.11 decodes opus at ~17 ms/sample (2 cores) → built an int8 S2S cache (`python -m nbfscil.cache`)
  with the exact upstream encoder config (48 kHz, hop 240, threshold 1). Upstream warns SNN is sensitive to
  the audio backend (trained with torchaudio 2.0.2 + sox_io) → expect some reproduction gap.
- Upstream RSNN training step: ~3.3 s/batch (Python time loop, CPU-bound launches) → ~9 h for the 50-epoch recipe.
  Not acceptable; will reimplement RadLIF graph-capturable (same math) and use CUDA graphs.

## 2026-09-29 13:45 UTC — Phase 1: reconnaissance (gate passed)
- `docs/protocol.md` and `docs/landscape.md` written.
- Key finding: leaderboard SNN base 93.48% uses the trained readout; session average uses prototypes
  (paper: prototype session 0 ≈ 84.3%). Most of the SNN deficit is the representation's poor fit to a
  prototype readout, not forgetting. New-class accuracy: ANN 79.6%, SNN 57.2%.
- No stronger FSCIL entry on the leaderboard / neurobench.ai / literature. Target: session avg ≥ 80% (min),
  ≥ 85% (stretch), ≥ 89.3% (ANN parity).
- Pseudo protocol decision: base has 5 languages × 20 classes; official sessions bring new languages,
  so pseudo-novel = 2 held-out base languages (fold 0: ca+de, fold 1: fr+rw), 4 sessions × 10-way 5-shot.
