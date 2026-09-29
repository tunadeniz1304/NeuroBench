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
