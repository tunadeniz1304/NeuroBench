# PR draft for NeuroBench (upstream) — not opened

Draft text for a leaderboard submission PR to `NeuroBench/neurobench`. The repo owner reviews and
submits it. Every number comes from `results/official_runs.jsonl` (official run 2/5).

---

**Title:** Add Keyword FSCIL leaderboard entry: SNN-CL2N (SNN with integer three-factor prototype learning)

## Summary

Adds one row to the Keyword FSCIL (MSWC) leaderboard. The model is a RadLIF recurrent SNN with the same
input encoding as the existing SNN baseline (Speech2Spikes, 48 kHz, hop 240). In the incremental sessions
it learns new classes with a local, integer three-factor Hebbian rule (centered, L2-normalized 8-bit
prototypes); there is no backprop after the base session. The backbone is pretrained offline with a centred
cosine classifier and relative gradient clipping, so that its spike counts suit a prototype readout.

| Method | Base / Session avg | Footprint | Exec. rate | Conn. sparsity | Act. sparsity | Dense | Eff_MACs | Eff_ACs |
|---|---|---|---|---|---|---|---|---|
| SNN (existing) | 93.48% / 75.27% | 1.36E7 | 200 | 0.0 | 0.916 | 3.39E6 | 0 | 3.65E5 |
| This entry | 93.48 ± 0.26% / 86.93 ± 0.40% | 1.36E7 | 200 | 0.049 | 0.900 | 3.37E6 ¹ | 0 | 3.39E5 ¹ |

Mean ± std (population std) over 3 seeds; seed s uses backbone seed s and its own session order and support
shots. Per seed: session average 86.56 / 86.75 / 87.49%.

¹ neurobench 2.3.0 reports Dense / Eff_ACs summed over the T = 200 steps of a sample (ours: 6.74E8 / 6.77E7).
The existing rows match a per-step count, so the values above are divided by 200. On the same harness the
existing SNN gives 6.74E8 / 7.12E7, i.e. 3.37E6 / 3.56E5 per step, slightly different from the 3.39E6 / 3.65E5
in the table; we would like the maintainers to confirm which convention the table should use.

## Evaluation details
- neurobench 2.3.0, unmodified `Benchmark`, metrics and task definition (100 base classes,
  10 sessions × 10-way 5-shot, support/query split (100, 100)).
- Base accuracy is measured with the prototype readout that is also used in sessions 1–10
  (the existing SNN row reports base accuracy with the trained readout).
- Inputs were S2S-encoded once with the upstream `S2SPreProcessor` configuration and cached; this is
  equivalent to encoding inline.
- Audio backend: torchaudio 2.11.0 (upstream baseline used 2.0.2 + sox_io). Our rerun of the existing SNN on
  this setup gives 84.21% base (prototype readout) / 76.09% session average (leaderboard: 75.27%).
- Learner state beyond model parameters: 5,124 B (base centre 1024 × 16 bit, one 1024 × 24-bit accumulator
  bank, PRNG state), listed next to the footprint; the harness footprint (13,550,384 B) does not include it.
- Synaptic ops and activation sparsity: session-0 values, as in the existing rows. Averaged over all 11
  sessions, Eff_ACs are 68.50M ± 5.90M per sample (existing SNN rerun: 71.20M); one of the three backbone seeds
  is above the baseline (76.72M).
- Connection sparsity 0.049 is the neurobench 2.3.0 metric; at session 0 the 100 not-yet-learned class rows of
  the readout are zero. The existing SNN gives 0.0199 with the same metric.
- Hyperparameters were chosen on a pseudo-incremental protocol built from the base train/val splits only;
  the evaluation languages were used for 2 official runs in total (the baseline rerun and this entry).

## Code and reproduction
- Code: https://github.com/tunadeniz1304/NeuroBench, tag `v1-final`.
- Checkpoints: three backbones, 13.6 MB each (`cos_1024_amp_clip_s{0,1,2}.pt`), attached to
  https://github.com/tunadeniz1304/NeuroBench/releases/tag/v1-final (SHA256 in `results/checkpoints.sha256`).
- Command: `./reproduce.sh final` then `CONFIRM_OFFICIAL=1 ./reproduce.sh official`
  (`FINAL_SYSTEM_CFG=configs/final/clip_cl2n_8bit.yaml`, the default).
- Hardware: final backbones trained on 1× NVIDIA A100 (about 25 s per epoch, 50 epochs).

## Checklist
- [ ] Row added to `leaderboard.rst` with aligned RST grid columns
- [ ] Docs build passes
- [ ] Numbers match the attached official-run log
