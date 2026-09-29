# PR draft for NeuroBench (upstream) — not opened

Draft text for a leaderboard submission PR to `NeuroBench/neurobench`. The repo owner reviews and
submits it. Everything marked TODO comes from logged results only.

---

**Title:** Add Keyword FSCIL leaderboard entry: TODO name (SNN with integer three-factor prototype learning)

## Summary

Adds one row to the Keyword FSCIL (MSWC) leaderboard. The model is a RadLIF recurrent SNN with the same
input encoding as the existing SNN baseline (Speech2Spikes, 48 kHz, hop 240). In the incremental sessions
it learns new classes with a local, integer three-factor Hebbian rule (centered, L2-normalized k-bit
prototypes); there is no backprop after the base session.

| Method | Base / Session avg | Footprint | Exec. rate | Conn. sparsity | Act. sparsity | Dense | Eff_MACs | Eff_ACs |
|---|---|---|---|---|---|---|---|---|
| SNN (existing) | 93.48% / 75.27% | 1.36E7 | 200 | 0.0 | 0.916 | 3.39E6 | 0 | 3.65E5 |
| This entry | TODO ± TODO / TODO ± TODO | TODO | 200 | TODO | TODO | TODO | 0 | TODO |

Mean ± std over TODO seeds (session orders and support shots differ per seed).

## Evaluation details
- neurobench 2.3.0, unmodified `Benchmark`, metrics and task definition (100 base classes,
  10 sessions × 10-way 5-shot, support/query split (100, 100)).
- Base accuracy is measured with the prototype readout that is also used in sessions 1–10
  (the existing SNN row reports base accuracy with the trained readout).
- Inputs were S2S-encoded once with the upstream `S2SPreProcessor` configuration and cached; this is
  equivalent to encoding inline.
- Audio backend: torchaudio TODO version (upstream baseline used 2.0.2 + sox_io).
- Learner state beyond model parameters: TODO bytes (TODO: included in / listed next to footprint).
- Synaptic ops and activation sparsity: TODO (session 0 value, as in the existing rows / per-session average).
- Hyperparameters were chosen on a pseudo-incremental protocol built from the base train/val splits only;
  the evaluation languages were used for TODO final runs.

## Code and reproduction
- Code: TODO link to the public repository / commit.
- Checkpoint: TODO (size, location).
- Command: `CONFIRM_OFFICIAL=1 ./reproduce.sh official` with `FINAL_SYSTEM_CFG=TODO`.
- Hardware: 1× Tesla T4.

## Checklist
- [ ] Row added to `leaderboard.rst` with aligned RST grid columns
- [ ] Docs build passes
- [ ] Numbers match the attached official-run log
