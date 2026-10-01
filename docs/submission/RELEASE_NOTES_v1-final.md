Final configuration for the NeuroBench v1.0 Keyword FSCIL task (MSWC): RadLIF RSNN backbone pretrained with a
centred cosine classifier and relative gradient clipping, plus an integer three-factor Hebbian learner
(centred, L2-normalised 8-bit prototypes) for the incremental sessions.

Official harness result (neurobench 2.3.0, seeds 0-2, `results/official_runs.jsonl`):

| | Ours | SNN baseline (rerun) |
|---|---|---|
| Session average | 86.93 ± 0.40% | 76.09 ± 0.36% |
| Base (prototype readout) | 93.48 ± 0.26% | 84.21 ± 0.06% |
| Footprint | 13,550,384 B + 5,124 B learner state | 13,550,384 B |
| Eff_ACs (mean over sessions, per sample) | 68.50M ± 5.90M | 71.20M |

Attached: the three final backbones (official seed s uses `cos_1024_amp_clip_s{s}.pt`). Verify with
`sha256sum -c results/checkpoints.sha256`, put them in `$CKPT_DIR` and run
`CONFIRM_OFFICIAL=1 ./reproduce.sh official`. Details: `report.md`.
