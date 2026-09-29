# Leaderboard row draft (Keyword FSCIL, `leaderboard.rst`)

Draft only. Fill in from the final official run in `results/official_runs.jsonl` (mean over seeds;
std goes into the PR text). Not submitted.

Format copied from upstream `leaderboard.rst` (commit e521c28):

```rst
+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+---------+--------------------+--------------------+---------------+
| Method    | Accuracy (Base / Session Average) | Footprint | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense   | Eff_MACs           | Eff_ACs            | Date Submitted|
+===========+===================================+===========+==================+=====================+=====================+=========+====================+====================+===============+
| M5 ANN    | (97.09% / 89.27%)                 | 6.03E6    | 1                | 0.0                 | 0.783               | 2.59E7  | 7.85E6             | 0                  | 2024-01-17    |
+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+---------+--------------------+--------------------+---------------+
| SNN       | (93.48% / 75.27%)                 | 1.36E7    | 200              | 0.0                 | 0.916               | 3.39E6  | 0                  | 3.65E5             | 2024-01-17    |
+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+---------+--------------------+--------------------+---------------+
| TODO name | (TODO% / TODO%)                   | TODO      | 200              | TODO                | TODO                | TODO    | 0                  | TODO               | TODO          |
+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+---------+--------------------+--------------------+---------------+
```

## Checklist before filling in
- [ ] Numbers come from one official run (config + commit recorded), mean over ≥ 3 seeds.
- [ ] Base accuracy is measured with the incremental (prototype) readout; say so in the PR, because the
      existing SNN row uses the backprop-trained readout for base accuracy.
- [ ] Footprint includes the learner state beyond parameters (`extra_footprint_bytes`), or the PR states
      explicitly that it is listed separately.
- [ ] Eff_ACs / Dense / Activation Sparsity: which session they refer to (upstream copies session 0).
- [ ] Units: neurobench 2.3.0 gives Dense / Eff_ACs summed over T = 200 steps; the existing rows are ≈ 200×
      smaller (our baseline rerun: 6.74E8 / 7.12E7 vs 3.39E6 / 3.65E5). Report both, or ask the maintainers
      which convention the table uses; never mix them in one table.
- [ ] Column widths re-aligned so the RST grid table still parses (`rst2html` / docs build).
