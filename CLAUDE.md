# CLAUDE.md — standing rules for this repo

Work autonomously. Do not ask questions. Make decisions, log why in `EXPERIMENT_LOG.md`, keep going.
Stop only for public or irreversible actions.

## Resume procedure
Read `EXPERIMENT_LOG.md`, `results/runs.csv` and the task list, then continue from the last passed gate.

## Mission
Beat the published SNN results on the NeuroBench v1.0 Keyword Few-Shot Class-Incremental Learning (FSCIL)
task (MSWC) with an original, defensible method — not just hyperparameter tuning.
Baselines: M5 ANN 97.09% base / 89.27% session avg; SNN 93.48% / 75.27% (the one to beat).

## Win condition (in order)
1. The SNN's session average rises clearly above 75.27%, ideally near or above 89.27%.
2. Effective synaptic ops (Eff_ACs) and footprint stay at or below the SNN baseline, or the trade-off is explicitly justified on a Pareto plot.
3. Learning in incremental sessions uses a **hardware-plausible local rule**: no backprop through the network, integer/low-bit arithmetic, only local and neuromodulatory signals. It must be implementable later as a small digital neuromorphic RTL block.

## Never do (verbatim)
- **Tuning on the official test/eval split.** All hyperparameter and design decisions are made on the validation protocol in Phase 3. The official evaluation is only for final configs, at most **5 runs total**. Log every official eval run in `EXPERIMENT_LOG.md`, including bad ones.
- Changing the task definition (number of shots, classes, sessions, data splits) or the harness metric code to get better numbers.
- Leaking data from future sessions or test data into training.
- Reporting only the best seed. Report mean ± std over at least 3 seeds.
- Hiding or omitting metrics (footprint, sparsity, ops). Report all harness metrics.
- Opening a PR to the upstream NeuroBench repo, uploading to arXiv, or posting anything publicly. I will review and submit those myself.
- Pushing to any remote other than `origin` (`https://github.com/tunadeniz1304/NeuroBench.git`). No `git push --force`, no history rewrite (`rebase -i`, `reset --hard` on pushed commits, `filter-branch`) on `main`.
- **Any AI attribution in git.** No `Co-Authored-By: Claude` (or any Co-Authored-By trailer), no "Generated with Claude Code" lines, no Claude/Anthropic mentions in commit messages, commit authors or PR texts. Commits are authored only with the git identity already configured on the machine; never set or fake `user.name`/`user.email` yourself.
- Committing datasets, raw audio, secrets/tokens, `.env` files, virtualenvs, or files larger than 50 MB.
- Deleting data or checkpoints you did not create yourself.

## Git and commit rules (verbatim)
- **Conventional Commits**, in English: `type(scope): summary` with types `feat`, `fix`, `exp`, `data`, `eval`, `docs`, `test`, `refactor`, `chore`. Summary in imperative mood, max ~72 chars, no trailing period. Add a short body when the why isn't obvious (for `exp` commits: hypothesis, config, key result).
  - Examples: `feat(model): add adaptive LIF neuron with learnable leak`, `exp(fscil): three-factor rule, 4-bit weights, pseudo-val session avg 81.2`, `fix(data): exclude pseudo-novel classes from base train split`.
- **Atomic commits:** one logical change per commit. Code, configs and results of one experiment go together; unrelated refactors go separately. No "wip", "update", "misc" or "final" commits.
- Never commit broken code on `main`: run `pytest` (and a quick smoke run for training code) before committing.
- **Tag** milestones: `v0-baseline-repro` after Phase 2, `v1-final` after Phase 5.
- **Push to `origin main`** after every passed gate and at least every few commits during Phase 4, so progress survives a runtime crash.
- Never include Co-Authored-By trailers or any AI/Claude attribution anywhere in git (see Section 2).

## Environment
- Dataset on fast local disk: `$NB_DATA` (default `/content/data`), never in the repo or Drive.
- Upstream NeuroBench in `third_party/neurobench` (gitignored); never commit it.
