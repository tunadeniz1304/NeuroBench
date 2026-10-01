# Leaderboard submission steps (done by the repo owner)

Nothing here is sent automatically. Numbers: official run 2/5 in `results/official_runs.jsonl`.

## 1. Checkpoints
Attach the three final backbones to a GitHub Release of tag `v1-final` in this repository (they are not
committed to git):

| File | Size | SHA256 |
|---|---|---|
| `cos_1024_amp_clip_s0.pt` | 13.6 MB | to be filled from the Colab hash cell |
| `cos_1024_amp_clip_s1.pt` | 13.6 MB | to be filled from the Colab hash cell |
| `cos_1024_amp_clip_s2.pt` | 13.6 MB | to be filled from the Colab hash cell |

To evaluate with them, put the files in `$CKPT_DIR` and run `CONFIRM_OFFICIAL=1 ./reproduce.sh official`.
The PR links this repository, so it must be public before the PR is opened.

## 2. Leaderboard row
Upstream asks for PRs against the `dev` branch (`CONTRIBUTING.rst`).

```bash
git clone https://github.com/<your-user>/neurobench.git   # your fork of NeuroBench/neurobench
cd neurobench
git checkout -b keyword-fscil-snn-cl2n origin/dev
git apply /path/to/NeuroBench/docs/submission/leaderboard_rst.patch
# replace YYYY-MM-DD in leaderboard.rst with the submission date (same width, 10 characters)
git commit -am "Add SNN-CL2N entry to the Keyword FSCIL leaderboard"
git push -u origin keyword-fscil-snn-cl2n
```

The patch was checked against upstream commit e521c28 (`main`); if `dev` has moved, add the row by hand below
the `SNN` row. The Keyword FSCIL table parses with docutils after the change (the task overview table at the top
of `leaderboard.rst` already reports "Malformed table" upstream, independent of this change).

## 3. Pull request
Open the PR from the branch above into `NeuroBench/neurobench:dev`, with the title and text of `PR_DRAFT.md`.
The PR asks the maintainers which unit convention the Dense / Eff_ACs columns use; the row uses the
neurobench 2.3.0 values divided by T = 200.
