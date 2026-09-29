#!/usr/bin/env bash
# Reproduce the NeuroBench Keyword FSCIL (MSWC) results of this repo.
#
# Usage: ./reproduce.sh <stage> [<stage> ...]
#   env       install pinned dependencies, clone upstream NeuroBench (pinned commit) into third_party/
#   cache     download MSWC FSCIL subset (via neurobench) and build the S2S / MFCC caches in $NB_DATA/cache
#   test      run pytest (data-dependent tests skip if the cache is missing)
#   baseline  train the upstream SNN recipe on our RSNN re-implementation (all 100 base classes)
#   pseudo    train pseudo-fold backbones and run the Phase 3 pseudo-incremental protocol
#   final     train the final backbone(s) on all 100 base classes
#   official  run the official harness evaluation (COUNTS AGAINST THE 5-RUN BUDGET, see below)
#
# Hardware used for the reported numbers: 1x Tesla T4 (15 GB), 2 CPU cores, Python 3.13, CUDA 12.8 wheels.
# `official` is never part of a default run: every invocation of nbfscil.official_eval is one of at most
# five official runs (CLAUDE.md) and is appended to results/official_runs.jsonl. It only executes with
# CONFIRM_OFFICIAL=1 set explicitly.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

export NB_DATA="${NB_DATA:-/content/data}"      # fast local disk, never inside the repo
CKPT_DIR="${CKPT_DIR:-$NB_DATA/ckpts}"
LOG_DIR="${LOG_DIR:-$REPO/runs_tmp}"           # gitignored
UPSTREAM_COMMIT="e521c28"
SEEDS=(0 1 2)
PSEUDO_FOLDS=(0 1)

# Final configuration. Filled in after the Phase 4 decision on the pseudo protocol.
FINAL_TRAIN_CFG="${FINAL_TRAIN_CFG:-TODO}"     # e.g. configs/<final_train>.yaml
FINAL_SYSTEM_CFG="${FINAL_SYSTEM_CFG:-TODO}"   # system: proto, checkpoint: ..., learner: {...}
PSEUDO_TRAIN_CFGS=(${PSEUDO_TRAIN_CFGS:-configs/rsnn_baseline_train.yaml})
PSEUDO_LEARNERS=(${PSEUDO_LEARNERS:-configs/learners/euclid.yaml configs/learners/float_cl2n.yaml
                 configs/learners/cl2n_8bit.yaml configs/learners/cl2n_4bit.yaml
                 configs/learners/center_only_8bit.yaml configs/learners/l2_only_8bit.yaml})

require_final() {
  if [[ "$FINAL_TRAIN_CFG" == TODO || "$FINAL_SYSTEM_CFG" == TODO ]]; then
    echo "FINAL_TRAIN_CFG / FINAL_SYSTEM_CFG are not set yet (Phase 4 not finished)." >&2
    exit 1
  fi
}

stage_env() {
  python -m pip install -r requirements.txt
  if [[ ! -d third_party/neurobench/.git ]]; then
    git clone https://github.com/NeuroBench/neurobench third_party/neurobench
  fi
  git -C third_party/neurobench checkout "$UPSTREAM_COMMIT"
}

stage_cache() {
  mkdir -p "$NB_DATA"
  python -m nbfscil.cache --encoding s2s
  python -m nbfscil.cache --encoding mfcc --splits base_train base_val   # M5 reference, pseudo protocol only
}

stage_test() {
  python -m pytest -q -rs
}

stage_baseline() {
  mkdir -p "$CKPT_DIR" "$LOG_DIR"
  for s in "${SEEDS[@]}"; do
    python -m nbfscil.train --config configs/rsnn_baseline_train.yaml --seed "$s" \
      --out "$CKPT_DIR/rsnn_baseline_s$s.pt" | tee "$LOG_DIR/rsnn_baseline_s$s.log"
  done
}

stage_pseudo() {
  mkdir -p "$CKPT_DIR" "$LOG_DIR"
  for cfg in "${PSEUDO_TRAIN_CFGS[@]}"; do
    name="$(basename "$cfg" .yaml)"
    for fold in "${PSEUDO_FOLDS[@]}"; do
      for s in "${SEEDS[@]}"; do
        ck="$CKPT_DIR/${name}_f${fold}_s$s.pt"
        [[ -f "$ck" ]] || python -m nbfscil.train --config "$cfg" --seed "$s" --fold "$fold" --out "$ck" \
          | tee "$LOG_DIR/${name}_f${fold}_s$s.log"
        python -m nbfscil.pseudo_eval --ckpt "$ck" --learner "${PSEUDO_LEARNERS[@]}" --seeds 0 1 2 3 4 \
          | tee -a "$LOG_DIR/pseudo_eval.jsonl"
      done
    done
  done
}

stage_final() {
  require_final
  mkdir -p "$CKPT_DIR" "$LOG_DIR"
  name="$(basename "$FINAL_TRAIN_CFG" .yaml)"
  for s in "${SEEDS[@]}"; do
    python -m nbfscil.train --config "$FINAL_TRAIN_CFG" --seed "$s" --out "$CKPT_DIR/${name}_s$s.pt" \
      | tee "$LOG_DIR/${name}_s$s.log"
  done
}

stage_official() {
  require_final
  if [[ "${CONFIRM_OFFICIAL:-0}" != 1 ]]; then
    echo "Refusing: 'official' uses one of the 5 official eval runs. Re-run with CONFIRM_OFFICIAL=1." >&2
    exit 1
  fi
  python -m nbfscil.official_eval --config "$FINAL_SYSTEM_CFG" --seeds "${SEEDS[@]}" \
    --note "reproduce.sh final config"
}

if [[ $# -eq 0 ]]; then
  sed -n '2,16p' "$0"
  exit 1
fi
for stage in "$@"; do
  case "$stage" in
    env|cache|test|baseline|pseudo|final|official) "stage_$stage" ;;
    *) echo "unknown stage: $stage" >&2; exit 1 ;;
  esac
done
