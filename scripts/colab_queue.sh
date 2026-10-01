#!/usr/bin/env bash
# Sequential train + pseudo_eval queue that survives runtime resets.
# Usage: bash scripts/colab_queue.sh <config.yaml>:<fold>:<seed> [...]
#   fold "all" trains on all 100 base classes (final backbone, <name>_s<seed>.pt); it is not pseudo-evaluated,
#   since the pseudo protocol needs held-out classes.
# Checkpoints, logs and pseudo-eval results go to $CKPT_DIR (Drive by default). A checkpoint that reached its
# last epoch is not retrained; a checkpoint whose results are already in $CKPT_DIR/pseudo_eval.jsonl is not
# re-evaluated, and an interrupted training resumes from its last finished epoch (nbfscil.train keeps a
# <ckpt>.resume file). Re-running the same command after a reset therefore continues where it stopped.
# DRY=1 only prints what would run.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CK="${CKPT_DIR:-/content/drive/MyDrive/nbfscil_ckpts}"
RES="$CK/pseudo_eval.jsonl"
LOGS="$CK/logs"
mkdir -p "$LOGS"

ckpt_done() {
  [[ -f "$1" ]] && python - "$1" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sys.exit(0 if ck.get("epoch") == ck["cfg"]["train"]["epochs"] else 1)
PY
}

for job in "$@"; do
  IFS=: read -r cfg fold seed <<< "$job"
  name="$(basename "$cfg" .yaml)"
  if [[ "$fold" == all ]]; then
    tag="s${seed}"; fold_arg=()
  else
    tag="f${fold}_s${seed}"; fold_arg=(--fold "$fold")
  fi
  out="$CK/${name}_${tag}.pt"
  echo "=== $name fold $fold seed $seed $(date -u +%H:%M)"
  if ckpt_done "$out"; then
    echo "    trained already"
  elif [[ -n "${DRY:-}" ]]; then
    echo "    would train -> $out"
  else
    python -m nbfscil.train --config "$cfg" --seed "$seed" "${fold_arg[@]}" --out "$out" \
      > "$LOGS/train_${name}_${tag}.log" 2>&1 || { echo "    TRAIN FAILED (see $LOGS)"; continue; }
  fi
  if [[ "$fold" == all ]]; then
    echo "    DONE $(date -u +%H:%M)"
  elif grep -qF "\"$out\"" "$RES" 2>/dev/null; then
    echo "    evaluated already"
  elif [[ -n "${DRY:-}" ]]; then
    echo "    would evaluate -> $RES"
  else
    python -m nbfscil.pseudo_eval --ckpt "$out" --learner configs/learners/*.yaml --seeds 0 1 2 3 4 >> "$RES" \
      && echo "    DONE $(date -u +%H:%M)" || echo "    EVAL FAILED"
  fi
done
echo "=== ALL DONE $(date -u +%H:%M)"
