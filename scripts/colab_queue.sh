#!/usr/bin/env bash
# Sequential train + pseudo_eval queue that survives runtime resets.
# Usage: bash scripts/colab_queue.sh <config.yaml>:<fold>:<seed> [...]
# Checkpoints, logs and pseudo-eval results go to $CKPT_DIR (Drive by default). A checkpoint that reached its
# last epoch is not retrained; a checkpoint whose results are already in $CKPT_DIR/pseudo_eval.jsonl is not
# re-evaluated. Re-running the same command after a reset therefore continues where it stopped.
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
  out="$CK/${name}_f${fold}_s${seed}.pt"
  echo "=== $name fold $fold seed $seed $(date -u +%H:%M)"
  if ckpt_done "$out"; then
    echo "    trained already"
  elif [[ -n "${DRY:-}" ]]; then
    echo "    would train -> $out"
  else
    python -m nbfscil.train --config "$cfg" --seed "$seed" --fold "$fold" --out "$out" \
      > "$LOGS/train_${name}_f${fold}_s${seed}.log" 2>&1 || { echo "    TRAIN FAILED (see $LOGS)"; continue; }
  fi
  if grep -qF "\"$out\"" "$RES" 2>/dev/null; then
    echo "    evaluated already"
  elif [[ -n "${DRY:-}" ]]; then
    echo "    would evaluate -> $RES"
  else
    python -m nbfscil.pseudo_eval --ckpt "$out" --learner configs/learners/*.yaml --seeds 0 1 2 3 4 >> "$RES" \
      && echo "    DONE $(date -u +%H:%M)" || echo "    EVAL FAILED"
  fi
done
echo "=== ALL DONE $(date -u +%H:%M)"
