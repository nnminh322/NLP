#!/usr/bin/env bash
set -euo pipefail

# Run from this script's directory (source/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure local src is on PYTHONPATH so the editable package is importable
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
# Optional: run training before hybrid evaluation. Toggle with RUN_TRAIN=0 to skip.
RUN_TRAIN=${RUN_TRAIN:-1}
DISABLE_ENTITY_SIGNAL=${DISABLE_ENTITY_SIGNAL:-1}
ENTITY_SIGNAL_ARGS=()
if [ "$DISABLE_ENTITY_SIGNAL" -eq 1 ]; then
  ENTITY_SIGNAL_ARGS+=(--disable-entity-signal)
fi
if [ "$RUN_TRAIN" -eq 1 ]; then
  echo "Running training: LoRA r=16, epochs=5, batch_size=8"
  python -m gsr_cacl.train \
    --dataset tatqa \
    --stage all \
    --preset t4 \
    --epochs 5 \
    --lora-r 16 \
    --lora-alpha 64 \
    --batch-size 4 \
    --gradient-checkpointing \
    --save ./outputs/tatqa_lora_r32_ep10 \
    "${ENTITY_SIGNAL_ARGS[@]}"
fi

echo "Running hybrid evaluation (GSR baseline → BM25 + RRF fusion)"

python -m gsr_cacl.tools.hybrid_eval \
  --dataset tatqa \
  --sample 200 \
  --candidate-n 50 \
  --top-k 3 \
  --output-dir outputs/hybrid_eval/tatqa_run \
  "${ENTITY_SIGNAL_ARGS[@]}"

echo "Done. Outputs in outputs/hybrid_eval/tatqa_run"
