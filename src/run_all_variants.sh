#!/usr/bin/env bash
set -e

# script is in repo/src → root is one level up
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

VARIANTS=(
  external_0
  o_010
  lr_half
  wd_high
)

SEED=42
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

for v in "${VARIANTS[@]}"; do
  LOG_FILE="$LOG_DIR/${v}.log"

  echo "===== Running variant: $v ====="
  echo "Logging to: $LOG_FILE"

  python src/train.py \
    --variant "$v" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_FILE"

  echo "===== Finished variant: $v ====="
done
