#!/usr/bin/env bash
# Reference solution. Proves the task is solvable: reaches val_loss 3.28 at step 2672
# on the slower of the two seeds, which is reward 1.0 (TARGET_STEPS = 2900).
set -euo pipefail
mkdir -p /workspace/submission
cp "$(dirname "$0")/optimizer.py" /workspace/submission/optimizer.py
python3 /workspace/runner/train_mini.py \
  --submission /workspace/submission \
  --target 3.28
