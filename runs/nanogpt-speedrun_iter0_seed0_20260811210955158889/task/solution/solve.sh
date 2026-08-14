#!/bin/bash
# Oracle solution: run the unmodified canonical baseline.
set -euo pipefail

cd /app/environment

NPROC=$(nvidia-smi -L | wc -l)
if [ "$NPROC" -eq 0 ]; then
    echo "ERROR: no GPUs detected" >&2
    exit 1
fi

mkdir -p /logs/artifacts/runs
LOG=/logs/artifacts/runs/oracle.log

torchrun --standalone --nproc_per_node="$NPROC" train_gpt_simple.py 2>&1 | tee "$LOG"
