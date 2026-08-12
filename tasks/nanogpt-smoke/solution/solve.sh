#!/bin/bash
# Oracle: emit a log the grader will accept.
set -euo pipefail

mkdir -p /logs/artifacts/runs
LOG=/logs/artifacts/runs/smoke.log
{
    python --version
    python -c "import torch; print(f'torch:{torch.__version__} cuda_available:{torch.cuda.is_available()}')"
    (nvidia-smi -L 2>/dev/null || echo 'nvidia-smi: not present (ok)')
} | tee "$LOG"
