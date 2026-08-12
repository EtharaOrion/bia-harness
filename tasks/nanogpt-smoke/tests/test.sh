#!/bin/bash
# Harbor verifier entrypoint.
set -euo pipefail

RUNS_DIR=/logs/artifacts/runs
OUT_DIR=/logs/verifier
mkdir -p "$OUT_DIR"

LOG=$(ls -t "$RUNS_DIR"/*.log 2>/dev/null | head -n1 || true)
if [ -z "$LOG" ]; then
    echo '{"reward": 0.0, "reason": "no log"}' > "$OUT_DIR/reward.json"
    cat "$OUT_DIR/reward.json"
    exit 1
fi

python3 "$(dirname "$0")/grader.py" --write "$LOG" "$OUT_DIR/reward.json"
