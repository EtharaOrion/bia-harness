#!/usr/bin/env bash
# Harbor verifier entry point. Terminates by writing the bound reward path.
set -euo pipefail
cd "$(dirname "$0")/.."

export TRACK3_BUNDLE="${TRACK3_BUNDLE:-$(pwd)}"
export REWARD_PATH="${REWARD_PATH:-$(pwd)/reward.json}"
export TRACK3_OUTCOMES="${TRACK3_OUTCOMES:-/tmp/track3_outcomes.json}"

# grade.py always writes the reward record, including on every failure path, so a zero is
# never attributable to an unwritten or empty reward file.
python3 tests/grade.py

# Compiled rubric tests assert the relation only. They run after grading because they read
# the outcomes grade.py emitted.
python3 -m pytest -q tests/test_output.py || true

test -s "$REWARD_PATH" || { echo "FATAL: reward file unwritten"; exit 1; }
