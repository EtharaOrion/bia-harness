#!/usr/bin/env bash
# Harbor verifier entrypoint. Runs in the agent's own container ([verifier] declares
# no environment, so environment_mode defaults to 'shared'), which is why grade.py is
# stdlib-only: there is nothing to install and nothing to keep in sync with a second
# image.
set -uo pipefail
mkdir -p /logs/verifier
cd /workspace

export MINICALC_BUNDLE=/workspace
export MINICALC_SUBMISSION=/workspace/submission
export REWARD_PATH=/logs/verifier/reward.json

# harbor copies the task's tests/ to the reserved /tests path
python3 /tests/grade.py 2>&1 | tee /logs/verifier/grade-stdout.md
# grade.py's status, not tee's. grade.py discards main()'s return value and exits 0 for
# every graded outcome including a legitimate 0.0, so a non-zero here means the VERIFIER
# failed, not that the submission scored badly.
GRADE_RC=${PIPESTATUS[0]}

test -s "$REWARD_PATH" || echo '{"reward":0.0,"reason":"verifier_produced_no_reward"}' > "$REWARD_PATH"

# Harbor parses EVERY key in reward.json as a number, so a string `reason` triggers
# pydantic int_parsing errors and is silently dropped. Emit numeric-only for harbor
# and keep the full record in reward_full.json -- which is the file
# runner/agentloop/trial_io.py reads, because it is the only one carrying `reason`
# and `metrics.n_seeds`. BOTH NAMES AND SHAPES ARE FIXED BY THAT READER.
# score.md is the delivery format's bare score and is written here rather than by the
# emitter, so it exists even when the emitter is skipped.
python3 - "$REWARD_PATH" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
pathlib.Path("/logs/verifier/reward_full.json").write_text(json.dumps(d, indent=2))
num = {k: v for k, v in d.items()
       if isinstance(v, (int, float)) and not isinstance(v, bool)}
num.setdefault("reward", float(d.get("reward", 0.0)))
p.write_text(json.dumps(num))
pathlib.Path("/logs/verifier/score.md").write_text(f"{num['reward']}\n")
PY

cat "$REWARD_PATH"; echo; echo "--- full record ---"; cat /logs/verifier/reward_full.json

# Advisory only; a failure here never changes the score. pytest is NOT installed in
# bia/minicalc:v1 -- environment_mode resolves to "shared", so this runs in the agent's
# own bare-python image -- and the bundle deliberately ships no test_output.py, which
# would be permanently unexecutable. Probe anyway and say so plainly: piping a
# "No module named pytest" into the log looks identical to a suite that ran and asserted
# nothing, which is the one outcome an advisory check must never be confused with.
if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest /tests -v --no-header -p no:cacheprovider \
    2>&1 | tee /logs/verifier/test-stdout.md || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so no advisory suite ran."
    echo "This bundle ships no tests/test_output.py for the same reason; the score"
    echo "comes from grade.py alone and is unaffected."
    echo "Nothing was asserted here -- do not read this as a passing suite."
  } | tee /logs/verifier/test-stdout.md
fi

if [ "$GRADE_RC" -ne 0 ]; then
  {
    echo
    echo "VERIFIER FAULT: grade.py exited $GRADE_RC. The reward above is the fallback,"
    echo "not a measurement of the submission. Do not report it as a score."
  } | tee -a /logs/verifier/test-stdout.md >&2
fi

# Delivery-format artifacts: verifier/score.json (numeric-only, the reason as an integer
# reason_code) and the full record appended to grade-stdout.md. The second argument is
# where the seed logs actually live in the container -- the run dir is /logs, and
# /logs/artifacts is empty until harbor collects. Non-fatal: a defect in these review
# aids must never fail a legitimate grade.
python3 /tests/emit_verifier_artifacts.py /logs /workspace/submission || true

# GRADE_RC is grade.py's own status, taken from PIPESTATUS[0] above so the tee does not
# mask it. A non-zero exit makes harbor record a trial-level failure rather than a
# reward-0 row, so a crashed verifier is never scored as a legitimate miss.
exit "$GRADE_RC"
