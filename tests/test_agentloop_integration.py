"""End-to-end integration tests for the agentloop refinement loop: real harbor, real docker.

Everything else in the agentloop test suite replaces `subprocess.run` with a fake that
materialises a fixture trial. That proves the loop's logic but not that the config we
build is one harbor will actually accept, nor that a real trial directory parses. This
module closes that gap by launching the real binary against the real image -- and it is
the only test in the suite that does, which is exactly why it is guarded four ways.

WHY IT IS OPT-IN. A single iteration starts a GPU container, an egress-control sidecar
and a verifier, and takes ~20 seconds even with the `nop` agent. That is far too slow
and far too environment-dependent to belong in `pytest tests/`, so the module skips
unless `TRACK3_INTEGRATION=1` is set explicitly. The env gate is checked FIRST and
short-circuits, so a normal test run never even shells out to `docker`.

WHAT IT ASSERTS BEYOND "IT RAN". Three of the assertions are isolation properties, not
functional ones, and they are the reason this file exists:

* `export_traces` must NOT appear in the written config. It is a CLI flag. Harbor's
  JobConfig is pydantic `extra="ignore"`, so the key would be silently DROPPED and the
  job would run with no trajectory export while the config looked correct.
* The string `track3-pipeline` must appear NOWHERE in the written config. That is the
  read-only 65GB production tree this loop was ported from; a path leaking into
  `jobs_dir` or `tasks[].path` would write into it.
* The run root is redirected into `tmp_path`, so a test run can never append to the
  real campaign ledger under `runs/agentloop/`.

CONTAINER HYGIENE. Harbor tears its compose project down, but a killed run leaves
`task__<id>__env-main-1` and `task__<id>__env-harbor-docker-egress-control-sidecar-1`
behind. `no_leaked_containers` snapshots the `task__*` containers that existed BEFORE
the test and force-removes only names that appear afterwards. Containers present in the
"before" snapshot are never touched: on a shared host they belong to somebody else's
run -- possibly a multi-hour campaign -- and killing them would be catastrophic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agentloop import loop
from agentloop.loop import refine

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

TASK_DIRNAME = "2739a678-1759-516d-8ba7-1cd023267ea8"
TASK_IMAGE = "bia/track3nov:v2"
FIXTURE_TRIAL = Path(__file__).parent / "fixtures" / "track3_trial"

# Every outcome `classify` can return. The integration test asserts membership rather
# than a specific value: which one a nop run lands on is a property of the verifier's
# reaction to an empty submission, and pinning it here would make this test fail for a
# reason that has nothing to do with the loop.
KNOWN_OUTCOMES = frozenset({
    "graded_pass",
    "graded_miss",
    "gate_fail",
    "agent_abandoned_run",
    "harness_incomplete",
    "unknown",
})

# Wall-clock ceiling for one harbor invocation. A nop iteration measures ~20s; this is
# a runaway guard, not a budget. `refine` passes no timeout of its own, and a test that
# can hang for the task's 8-hour agent budget is not a test.
HARBOR_TIMEOUT_SEC = 600.0

# The `nop` agent starts the container and does nothing. It exercises the whole harbor
# path -- image, mounts, egress sidecar, verifier, trial layout -- without an LLM, a
# network bridge, or GPU minutes.
NOP_AGENT_OVERRIDE = {"agents": [{"name": "nop"}]}

CONTAINER_PREFIX = "task__"


# --------------------------------------------------------------------------- #
# skip guards
# --------------------------------------------------------------------------- #


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True,
                              timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _image_present(image: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", image],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _skip_reason() -> str | None:
    """Why this module cannot run here, or None if it can.

    Ordered cheapest-first and short-circuiting: the env gate is evaluated before any
    subprocess, so collecting this module during a normal `pytest tests/` costs nothing.
    """
    if os.environ.get("TRACK3_INTEGRATION") != "1":
        return ("TRACK3_INTEGRATION=1 not set: these tests launch real docker "
                "containers via harbor (~20s each) and are opt-in")

    from harness import HARNESS_ROOT, resolve_harbor_bin

    harbor_bin = resolve_harbor_bin()
    if not Path(harbor_bin).is_file():
        return f"harbor binary not found at {harbor_bin!r}"
    if not (HARNESS_ROOT / "tasks" / TASK_DIRNAME / "task.toml").is_file():
        return f"task bundle not present: tasks/{TASK_DIRNAME}"
    if not _docker_ok():
        return "`docker info` returned nonzero: docker daemon unavailable"
    if not _image_present(TASK_IMAGE):
        return f"docker image {TASK_IMAGE} not present locally"
    return None


SKIP_REASON = _skip_reason()

pytestmark = pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _task_containers() -> set[str]:
    """Names of every existing container (any state) whose name starts with `task__`."""
    try:
        proc = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if proc.returncode != 0:
        return set()
    return {n.strip() for n in (proc.stdout or "").splitlines() if n.strip()}


@pytest.fixture
def no_leaked_containers():
    """Force-remove the `task__*` containers THIS test created, and prove none survive.

    The before/after set difference is load-bearing. A blanket `docker rm -f task__*`
    would be simpler and would also destroy any concurrently running harbor job on the
    same host, so containers that predate the test are excluded from removal by
    construction rather than by a filter that could be got wrong.

    The assertion is that nothing this test spawned OUTLIVES it, not that harbor's own
    teardown was perfect: harbor is expected to remove its project itself, and on a
    killed run it does not. Cleanup runs in a `finally` so it happens even when the
    test body fails -- a failed assertion must not also leak a GPU container.
    """
    before = _task_containers()
    removed: set[str] = set()
    try:
        yield removed
    finally:
        spawned = _task_containers() - before
        for name in sorted(spawned):
            subprocess.run(["docker", "rm", "-f", name],
                           capture_output=True, timeout=120)
            removed.add(name)
        survivors = _task_containers() & spawned
        assert not survivors, (
            f"containers spawned by this test could not be removed: {sorted(survivors)}"
        )


@pytest.fixture
def task_dir():
    from harness import HARNESS_ROOT

    return HARNESS_ROOT / "tasks" / TASK_DIRNAME


@pytest.fixture
def isolated_run_root(monkeypatch, tmp_path):
    """Redirect the loop's run root into tmp so the real campaign ledger is untouched.

    `refine` resolves its run root through `loop.resolve_run_root`, so patching that one
    name moves the ledger, the history dir, the facts checkpoints, the written configs
    and `jobs_dir` together -- there is no second place a stray artifact can land.
    """
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: run_root)
    return run_root


@pytest.fixture
def bounded_harbor(monkeypatch):
    """Cap every harbor invocation at `HARBOR_TIMEOUT_SEC`.

    `refine` exposes no timeout, and `run_iteration` defaults to None. Wrapping the real
    function keeps the code under test genuine while making a stalled container fail the
    test in ten minutes instead of hanging a CI worker for the task's 8-hour budget.
    """
    real = loop.run_iteration

    def bounded(*args, **kwargs):
        kwargs.setdefault("timeout", HARBOR_TIMEOUT_SEC)
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "run_iteration", bounded)


# --------------------------------------------------------------------------- #
# the real thing: harbor + docker
# --------------------------------------------------------------------------- #


def test_one_real_iteration_records_a_ledger_row_and_leaks_nothing(
    task_dir, isolated_run_root, bounded_harbor, no_leaked_containers
):
    """One `nop` iteration against real harbor + real docker, end to end.

    Summarisation and judging are off: both reach the LLM bridge, neither can move a
    reward, and this test is about the harbor path, not about enrichment.
    """
    rows = refine(
        str(task_dir),
        iterations=1,
        summarise=False,
        judge_enabled=False,
        base_cfg_overrides=NOP_AGENT_OVERRIDE,
    )

    # -- the ledger is the loop state, so it is the primary artifact ------------
    ledger = isolated_run_root / "ledger.jsonl"
    assert ledger.is_file(), "no ledger written"
    lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one appended row, got {len(lines)}"

    row = json.loads(lines[0])
    assert row == json.loads(json.dumps(rows[-1], sort_keys=True, default=str)), \
        "the returned row and the persisted row disagree"

    assert row["iteration"] == 1
    assert row["outcome"] in KNOWN_OUTCOMES, f"unclassified outcome: {row['outcome']!r}"

    # returncode 0 is the point of the test: it means harbor ACCEPTED the config we
    # built and validated, which no mocked test can establish.
    assert row["harbor_returncode"] == 0, (
        f"harbor exited {row['harbor_returncode']} "
        f"({'our timeout' if row['harbor_returncode'] == loop.TIMEOUT_RETURNCODE else 'harbor'})"
    )

    # -- durability: measured facts hit disk before any enrichment is attempted --
    facts = isolated_run_root / "history" / "iter01_facts.json"
    assert facts.is_file(), "facts checkpoint missing"
    assert json.loads(facts.read_text())["iteration"] == 1

    # iteration 1 has no prior rows, so there is nothing to inject and no history file.
    assert not (isolated_run_root / "history" / "iter01_history.md").exists()

    # -- isolation properties ---------------------------------------------------
    cfg_path = isolated_run_root / ".cfg_iter01.json"
    assert cfg_path.is_file(), "the launched config was not recorded"
    cfg_text = cfg_path.read_text()
    cfg = json.loads(cfg_text)

    assert "export_traces" not in cfg, (
        "export_traces is a CLI flag; as a config key harbor's extra='ignore' JobConfig "
        "drops it silently and no trajectory is exported"
    )
    assert "--export-traces" not in cfg_text

    assert "track3-pipeline" not in cfg_text, (
        "the read-only production tree leaked into the launched config"
    )
    assert str(isolated_run_root) in cfg["jobs_dir"], "jobs_dir escaped the tmp run root"

    # the trial harbor produced really is under our tmp jobs_dir, not anywhere else
    if row.get("trial_dir"):
        assert Path(row["trial_dir"]).is_relative_to(isolated_run_root)


# --------------------------------------------------------------------------- #
# resumability -- NO docker, NO harbor: a pre-seeded ledger and a mocked subprocess
# --------------------------------------------------------------------------- #
#
# Deliberately separated from the test above and deliberately not a second real
# iteration. Resumption is a property of `refine`'s arithmetic over the ledger
# (`start = len(rows) + 1`), and driving it with a fake harbor tests exactly that at
# zero cost -- while a second live run would double the container time and the risk to
# anything else sharing this host, to re-prove a fact the first test already proved.
# It sits in this module, under the same module-level gate, so the default suite stays
# untouched and the two halves of the story are read together.


def _fake_harbor(calls: list):
    """Stand-in for `harbor run` that materialises a fixture trial, like the real one."""

    def _run(cmd, **kwargs):
        calls.append([str(c) for c in cmd])
        cfg = json.loads(Path(cmd[cmd.index("--config") + 1]).read_text())
        dest = Path(cfg["jobs_dir"]) / cfg["job_name"] / "trial-1"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(FIXTURE_TRIAL, dest)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


def test_refine_resumes_from_a_preseeded_ledger_and_injects_history(
    monkeypatch, task_dir, isolated_run_root
):
    """A ledger with one row makes the next call iteration 2, with history attached.

    This is what makes an interrupted campaign restartable by re-running the same
    command: there is no counter and no state file to fall out of sync with the ledger.
    """
    ledger = isolated_run_root / "ledger.jsonl"
    ledger.write_text(json.dumps({
        "iteration": 1, "reward": 0.0, "reason": "telemetry_absent",
        "outcome": "agent_abandoned_run", "n_seeds": 0,
        "job_name": "agentic_iter01", "harbor_returncode": 0,
    }) + "\n")

    calls: list = []
    monkeypatch.setattr(loop.subprocess, "run", _fake_harbor(calls))

    rows = refine(
        str(task_dir),
        iterations=1,
        summarise=False,
        judge_enabled=False,
        harbor_bin="/nonexistent/harbor",
        base_cfg_overrides=NOP_AGENT_OVERRIDE,
    )

    # resumed rather than restarted: prior row kept, new one numbered 2
    assert [r["iteration"] for r in rows] == [1, 2]
    assert len([ln for ln in ledger.read_text().splitlines() if ln.strip()]) == 2

    row = rows[-1]
    assert row["history_injected"] is True
    assert row["job_name"] == "agentic_iter02"

    history = isolated_run_root / "history" / "iter02_history.md"
    assert history.is_file(), "no history rendered for the resumed iteration"
    text = history.read_text()
    assert "Attempt 2" in text
    assert "agent_abandoned_run" in text, "the prior row is missing from the history"

    # the history file is what the agent actually receives, via the config
    cfg = json.loads((isolated_run_root / ".cfg_iter02.json").read_text())
    assert cfg["extra_instruction_paths"] == [str(history)]
    assert "export_traces" not in cfg

    # ...and --export-traces went on the command line instead
    assert "--export-traces" in calls[-1]
