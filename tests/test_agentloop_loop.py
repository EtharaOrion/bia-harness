"""The code-driven refinement loop: durability, resumability, and isolation.

These tests never launch harbor or docker. `subprocess.run` is replaced by a fake
that behaves the way harbor does -- it reads the `--config` file it was handed and
materialises a trial directory under `jobs_dir/job_name/` -- so the loop is exercised
end to end against real fixture artifacts while nothing leaves this process.

The headline property under test is DURABILITY. `judge.grade_attempt` raises
`SystemExit` at two sites by design, and `SystemExit` does not inherit from
`Exception`. In the reference pipeline that gap destroyed a completed 71-minute,
$9.17 trial: the judge 403'd, `SystemExit` unwound the loop, and the process died
BEFORE the ledger row was written. Every measured fact was lost to save a discarded
advisory verdict. So the tests below assert the opposite behaviour at three layers:
`judge_trajectory` returns an error dict, `run_iteration` still returns a full row,
and the facts checkpoint is on disk before enrichment is ever attempted.

THE JUDGE AND SUMMARISER ARE NOW OFF BY DEFAULT. They are unwired, not deleted: the
parameters and both code paths survive verbatim and every durability test above is
still exercised by passing `summarise=True, judge_enabled=True` explicitly. What
changed is the default and the CLI surface -- `--summarise`/`--judge` are opt-IN, so
a bare invocation reaches no LLM at all. `test_default_run_calls_neither_*` asserts
that directly, because a default that silently re-enables an LLM is exactly the
regression this unwiring exists to prevent.

FOUR MORE PROPERTIES are asserted below, each one a defect found in production. The
`--timeout` must actually REACH `subprocess.run`, or the TimeoutExpired handler is
unreachable code and a wedged container hangs the campaign forever. Containers that
appeared during an iteration must be removed afterwards, and containers that were
ALREADY RUNNING must not be -- `test_cleanup_removes_only_containers_that_appeared_
this_iteration` is the one that stops a cleanup from killing a concurrent campaign's
live GPU trial on the same host. A Ctrl-C must still leave a ledger row for whatever
was measured. And a second `refine` on the same task must be refused outright, because
`resolve_run_root` is a pure function of the task and two runs would otherwise share
one ledger and one iteration counter.

The reward is likewise no longer the verifier's. `read_trial` computes it from the
fixture's own loss curve via `agentloop.reward`, at FULL log density: the two seeds first
reach 3.28 at steps 3150 and 3175, so (3500-3175)/600 = 0.541666... is the number
every assertion below expects, replacing the fixture verifier's 0.5. It is NOT read
off the thinned `parent_curve`, which would say 3250 / 0.41666... and would make the
`MAX_CURVE_POINTS` display budget load-bearing on the score; `test_trial_io.py` owns
the regression tests for that.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from agentloop import judge, summariser
from agentloop.harbor_config import HarborUnavailable, build_base_cfg, resolve_run_root
from agentloop.loop import judge_trajectory, load_ledger, main, refine, run_iteration

FIXTURE_TRIAL = Path(__file__).parent / "fixtures" / "track3_trial"
TASK_DIRNAME = "2739a678-1759-516d-8ba7-1cd023267ea8"

# What the fixture trial's own full-density curve is worth: (3500-3175)/600.
FIXTURE_REWARD = 0.5416666666666666

# Enrichment is opt-in now, so the durability tests must ask for it by name.
ENRICHED = {"summarise": True, "judge_enabled": True}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def task_dir():
    from harness import HARNESS_ROOT

    d = HARNESS_ROOT / "tasks" / TASK_DIRNAME
    if not (d / "task.toml").is_file():
        pytest.skip(f"task dir not present: {d}")
    return d


@pytest.fixture
def task_slug(task_dir):
    """How this task names its own run root.

    Deliberately resolved rather than hardcoded: this task.toml declares no `uuid`,
    so `resolve_task_uuid` falls back to a slug of the task name. Pinning the
    directory name here would assert the fallback instead of the isolation property
    the test is actually about.
    """
    from harness import resolve_task_uuid

    return resolve_task_uuid(task_dir)


@pytest.fixture
def base_cfg(task_dir, tmp_path):
    """A real, harbor-valid base config whose jobs_dir is redirected into tmp."""
    cfg = build_base_cfg(task_dir, tmp_path)
    cfg["jobs_dir"] = str(tmp_path / "jobs")
    return cfg


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Hard backstop: no test in this module may reach the LLM bridge.

    The judge and the summariser both read the fixture trial happily and would then
    POST to 127.0.0.1:8765. Their entry points are stubbed with deterministic doubles,
    and `urlopen` itself is poisoned so a future test that forgets to stub one fails
    loudly instead of hanging on a retry/backoff loop.
    """
    import urllib.request

    def _boom(*a, **k):  # pragma: no cover - only runs if a stub is missing
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(
        judge,
        "grade_attempt",
        lambda *a, **k: {
            "verdicts": {"J1": {"pass": True}, "J2": {"pass": False}},
            "overall_pass": False,
            "summary": "stub verdict",
        },
    )
    monkeypatch.setattr(
        summariser,
        "summarize_iteration",
        lambda *a, **k: {"mechanism": "stub summary"},
    )


@pytest.fixture(autouse=True)
def no_real_subprocess(monkeypatch):
    """Hard backstop: no test in this module may launch a real process.

    The loop now shells out to `docker ps` around every launch and to `docker rm`
    after it, so a test that forgets to stub `subprocess.run` would interrogate this
    host's real daemon -- and could reach a real removal. Every test that needs a
    process overrides this with its own double.
    """
    from agentloop import loop

    def _boom(cmd, *a, **k):  # pragma: no cover - only runs if a stub is missing
        raise AssertionError(f"test attempted to launch a real process: {cmd}")

    monkeypatch.setattr(loop.subprocess, "run", _boom)


def fake_harbor(calls: list, *, produce_trial: bool = True, returncode: int = 0,
                docker_calls: list | None = None, ps_outputs: list | None = None):
    """Stand-in for `harbor run` that mimics the one behaviour the loop depends on.

    Like the real binary it reads the config file off its own argv rather than
    receiving state out of band, which is what lets the tests assert on what was
    actually written to disk instead of on what the loop intended to write.

    It also stands in for the `docker` binary, because the loop now snapshots the
    running `task__*` containers around each launch so it can clean up the ones it
    created. Docker invocations are recorded in `docker_calls`, NOT in `calls`, so
    every existing assertion about `calls[0]` still means "the harbor launch".
    `ps_outputs` is the stdout handed back to successive `docker ps` calls -- by
    default an empty host, so no container is ever seen to appear and nothing is
    removed.
    """
    ps = list(ps_outputs or [])

    def _run(cmd, **kwargs):
        argv = [str(c) for c in cmd]
        if argv and Path(argv[0]).name == "docker":
            if docker_calls is not None:
                docker_calls.append(argv)
            out = ps.pop(0) if (argv[1:2] == ["ps"] and ps) else ""
            return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
        calls.append({"cmd": argv, "kwargs": kwargs})
        cfg = json.loads(Path(cmd[cmd.index("--config") + 1]).read_text())
        if produce_trial:
            dest = Path(cfg["jobs_dir"]) / cfg["job_name"] / "trial-1"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(FIXTURE_TRIAL, dest)
            # copytree copies the fixture's mtime too, which is as old as the
            # checkout. `trial_io.find_trial` refuses a trial older than the launch
            # as belonging to an earlier run, so without this the fake harbor
            # "produces" a trial that the loop correctly ignores.
            os.utime(dest, None)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="")

    return _run


def _never_called(what: str):
    """A stub that turns "was silently invoked" into a failure instead of a network call."""
    def _boom(*a, **k):
        raise AssertionError(f"{what} was called despite being unwired")
    return _boom


FAKE_HARBOR_KWARGS = ("produce_trial", "returncode", "docker_calls", "ps_outputs")


def iterate(monkeypatch, i, base_cfg, history, tmp_path, task_dir, **kw):
    """Run one iteration against the fake harbor; return (row, recorded calls)."""
    from agentloop import loop

    calls: list = []
    monkeypatch.setattr(
        loop.subprocess, "run", fake_harbor(calls, **{k: kw.pop(k) for k in
                                                      FAKE_HARBOR_KWARGS
                                                      if k in kw})
    )
    row = run_iteration(
        i, base_cfg, history,
        run_root=tmp_path, task_dir=task_dir, harbor_bin="/nonexistent/harbor", **kw
    )
    return row, calls


# --------------------------------------------------------------------------- #
# load_ledger
# --------------------------------------------------------------------------- #


def test_load_ledger_missing_file_returns_empty(tmp_path):
    assert load_ledger(tmp_path / "nope.jsonl") == []


def test_load_ledger_skips_blank_and_malformed_lines_preserving_order(tmp_path):
    """A half-written final line is the normal shape of an interrupted campaign.

    The ledger is appended to after every iteration, so a kill during that write
    leaves a truncated line. Refusing to parse the file at that point would strand
    every completed iteration before it.
    """
    p = tmp_path / "ledger.jsonl"
    p.write_text(
        '{"iteration": 1, "reward": 0.5}\n'
        "\n"
        "   \n"
        "not json at all\n"
        '{"iteration": 2, "reward": 0.7}\n'
        '{"iteration": 3, "rew\n'  # truncated by a kill mid-write
    )
    rows = load_ledger(p)
    assert [r["iteration"] for r in rows] == [1, 2]


def test_load_ledger_accepts_str_path(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text('{"iteration": 1}\n')
    assert load_ledger(str(p)) == [{"iteration": 1}]


# --------------------------------------------------------------------------- #
# judge_trajectory
# --------------------------------------------------------------------------- #


def test_judge_trajectory_shapes_a_normal_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(
        judge, "grade_attempt",
        lambda *a, **k: {
            "verdicts": {"J1": {"pass": True}, "J2": {"pass": False},
                         "J3": {"pass": False}},
            "overall_pass": False,
            "summary": "one sentence",
        },
    )
    out = judge_trajectory(tmp_path)
    assert out["overall_pass"] is False
    assert out["summary"] == "one sentence"
    assert sorted(out["failed_rubrics"]) == ["J2", "J3"]


def test_judge_trajectory_tolerates_missing_verdicts_key(monkeypatch, tmp_path):
    monkeypatch.setattr(judge, "grade_attempt",
                        lambda *a, **k: {"overall_pass": None, "verdicts": None})
    assert judge_trajectory(tmp_path)["failed_rubrics"] == []


def test_judge_trajectory_catches_systemexit(monkeypatch, tmp_path):
    """SystemExit is a BaseException; `except Exception` cannot catch it.

    `grade_attempt` raises it for an unreachable bridge and for empty input. Both are
    advisory-layer failures that must never cost a measured iteration.
    """
    def _exit(*a, **k):
        raise SystemExit("judge unreachable")

    monkeypatch.setattr(judge, "grade_attempt", _exit)
    out = judge_trajectory(tmp_path)
    assert "SystemExit" in out["error"]
    assert "judge unreachable" in out["error"]
    assert "overall_pass" not in out


def test_judge_trajectory_catches_keyboardinterrupt_and_ordinary_errors(
    monkeypatch, tmp_path
):
    for exc, name in ((KeyboardInterrupt(), "KeyboardInterrupt"),
                      (RuntimeError("boom"), "RuntimeError")):
        monkeypatch.setattr(
            judge, "grade_attempt",
            lambda *a, _e=exc, **k: (_ for _ in ()).throw(_e))
        assert name in judge_trajectory(tmp_path)["error"]


def test_judge_trajectory_error_is_bounded(monkeypatch, tmp_path):
    """A judge can echo a whole HTML error page back; the ledger must stay readable."""
    monkeypatch.setattr(
        judge, "grade_attempt",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("x" * 5000)))
    assert len(judge_trajectory(tmp_path)["error"]) < 260


# --------------------------------------------------------------------------- #
# DURABILITY -- the headline
# --------------------------------------------------------------------------- #


def test_judge_systemexit_does_not_destroy_the_iteration(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """The regression that cost a completed 71-minute, $9.17 trial.

    The trial finished, was graded, and its reward was on disk. The judge then 403'd,
    SystemExit unwound the loop, and the process died before the ledger was written.
    """
    def _exit(*a, **k):
        raise SystemExit("judge unreachable")

    monkeypatch.setattr(judge, "grade_attempt", _exit)
    row, _ = iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir, **ENRICHED)

    # The measured facts survive intact...
    assert row["iteration"] == 1
    assert row["reward"] == pytest.approx(FIXTURE_REWARD)
    assert row["outcome"] == "graded_pass"
    assert row["n_seeds"] == 2
    assert row["trial_dir"]
    # ...and the judge failure is recorded as data, not raised as control flow.
    assert "SystemExit" in row["rubric_verdicts"]["error"]
    # The checkpoint was written BEFORE the judge ran, which is why it exists at all.
    facts = tmp_path / "history" / "iter01_facts.json"
    assert facts.is_file()
    assert json.loads(facts.read_text())["reward"] == pytest.approx(FIXTURE_REWARD)


def test_summariser_systemexit_degrades_rather_than_propagates(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    monkeypatch.setattr(
        summariser, "summarize_iteration",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("summariser gone")))
    row, _ = iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir, **ENRICHED)

    assert row["reward"] == pytest.approx(FIXTURE_REWARD)
    assert "summariser_failed" in row["summary"]["_degraded"]
    assert "SystemExit" in row["summary"]["_degraded"]
    assert (tmp_path / "history" / "iter01_facts.json").is_file()


def test_facts_checkpoint_precedes_enrichment(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """Assert the ORDER, not just the outcome.

    A checkpoint written after the judge would pass the two tests above by accident
    (they only fail the judge, not the process). This one proves the checkpoint is
    already durable at the moment enrichment starts, which is the property that
    survives a SIGKILL rather than a caught exception.
    """
    seen = {}

    def _record(*a, **k):
        p = tmp_path / "history" / "iter01_facts.json"
        seen["existed_when_judge_ran"] = p.is_file()
        raise SystemExit("judge unreachable")

    monkeypatch.setattr(judge, "grade_attempt", _record)
    iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir, **ENRICHED)
    assert seen["existed_when_judge_ran"] is True


def test_ledger_row_appended_when_both_judge_and_summariser_fail(
    monkeypatch, task_dir, tmp_path
):
    """Total enrichment failure still yields a durable, complete ledger row."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(
        judge, "grade_attempt",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("judge unreachable")))
    monkeypatch.setattr(
        summariser, "summarize_iteration",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("summariser gone")))
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    rows = refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor",
                  **ENRICHED)

    written = load_ledger(tmp_path / "ledger.jsonl")
    assert len(written) == 1
    assert written[0]["reward"] == pytest.approx(FIXTURE_REWARD)
    assert "SystemExit" in written[0]["rubric_verdicts"]["error"]
    assert "summariser_failed" in written[0]["summary"]["_degraded"]
    # What was returned is what was persisted. Compared key-by-key rather than with
    # `==` because JSON has no tuple: `parent_curve`'s (step, loss) pairs come back
    # as lists, which is a serialisation detail and not a discrepancy.
    assert set(written[0]) == set(rows[0])
    for key in set(written[0]) - {"parent_curve"}:
        assert written[0][key] == rows[0][key], key


# --------------------------------------------------------------------------- #
# UNWIRED BY DEFAULT -- judge and summariser are off unless asked for
# --------------------------------------------------------------------------- #


def test_default_run_iteration_calls_neither_judge_nor_summariser(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """The default path must not touch an LLM, and must still produce a full row."""
    monkeypatch.setattr(judge, "grade_attempt", _never_called("judge"))
    monkeypatch.setattr(summariser, "summarize_iteration", _never_called("summariser"))

    row, _ = iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir)

    assert "rubric_verdicts" not in row
    assert "summary" not in row
    assert row["reward"] == pytest.approx(FIXTURE_REWARD)
    assert row["outcome"] == "graded_pass"
    assert row["n_seeds"] == 2
    assert (tmp_path / "history" / "iter01_facts.json").is_file()


def test_default_refine_calls_neither_judge_nor_summariser(
    monkeypatch, task_dir, tmp_path
):
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))
    monkeypatch.setattr(judge, "grade_attempt", _never_called("judge"))
    monkeypatch.setattr(summariser, "summarize_iteration", _never_called("summariser"))

    rows = refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    assert "rubric_verdicts" not in rows[0]
    assert "summary" not in rows[0]
    assert rows[0]["reward"] == pytest.approx(FIXTURE_REWARD)


def test_signature_defaults_are_off():
    """Assert the defaults themselves, so a caller that omits them cannot be surprised."""
    import inspect

    for fn in (run_iteration, refine):
        params = inspect.signature(fn).parameters
        assert params["summarise"].default is False, fn.__name__
        assert params["judge_enabled"].default is False, fn.__name__


def test_enrichment_is_unwired_not_removed(monkeypatch, base_cfg, tmp_path, task_dir):
    """Passing True re-enables both paths verbatim; the modules stay importable."""
    row, _ = iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir, **ENRICHED)

    assert row["rubric_verdicts"]["summary"] == "stub verdict"
    assert row["rubric_verdicts"]["failed_rubrics"] == ["J2"]
    assert row["summary"] == {"mechanism": "stub summary"}


# --------------------------------------------------------------------------- #
# RESUMABILITY -- the ledger is the loop state
# --------------------------------------------------------------------------- #


def _seed_ledger(path: Path, n: int) -> None:
    with path.open("w") as f:
        for i in range(1, n + 1):
            f.write(json.dumps({
                "iteration": i, "reward": 0.4, "outcome": "graded_miss",
                "graded_step": 3400, "n_seeds": 2, "findings": f"attempt {i}",
            }) + "\n")


def test_resumes_from_ledger_length(monkeypatch, task_dir, tmp_path):
    """An interrupted campaign restarts where it stopped, with no bookkeeping flag."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    _seed_ledger(tmp_path / "ledger.jsonl", 2)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    rows = refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    assert len(rows) == 3
    assert rows[-1]["iteration"] == 3
    assert rows[-1]["job_name"].endswith("_iter03")


def test_start_at_overrides_ledger_length(monkeypatch, task_dir, tmp_path):
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    _seed_ledger(tmp_path / "ledger.jsonl", 2)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    rows = refine(str(task_dir), iterations=1, start_at=7,
                  harbor_bin="/nonexistent/harbor")

    assert rows[-1]["iteration"] == 7
    assert rows[-1]["job_name"].endswith("_iter07")


def test_prior_rows_are_rendered_into_the_next_prompt(
    monkeypatch, task_dir, tmp_path
):
    """Resuming is only useful if the recovered rows actually reach the agent."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    _seed_ledger(tmp_path / "ledger.jsonl", 2)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    hist = (tmp_path / "history" / "iter03_history.md").read_text()
    assert "Attempt 3" in hist
    assert "attempt 1" in hist and "attempt 2" in hist


# --------------------------------------------------------------------------- #
# the config the loop hands harbor
# --------------------------------------------------------------------------- #


def test_export_traces_is_a_cli_flag_and_not_a_config_key(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """Harbor's JobConfig is pydantic extra="ignore": an unknown key is DROPPED.

    Put `export_traces` in the JSON and the run looks perfectly configured while
    quietly writing no `agent/trajectory.json` -- leaving the summariser and the
    judge with nothing to read and every seed flagged verification_incomplete.
    """
    _, calls = iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir)

    argv = calls[0]["cmd"]
    assert "--export-traces" in argv
    assert argv[0] == "/nonexistent/harbor" and argv[1] == "run"

    written = json.loads((tmp_path / ".cfg_iter01.json").read_text())
    assert "export_traces" not in written
    assert not any("export" in k for k in written)


def test_harbor_output_is_captured_not_inherited(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    _, calls = iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir)
    assert calls[0]["kwargs"]["capture_output"] is True
    assert calls[0]["kwargs"]["text"] is True


def test_history_is_absent_on_first_iteration(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """Iteration 1 has nothing to say; an empty instruction file would be noise."""
    iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir)
    written = json.loads((tmp_path / ".cfg_iter01.json").read_text())
    assert "extra_instruction_paths" not in written
    assert not (tmp_path / "history" / "iter01_history.md").exists()


def test_history_is_injected_on_later_iterations(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    row, _ = iterate(monkeypatch, 2, base_cfg, "# Attempt 2\n\nprior findings",
                     tmp_path, task_dir)
    written = json.loads((tmp_path / ".cfg_iter02.json").read_text())
    paths = written["extra_instruction_paths"]
    assert len(paths) == 1
    hp = Path(paths[0])
    assert hp.is_file()
    assert hp.name == "iter02_history.md"
    assert "prior findings" in hp.read_text()
    assert row["history_injected"] is True


def test_job_name_is_suffixed_per_iteration(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    row, _ = iterate(monkeypatch, 2, base_cfg, "", tmp_path, task_dir)
    assert row["job_name"] == "agentic_iter02"
    assert json.loads((tmp_path / ".cfg_iter02.json").read_text())["job_name"] == \
        "agentic_iter02"


def test_base_cfg_is_not_mutated_across_iterations(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """The deep copy is what keeps iteration 3's job_name from becoming
    `agentic_iter01_iter02_iter03` and its history from stacking."""
    before = json.loads(json.dumps(base_cfg))
    iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir)
    iterate(monkeypatch, 2, base_cfg, "some history", tmp_path, task_dir)
    assert base_cfg == before
    assert json.loads((tmp_path / ".cfg_iter02.json").read_text())["job_name"] == \
        "agentic_iter02"


def test_invalid_cfg_is_rejected_before_harbor_is_launched(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """Validation is worthless if it happens after the container starts."""
    try:
        from agentloop.harbor_config import _load_job_config_cls

        _load_job_config_cls()
    except HarborUnavailable as exc:  # pragma: no cover - host without harbor
        pytest.skip(f"harbor not importable: {exc}")

    bad = dict(base_cfg, modle_name="typo")
    from agentloop import loop

    calls: list = []
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor(calls))
    with pytest.raises(Exception):
        run_iteration(1, bad, "", run_root=tmp_path, task_dir=task_dir,
                      harbor_bin="/nonexistent/harbor")
    assert calls == []


# --------------------------------------------------------------------------- #
# a run that produced nothing
# --------------------------------------------------------------------------- #


def test_no_trial_produced_yields_a_sparse_row(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """A crashed launch is still an iteration and still belongs in the ledger."""
    row, _ = iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir,
                     produce_trial=False, returncode=1)

    assert row["outcome"] == "harness_incomplete"
    assert row["reason"] == "no_trial_produced"
    assert row["reward"] == 0.0
    assert row["n_seeds"] == 0
    assert row["findings"] == ""
    assert row["harbor_returncode"] == 1
    assert row["iteration"] == 1
    assert row["job_name"] == "agentic_iter01"
    assert row["timestamp_utc"]
    # Nothing to enrich, so no judge/summariser keys are invented.
    assert "rubric_verdicts" not in row


def test_sparse_row_is_renderable_as_history(monkeypatch, task_dir, tmp_path):
    """The next iteration must survive a predecessor that produced no trial."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run",
                        fake_harbor([], produce_trial=False, returncode=1))
    rows = refine(str(task_dir), iterations=2, harbor_bin="/nonexistent/harbor")

    assert [r["iteration"] for r in rows] == [1, 2]
    assert (tmp_path / "history" / "iter02_history.md").is_file()


def test_harbor_timeout_is_recorded_rather_than_raised(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    from agentloop import loop

    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(loop.subprocess, "run", _timeout)
    row = run_iteration(1, base_cfg, "", run_root=tmp_path, task_dir=task_dir,
                        harbor_bin="/nonexistent/harbor", timeout=1)
    assert row["outcome"] == "harness_incomplete"
    assert row["harbor_returncode"] == 124


# --------------------------------------------------------------------------- #
# TIMEOUT -- the handler is only worth having if something can reach it
# --------------------------------------------------------------------------- #


def test_timeout_defaults_to_none_at_every_layer(monkeypatch, task_dir, tmp_path):
    """No limit unless asked for -- a legitimate multi-hour GPU job must not be killed."""
    import inspect

    from agentloop import loop

    for fn in (run_iteration, refine):
        assert inspect.signature(fn).parameters["timeout"].default is None, fn.__name__

    calls: list = []
    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor(calls))
    refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")
    assert calls[0]["kwargs"]["timeout"] is None


def test_refine_threads_the_timeout_into_subprocess(monkeypatch, task_dir, tmp_path):
    from agentloop import loop

    calls: list = []
    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor(calls))
    refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor", timeout=45)
    assert calls[0]["kwargs"]["timeout"] == 45


def test_main_timeout_flag_reaches_subprocess_run(monkeypatch, task_dir, tmp_path):
    """The whole point of DEFECT 1: `--timeout 30` must arrive at `subprocess.run`.

    Before this, `run_iteration` had the parameter and the handler but nothing on the
    path from the CLI ever passed it, so `timeout=None` was the only value harbor was
    ever launched with and the TimeoutExpired handler was unreachable code.
    """
    from agentloop import loop

    calls: list = []
    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor(calls))

    rc = main(["--task", str(task_dir), "--iterations", "1", "--timeout", "30",
               "--harbor-bin", "/nonexistent/harbor"])

    assert rc == 0
    assert calls[0]["kwargs"]["timeout"] == 30.0
    assert isinstance(calls[0]["kwargs"]["timeout"], float)


def test_timeout_flag_accepts_fractional_seconds(monkeypatch, task_dir, tmp_path):
    from agentloop import loop

    calls: list = []
    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor(calls))
    main(["--task", str(task_dir), "--iterations", "1", "--timeout", "0.5",
          "--harbor-bin", "/nonexistent/harbor"])
    assert calls[0]["kwargs"]["timeout"] == 0.5


def test_timeout_help_states_the_default_explicitly():
    """"default: None" has to be in the help text; an invisible default is a trap."""
    from agentloop import loop

    help_text = loop.build_parser().format_help()
    assert "--timeout" in help_text
    idx = help_text.rindex("--timeout")  # the options block, not the usage line
    blurb = help_text[idx:idx + 400].lower()
    assert "no limit" in blurb or "no timeout" in blurb
    assert "default" in blurb


def test_timeout_through_refine_records_124_and_does_not_propagate(
    monkeypatch, task_dir, tmp_path
):
    """A wedged container must end as a ledger row, not as a hung campaign."""
    from agentloop import loop

    def _timeout(cmd, **kwargs):
        if Path(str(cmd[0])).name == "docker":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 1)

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", _timeout)

    rows = refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor",
                  timeout=1)

    assert len(rows) == 1
    assert rows[0]["harbor_returncode"] == 124
    written = load_ledger(tmp_path / "ledger.jsonl")
    assert len(written) == 1
    assert written[0]["harbor_returncode"] == 124


# --------------------------------------------------------------------------- #
# CONTAINER CLEANUP -- harbor's containers outlive the harbor process
# --------------------------------------------------------------------------- #


def _docker_rm_args(docker_calls: list) -> list:
    """Every container id handed to a `docker rm`, flattened."""
    ids: list = []
    for argv in docker_calls:
        if "rm" in argv:
            ids.extend(a for a in argv[argv.index("rm") + 1:]
                       if not a.startswith("-"))
    return ids


def test_cleanup_removes_only_containers_that_appeared_this_iteration(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """THE production-safety property. A blanket `docker rm` of task__* would kill a
    concurrent campaign's live GPU containers on this host, so only the difference
    between the before and after snapshots may ever be removed.
    """
    docker_calls: list = []
    row, _ = iterate(
        monkeypatch, 1, base_cfg, "", tmp_path, task_dir,
        docker_calls=docker_calls,
        ps_outputs=["someone_elses_run\n",
                    "someone_elses_run\nmine_main\nmine_sidecar\n"],
    )

    removed = _docker_rm_args(docker_calls)
    assert sorted(removed) == ["mine_main", "mine_sidecar"]
    assert "someone_elses_run" not in removed
    assert all("someone_elses_run" not in argv for argv in docker_calls
               if "rm" in argv)
    assert row["reward"] == pytest.approx(FIXTURE_REWARD)


def test_cleanup_snapshots_before_and_after_the_launch(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    docker_calls: list = []
    iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir,
            docker_calls=docker_calls)

    ps = [argv for argv in docker_calls if argv[1:2] == ["ps"]]
    assert len(ps) == 2
    for argv in ps:
        assert "-q" in argv
        assert "--filter" in argv
        assert argv[argv.index("--filter") + 1] == "name=task__"


def test_cleanup_removes_nothing_when_no_container_appeared(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """An unchanged host must produce no `docker rm` at all, not `docker rm` of nothing."""
    docker_calls: list = []
    iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir,
            docker_calls=docker_calls,
            ps_outputs=["untouched_a\nuntouched_b\n", "untouched_a\nuntouched_b\n"])

    assert _docker_rm_args(docker_calls) == []


def test_cleanup_failure_does_not_lose_the_row(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """A cleanup failure costs some disk. Losing the row costs the GPU hours."""
    from agentloop import loop

    inner = fake_harbor([], ps_outputs=["", "leaked_one\n"])

    def _run(cmd, **kwargs):
        argv = [str(c) for c in cmd]
        if argv[:2] == ["docker", "rm"]:
            raise OSError("docker daemon exploded")
        return inner(cmd, **kwargs)

    monkeypatch.setattr(loop.subprocess, "run", _run)
    row = run_iteration(1, base_cfg, "", run_root=tmp_path, task_dir=task_dir,
                        harbor_bin="/nonexistent/harbor")

    assert row["reward"] == pytest.approx(FIXTURE_REWARD)
    assert row["outcome"] == "graded_pass"


def test_missing_docker_binary_is_skipped_silently(
    monkeypatch, base_cfg, tmp_path, task_dir, capsys
):
    """A host that only rehearses the loop has no docker; that is not an error."""
    from agentloop import loop

    inner = fake_harbor([])

    def _run(cmd, **kwargs):
        if Path(str(cmd[0])).name == "docker":
            raise FileNotFoundError(2, "No such file or directory: 'docker'")
        return inner(cmd, **kwargs)

    monkeypatch.setattr(loop.subprocess, "run", _run)
    row = run_iteration(1, base_cfg, "", run_root=tmp_path, task_dir=task_dir,
                        harbor_bin="/nonexistent/harbor")

    assert row["reward"] == pytest.approx(FIXTURE_REWARD)
    assert "docker" not in capsys.readouterr().err.lower()


def test_cleanup_runs_even_when_harbor_times_out(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    """The timed-out case is exactly the case that leaves a container behind."""
    from agentloop import loop

    docker_calls: list = []
    ps = ["", "wedged_main\n"]

    def _run(cmd, **kwargs):
        argv = [str(c) for c in cmd]
        if Path(argv[0]).name == "docker":
            docker_calls.append(argv)
            out = ps.pop(0) if (argv[1:2] == ["ps"] and ps) else ""
            return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(loop.subprocess, "run", _run)
    row = run_iteration(1, base_cfg, "", run_root=tmp_path, task_dir=task_dir,
                        harbor_bin="/nonexistent/harbor", timeout=1)

    assert row["harbor_returncode"] == 124
    assert _docker_rm_args(docker_calls) == ["wedged_main"]


# --------------------------------------------------------------------------- #
# CTRL-C -- an iteration that consumed GPU time must leave a trace
# --------------------------------------------------------------------------- #


def _interrupting_harbor(docker_calls=None, ps_outputs=None, produce_trial=True):
    """Harbor that does the work, then dies to a Ctrl-C before it can return."""
    inner = fake_harbor([], produce_trial=produce_trial,
                        docker_calls=docker_calls, ps_outputs=ps_outputs)

    def _run(cmd, **kwargs):
        argv = [str(c) for c in cmd]
        if Path(argv[0]).name == "docker":
            return inner(cmd, **kwargs)
        inner(cmd, **kwargs)
        raise KeyboardInterrupt("ctrl-c")

    return _run


def test_keyboardinterrupt_still_writes_the_ledger_row_then_reraises(
    monkeypatch, task_dir, tmp_path
):
    """The row is appended AFTER run_iteration returns, so a Ctrl-C used to lose a
    completed multi-hour trial outright. Whatever was measured must reach the ledger.
    """
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", _interrupting_harbor())

    with pytest.raises(KeyboardInterrupt):
        refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    written = load_ledger(tmp_path / "ledger.jsonl")
    assert len(written) == 1
    assert written[0]["iteration"] == 1
    assert written[0]["reward"] == pytest.approx(FIXTURE_REWARD)
    assert written[0]["interrupted"] is True
    assert written[0]["harbor_returncode"] == loop.INTERRUPT_RETURNCODE


def test_keyboardinterrupt_with_no_trial_still_writes_a_sparse_row(
    monkeypatch, task_dir, tmp_path
):
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run",
                        _interrupting_harbor(produce_trial=False))

    with pytest.raises(KeyboardInterrupt):
        refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    written = load_ledger(tmp_path / "ledger.jsonl")
    assert len(written) == 1
    assert written[0]["outcome"] == "harness_incomplete"
    assert written[0]["interrupted"] is True


def test_keyboardinterrupt_cleans_up_the_containers_it_started(
    monkeypatch, task_dir, tmp_path
):
    """Ctrl-C is the orphan case: harbor dies, its containers do not."""
    from agentloop import loop

    docker_calls: list = []
    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", _interrupting_harbor(
        docker_calls=docker_calls, ps_outputs=["other\n", "other\norphan\n"]))

    with pytest.raises(KeyboardInterrupt):
        refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    assert _docker_rm_args(docker_calls) == ["orphan"]


def test_keyboardinterrupt_stops_the_campaign_rather_than_continuing(
    monkeypatch, task_dir, tmp_path
):
    """Ctrl-C means stop: iteration 2 must never launch."""
    from agentloop import loop

    launches: list = []
    inner = _interrupting_harbor()

    def _run(cmd, **kwargs):
        if Path(str(cmd[0])).name != "docker":
            launches.append(cmd)
        return inner(cmd, **kwargs)

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", _run)

    with pytest.raises(KeyboardInterrupt):
        refine(str(task_dir), iterations=3, harbor_bin="/nonexistent/harbor")

    assert len(launches) == 1
    assert len(load_ledger(tmp_path / "ledger.jsonl")) == 1


# --------------------------------------------------------------------------- #
# CONCURRENCY -- one writer per run root
# --------------------------------------------------------------------------- #


@contextmanager
def hold_lock(run_root: Path):
    """Hold the run root's flock the way a second live campaign would.

    A separate open file description, so the kernel treats it as a separate holder
    even though it is the same process -- which is exactly what makes the contention
    testable without spawning anything.
    """
    run_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(run_root / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def test_second_concurrent_refine_is_refused_with_a_clear_error(
    monkeypatch, task_dir, tmp_path
):
    """`resolve_run_root` is a pure function of the task, so two runs of the same task
    share run_root, ledger.jsonl, `start = len(rows)+1`, job_name and .cfg_iterNN.json.
    Left unguarded they interleave and corrupt each other's numbering.
    """
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", _never_called("harbor"))

    with hold_lock(tmp_path), pytest.raises(loop.RunRootBusy) as exc:
        refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    message = str(exc.value)
    assert str(task_dir) in message or task_dir.name in message
    assert str(tmp_path) in message
    assert "already" in message.lower() or "another" in message.lower()


def test_a_busy_run_root_launches_nothing_and_writes_nothing(
    monkeypatch, task_dir, tmp_path
):
    """Refusal must happen before any side effect, or the guard is theatre."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", _never_called("harbor"))
    _seed_ledger(tmp_path / "ledger.jsonl", 2)

    with hold_lock(tmp_path), pytest.raises(loop.RunRootBusy):
        refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    assert len(load_ledger(tmp_path / "ledger.jsonl")) == 2
    assert not (tmp_path / ".cfg_iter03.json").exists()


def test_lock_is_released_when_refine_returns(monkeypatch, task_dir, tmp_path):
    """A lock that outlives its run turns a resumable campaign into a dead one."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    with hold_lock(tmp_path):
        pass

    rows = refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")
    assert [r["iteration"] for r in rows] == [1, 2]


def test_lock_is_released_when_the_iteration_is_interrupted(
    monkeypatch, task_dir, tmp_path
):
    """Ctrl-C must not leave the task unrunnable until someone finds the lock file."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", _interrupting_harbor())

    with pytest.raises(KeyboardInterrupt):
        refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    with hold_lock(tmp_path):
        pass


def test_lock_is_released_when_the_iteration_raises(monkeypatch, task_dir, tmp_path):
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop, "run_iteration",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    with hold_lock(tmp_path):
        pass


def test_the_lock_file_lives_in_the_run_root(monkeypatch, task_dir, tmp_path):
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    assert (tmp_path / ".lock").is_file()


def test_two_different_tasks_do_not_block_each_other(monkeypatch, task_dir, tmp_path):
    """The lock is per run root. Serialising unrelated campaigns would be a new bug."""
    from agentloop import loop

    other = tmp_path / "other-task"
    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: other)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    with hold_lock(tmp_path / "this-task"):
        rows = refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# LEDGER INTEGRITY -- a torn row must be impossible, and never silent
# --------------------------------------------------------------------------- #


def test_load_ledger_names_the_line_number_of_a_malformed_line(tmp_path, capsys):
    """Silence is the actual bug. A dropped row loses a GPU-expensive iteration AND
    renumbers every later one, because `start = len(rows)+1` counts what survived.
    """
    p = tmp_path / "ledger.jsonl"
    p.write_text(
        '{"iteration": 1, "reward": 0.5}\n'
        '{"iteration": 2, "rew\n'
        '{"iteration": 3, "reward": 0.7}\n'
    )

    rows = load_ledger(p)

    assert [r["iteration"] for r in rows] == [1, 3]
    err = capsys.readouterr().err
    assert ":2" in err
    assert "ledger.jsonl" in err


def test_load_ledger_counts_every_dropped_line(tmp_path, capsys):
    p = tmp_path / "ledger.jsonl"
    p.write_text(
        '{"iteration": 1}\n'
        "torn\n"
        '{"iteration": 2}\n'
        '{"itera\n'
    )

    rows = load_ledger(p)

    assert len(rows) == 2
    err = capsys.readouterr().err
    assert ":2" in err and ":4" in err
    assert "2" in err


def test_load_ledger_warns_about_a_line_that_is_not_an_object(tmp_path, capsys):
    """Valid JSON that is not a row is still a lost row."""
    p = tmp_path / "ledger.jsonl"
    p.write_text('{"iteration": 1}\n[1, 2, 3]\n')

    assert len(load_ledger(p)) == 1
    assert ":2" in capsys.readouterr().err


def test_load_ledger_is_silent_on_a_clean_ledger(tmp_path, capsys):
    """No crying wolf: blank lines are normal and must not produce a warning."""
    p = tmp_path / "ledger.jsonl"
    p.write_text('{"iteration": 1}\n\n{"iteration": 2}\n')

    assert len(load_ledger(p)) == 2
    assert capsys.readouterr().err == ""


def _write_spy(monkeypatch, needle: bytes):
    """Record every os.write whose payload contains `needle`, and pass it through."""
    from agentloop import loop

    seen: list = []
    real = os.write

    def _spy(fd, data):
        if needle in bytes(data):
            seen.append(bytes(data))
        return real(fd, data)

    monkeypatch.setattr(loop.os, "write", _spy)
    return seen


def test_ledger_row_is_written_in_a_single_write_call(monkeypatch, tmp_path):
    """POSIX append is atomic only up to PIPE_BUF, and only per write() call.

    A row split across two writes can interleave with a concurrent appender and tear,
    so the whole line must leave in one syscall.
    """
    from agentloop import loop

    seen = _write_spy(monkeypatch, b'"iteration"')
    row = {"iteration": 1, "parent_source": "x" * 18000, "reward": 0.5}
    loop._append_row(tmp_path / "ledger.jsonl", row)

    assert len(seen) == 1
    assert seen[0].endswith(b"\n")
    assert len(seen[0]) > 18000


def test_a_row_far_larger_than_pipe_buf_round_trips(tmp_path):
    """`parent_source` carries up to 18000 chars, well past the 4096-byte guarantee."""
    from agentloop import loop

    ledger = tmp_path / "ledger.jsonl"
    row = {"iteration": 1, "parent_source": "y" * 18000, "reward": 0.5}
    loop._append_row(ledger, row)
    loop._append_row(ledger, {"iteration": 2, "reward": 0.25})

    written = load_ledger(ledger)
    assert len(ledger.read_bytes()) > 4096
    assert [r["iteration"] for r in written] == [1, 2]
    assert written[0]["parent_source"] == "y" * 18000


def test_append_is_durable_before_it_returns(monkeypatch, tmp_path):
    """The next iteration's crash must not take the previous row's write cache with it."""
    from agentloop import loop

    fsynced: list = []
    real = os.fsync
    monkeypatch.setattr(loop.os, "fsync",
                        lambda fd: (fsynced.append(fd), real(fd))[1])

    loop._append_row(tmp_path / "ledger.jsonl", {"iteration": 1})

    assert fsynced


def test_append_appends_rather_than_truncating(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _seed_ledger(ledger, 2)
    from agentloop import loop

    loop._append_row(ledger, {"iteration": 3, "reward": 0.9})

    assert [r["iteration"] for r in load_ledger(ledger)] == [1, 2, 3]


def test_the_append_is_serialised_by_the_run_root_lock(
    monkeypatch, task_dir, tmp_path
):
    """Atomicity of one write is only half of it: the lock is what makes the writers
    a single writer. Proven from inside a live iteration, where a second campaign
    would be trying to append.
    """
    from agentloop import loop

    contended: list = []
    inner = fake_harbor([])

    def _run(cmd, **kwargs):
        if Path(str(cmd[0])).name != "docker":
            fd = os.open(tmp_path / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                contended.append("acquired")
            except OSError:
                contended.append("refused")
            finally:
                os.close(fd)
        return inner(cmd, **kwargs)

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", _run)

    refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    assert contended == ["refused"]


# --------------------------------------------------------------------------- #
# JOB PRUNING -- opt-in, because the default may never delete a GPU artifact
# --------------------------------------------------------------------------- #


def _seed_jobs(jobs_dir: Path, n: int) -> list[Path]:
    """`n` job dirs aged oldest-first, so "newest" is a fact and not a race."""
    made = []
    for i in range(1, n + 1):
        d = jobs_dir / f"agentic_iter{i:02d}" / "trial-1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text("{}")
        os.utime(d.parent, (1_000_000 + i * 60, 1_000_000 + i * 60))
        made.append(d.parent)
    return made


def _run_with_jobs(monkeypatch, task_dir, tmp_path, prior=3, **kw):
    """Resume a campaign that already has `prior` job dirs on disk, and run one more."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    _seed_ledger(tmp_path / "ledger.jsonl", prior)
    _seed_jobs(tmp_path / "jobs", prior)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))
    refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor", **kw)
    return tmp_path / "jobs"


def test_nothing_is_pruned_by_default(monkeypatch, task_dir, tmp_path):
    """Deleting a GPU trial nobody asked to delete is worse than a full disk."""
    jobs = _run_with_jobs(monkeypatch, task_dir, tmp_path)

    assert sorted(d.name for d in jobs.iterdir()) == [
        "agentic_iter01", "agentic_iter02", "agentic_iter03", "agentic_iter04"]


def test_keep_jobs_deletes_only_the_older_dirs(monkeypatch, task_dir, tmp_path):
    jobs = _run_with_jobs(monkeypatch, task_dir, tmp_path, keep_jobs=2)

    assert sorted(d.name for d in jobs.iterdir()) == [
        "agentic_iter03", "agentic_iter04"]


def test_keep_jobs_never_deletes_the_dir_just_produced(
    monkeypatch, task_dir, tmp_path
):
    """The newest artifact is the one the operator is about to look at."""
    jobs = _run_with_jobs(monkeypatch, task_dir, tmp_path, keep_jobs=1)

    assert [d.name for d in jobs.iterdir()] == ["agentic_iter04"]
    assert (jobs / "agentic_iter04" / "trial-1" / "result.json").is_file()


def test_keep_jobs_larger_than_the_backlog_deletes_nothing(
    monkeypatch, task_dir, tmp_path
):
    jobs = _run_with_jobs(monkeypatch, task_dir, tmp_path, keep_jobs=99)

    assert len(list(jobs.iterdir())) == 4


def test_pruning_failure_is_not_fatal_and_keeps_the_row(
    monkeypatch, task_dir, tmp_path
):
    """A full or read-only disk must not turn a completed iteration into an exception."""
    from agentloop import loop

    monkeypatch.setattr(loop.shutil, "rmtree",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    jobs = _run_with_jobs(monkeypatch, task_dir, tmp_path, keep_jobs=1)

    assert len(list(jobs.iterdir())) == 4
    assert len(load_ledger(tmp_path / "ledger.jsonl")) == 4


def test_an_interrupted_iteration_prunes_nothing(monkeypatch, task_dir, tmp_path):
    """Ctrl-C is not consent to delete artifacts."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    _seed_ledger(tmp_path / "ledger.jsonl", 3)
    _seed_jobs(tmp_path / "jobs", 3)
    monkeypatch.setattr(loop.subprocess, "run", _interrupting_harbor())

    with pytest.raises(KeyboardInterrupt):
        refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor",
               keep_jobs=1)

    assert len(list((tmp_path / "jobs").iterdir())) == 4


def test_main_keep_jobs_flag_prunes(monkeypatch, task_dir, tmp_path):
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    _seed_ledger(tmp_path / "ledger.jsonl", 3)
    _seed_jobs(tmp_path / "jobs", 3)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    rc = main(["--task", str(task_dir), "--iterations", "1", "--keep-jobs", "2",
               "--harbor-bin", "/nonexistent/harbor"])

    assert rc == 0
    assert sorted(d.name for d in (tmp_path / "jobs").iterdir()) == [
        "agentic_iter03", "agentic_iter04"]


def test_keep_jobs_help_states_that_the_default_deletes_nothing():
    from agentloop import loop

    help_text = loop.build_parser().format_help()
    idx = help_text.rindex("--keep-jobs")  # the options block, not the usage line
    blurb = help_text[idx:idx + 400].lower()
    assert "default" in blurb
    assert "keep" in blurb and ("all" in blurb or "nothing" in blurb)


# --------------------------------------------------------------------------- #
# ISOLATION -- never write into the read-only production tree
# --------------------------------------------------------------------------- #


def test_written_cfg_never_points_at_the_production_tree(
    monkeypatch, task_dir, task_slug, tmp_path
):
    """track3-pipeline is a 65GB read-only reference. Nothing here may write to it."""
    real_run_root = resolve_run_root(task_dir)
    cfg = build_base_cfg(task_dir, real_run_root, job_name="isolationprobe")

    from agentloop import loop

    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([], produce_trial=False))
    run_iteration(1, cfg, "", run_root=tmp_path, task_dir=task_dir,
                  harbor_bin="/nonexistent/harbor")

    raw = (tmp_path / ".cfg_iter01.json").read_text()
    written = json.loads(raw)

    assert "track3-pipeline" not in raw
    jobs_dir = Path(written["jobs_dir"])
    assert "runs" in jobs_dir.parts and "agentloop" in jobs_dir.parts
    assert jobs_dir.parts.index("runs") + 1 == jobs_dir.parts.index("agentloop")
    assert task_slug in jobs_dir.parts


def test_run_root_is_under_runs_agentloop(task_dir, task_slug):
    parts = resolve_run_root(task_dir).parts
    assert parts[-3:] == ("runs", "agentloop", task_slug)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_main_with_zero_iterations_is_a_noop(monkeypatch, task_dir, tmp_path, capsys):
    """`--iterations 0` inspects an empty ledger; it must not IndexError on rows[-1]."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)

    def _never(*a, **k):  # pragma: no cover - proves harbor is never launched
        raise AssertionError("harbor launched with --iterations 0")

    monkeypatch.setattr(loop.subprocess, "run", _never)

    assert main(["--task", str(task_dir), "--iterations", "0"]) == 0
    assert not (tmp_path / "ledger.jsonl").exists()
    assert "iterations in ledger : 0" in capsys.readouterr().out


def test_main_runs_and_reports_the_best_iteration(
    monkeypatch, task_dir, tmp_path, capsys
):
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    rc = main(["--task", str(task_dir), "--iterations", "1",
               "--harbor-bin", "/nonexistent/harbor"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "iterations in ledger : 1" in out
    assert "best" in out and f"{FIXTURE_REWARD:.4f}" in out
    assert "0.5417" in out
    assert len(load_ledger(tmp_path / "ledger.jsonl")) == 1


def test_main_default_invocation_enriches_nothing(monkeypatch, task_dir, tmp_path):
    """A bare CLI run must reach no LLM at all -- the whole point of the unwiring."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))
    monkeypatch.setattr(judge, "grade_attempt", _never_called("judge"))
    monkeypatch.setattr(summariser, "summarize_iteration", _never_called("summariser"))

    main(["--task", str(task_dir), "--iterations", "1",
          "--harbor-bin", "/nonexistent/harbor"])

    row = load_ledger(tmp_path / "ledger.jsonl")[0]
    assert "rubric_verdicts" not in row
    assert "summary" not in row
    assert row["reward"] == pytest.approx(FIXTURE_REWARD)


def test_main_opt_in_flags_enable_enrichment(monkeypatch, task_dir, tmp_path):
    """The code paths are unwired, not removed: --judge/--summarise bring them back."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    main(["--task", str(task_dir), "--iterations", "1",
          "--judge", "--summarise", "--harbor-bin", "/nonexistent/harbor"])

    row = load_ledger(tmp_path / "ledger.jsonl")[0]
    assert row["rubric_verdicts"]["summary"] == "stub verdict"
    assert row["summary"] == {"mechanism": "stub summary"}
    assert row["reward"] == pytest.approx(FIXTURE_REWARD)


def test_main_rejects_the_retired_opt_out_flags(monkeypatch, task_dir, tmp_path):
    """`--no-judge` must fail loudly, not be silently accepted as a no-op.

    Left as an ignored argument it would read as "judging is off because I asked",
    hiding the fact that it is off unconditionally.
    """
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    # argparse must reject before anything launches; this makes a regression fail
    # fast instead of blocking on a real `harbor run`.
    monkeypatch.setattr(loop.subprocess, "run", _never_called("harbor"))
    for flag in ("--no-judge", "--no-summarise"):
        with pytest.raises(SystemExit):
            main(["--task", str(task_dir), "--iterations", "1", flag])


# --------------------------------------------------------------------------- #
# agent selection -- the same OAuth bridge, driven by a different agent
# --------------------------------------------------------------------------- #


def _record_agent(monkeypatch, tmp_path):
    """Capture the agent_name build_base_cfg is actually called with."""
    from agentloop import loop

    seen: dict = {}
    real = loop.build_base_cfg

    def _spy(task_dir, run_root, **kw):
        seen.update(kw)
        return real(task_dir, run_root, **kw)

    monkeypatch.setattr(loop, "build_base_cfg", _spy)
    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))
    return seen


def test_refine_threads_the_agent_name_into_the_config(
    monkeypatch, task_dir, tmp_path
):
    seen = _record_agent(monkeypatch, tmp_path)

    refine(task_dir, iterations=1, harbor_bin="/nonexistent/harbor",
           agent_name="openhands-sdk")

    assert seen["agent_name"] == "openhands-sdk"
    cfg = json.loads((tmp_path / ".cfg_iter01.json").read_text())
    assert cfg["agents"][0]["name"] == "openhands-sdk"
    assert cfg["agents"][0]["env"]["LLM_BASE_URL"] == "http://172.17.0.1:8765"
    assert "ANTHROPIC" not in json.dumps(cfg)


def test_refine_defaults_to_claude_code(monkeypatch, task_dir, tmp_path):
    """Existing callers must keep getting today's behaviour, unchanged."""
    _record_agent(monkeypatch, tmp_path)

    refine(task_dir, iterations=1, harbor_bin="/nonexistent/harbor")

    cfg = json.loads((tmp_path / ".cfg_iter01.json").read_text())
    assert cfg["agents"][0]["name"] == "claude-code"
    assert cfg["agents"][0]["env"]["ANTHROPIC_BASE_URL"] == "http://172.17.0.1:8765"


def test_main_accepts_the_openhands_agent(monkeypatch, task_dir, tmp_path):
    seen = _record_agent(monkeypatch, tmp_path)

    rc = main(["--task", str(task_dir), "--iterations", "1",
               "--agent", "openhands-sdk", "--harbor-bin", "/nonexistent/harbor"])

    assert rc == 0
    assert seen["agent_name"] == "openhands-sdk"


def test_main_default_agent_is_claude_code(monkeypatch, task_dir, tmp_path):
    seen = _record_agent(monkeypatch, tmp_path)

    main(["--task", str(task_dir), "--iterations", "1",
          "--harbor-bin", "/nonexistent/harbor"])

    assert seen["agent_name"] == "claude-code"


def test_main_rejects_an_unknown_agent(monkeypatch, task_dir, tmp_path):
    """An unrecognised agent must exit non-zero, not fall through to a harbor launch."""
    from agentloop import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", _never_called("harbor"))
    with pytest.raises(SystemExit) as exc:
        main(["--task", str(task_dir), "--iterations", "1", "--agent", "bogus"])
    assert exc.value.code != 0


# --------------------------------------------------------------------------- #
# codex -- a third agent, on its own OpenAI-compatible bridge
# --------------------------------------------------------------------------- #

CODEX_BRIDGE = "http://172.17.0.1:8788"


def test_refine_threads_model_and_bridge_url_into_the_config(
    monkeypatch, task_dir, tmp_path
):
    seen = _record_agent(monkeypatch, tmp_path)

    refine(task_dir, iterations=1, harbor_bin="/nonexistent/harbor",
           agent_name="codex", model_name="gpt-5-codex",
           bridge_url=CODEX_BRIDGE)

    assert seen["agent_name"] == "codex"
    assert seen["model_name"] == "gpt-5-codex"
    assert seen["bridge_url"] == CODEX_BRIDGE
    cfg = json.loads((tmp_path / ".cfg_iter01.json").read_text())
    agent = cfg["agents"][0]
    assert agent["name"] == "codex"
    assert agent["model_name"] == "gpt-5-codex"
    assert agent["env"]["OPENAI_BASE_URL"] == CODEX_BRIDGE
    assert agent["env"]["OPENAI_API_KEY"]
    assert "ANTHROPIC" not in json.dumps(cfg)


def test_refine_omits_model_and_bridge_url_when_they_are_none(
    monkeypatch, task_dir, tmp_path
):
    """Unset must mean "do not pass", not "pass None". Forwarding None would
    override build_base_cfg's own defaults with a null and break every existing
    caller that relies on them."""
    seen = _record_agent(monkeypatch, tmp_path)

    refine(task_dir, iterations=1, harbor_bin="/nonexistent/harbor")

    assert "model_name" not in seen
    assert "bridge_url" not in seen
    cfg = json.loads((tmp_path / ".cfg_iter01.json").read_text())
    assert cfg["agents"][0]["model_name"] == "claude-opus-5"
    assert cfg["agents"][0]["env"]["ANTHROPIC_BASE_URL"] == "http://172.17.0.1:8765"


def test_refine_accepts_one_override_without_the_other(
    monkeypatch, task_dir, tmp_path
):
    seen = _record_agent(monkeypatch, tmp_path)

    refine(task_dir, iterations=1, harbor_bin="/nonexistent/harbor",
           model_name="claude-sonnet-4")

    assert seen["model_name"] == "claude-sonnet-4"
    assert "bridge_url" not in seen
    cfg = json.loads((tmp_path / ".cfg_iter01.json").read_text())
    assert cfg["agents"][0]["env"]["ANTHROPIC_BASE_URL"] == "http://172.17.0.1:8765"


def test_main_accepts_the_codex_agent(monkeypatch, task_dir, tmp_path):
    seen = _record_agent(monkeypatch, tmp_path)

    rc = main(["--task", str(task_dir), "--iterations", "1",
               "--agent", "codex", "--model", "gpt-5-codex",
               "--bridge-url", CODEX_BRIDGE,
               "--harbor-bin", "/nonexistent/harbor"])

    assert rc == 0
    assert seen["agent_name"] == "codex"
    assert seen["model_name"] == "gpt-5-codex"
    assert seen["bridge_url"] == CODEX_BRIDGE


def test_main_model_and_bridge_url_default_to_unset(monkeypatch, task_dir, tmp_path):
    seen = _record_agent(monkeypatch, tmp_path)

    main(["--task", str(task_dir), "--iterations", "1",
          "--harbor-bin", "/nonexistent/harbor"])

    assert "model_name" not in seen
    assert "bridge_url" not in seen


def test_model_and_bridge_url_help_state_the_unset_behaviour_and_both_ports():
    """An invisible default is a trap, and picking the wrong port silently sends
    codex at the Claude bridge -- so both must be named in --help."""
    from agentloop import loop

    help_text = loop.build_parser().format_help()
    for flag in ("--model", "--bridge-url"):
        assert flag in help_text
        blurb = help_text[help_text.rindex(flag):][:400].lower()
        assert "default" in blurb
    assert "8765" in help_text
    assert "8788" in help_text


def test_agent_help_lists_codex():
    from agentloop import loop

    help_text = loop.build_parser().format_help()
    idx = help_text.rindex("--agent")
    assert "codex" in help_text[idx:idx + 400]
