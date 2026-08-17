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
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from track3 import judge, summariser
from track3.harbor_config import HarborUnavailable, build_base_cfg, resolve_run_root
from track3.loop import judge_trajectory, load_ledger, main, refine, run_iteration

FIXTURE_TRIAL = Path(__file__).parent / "fixtures" / "track3_trial"
TASK_DIRNAME = "2739a678-1759-516d-8ba7-1cd023267ea8"


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


def fake_harbor(calls: list, *, produce_trial: bool = True, returncode: int = 0):
    """Stand-in for `harbor run` that mimics the one behaviour the loop depends on.

    Like the real binary it reads the config file off its own argv rather than
    receiving state out of band, which is what lets the tests assert on what was
    actually written to disk instead of on what the loop intended to write.
    """

    def _run(cmd, **kwargs):
        calls.append({"cmd": [str(c) for c in cmd], "kwargs": kwargs})
        cfg = json.loads(Path(cmd[cmd.index("--config") + 1]).read_text())
        if produce_trial:
            dest = Path(cfg["jobs_dir"]) / cfg["job_name"] / "trial-1"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(FIXTURE_TRIAL, dest)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="")

    return _run


def iterate(monkeypatch, i, base_cfg, history, tmp_path, task_dir, **kw):
    """Run one iteration against the fake harbor; return (row, recorded calls)."""
    from track3 import loop

    calls: list = []
    monkeypatch.setattr(
        loop.subprocess, "run", fake_harbor(calls, **{k: kw.pop(k) for k in
                                                      ("produce_trial", "returncode")
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
    row, _ = iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir)

    # The measured facts survive intact...
    assert row["iteration"] == 1
    assert row["reward"] == 0.5
    assert row["outcome"] == "graded_pass"
    assert row["n_seeds"] == 2
    assert row["trial_dir"]
    # ...and the judge failure is recorded as data, not raised as control flow.
    assert "SystemExit" in row["rubric_verdicts"]["error"]
    # The checkpoint was written BEFORE the judge ran, which is why it exists at all.
    facts = tmp_path / "history" / "iter01_facts.json"
    assert facts.is_file()
    assert json.loads(facts.read_text())["reward"] == 0.5


def test_summariser_systemexit_degrades_rather_than_propagates(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    monkeypatch.setattr(
        summariser, "summarize_iteration",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("summariser gone")))
    row, _ = iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir)

    assert row["reward"] == 0.5
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
    iterate(monkeypatch, 1, base_cfg, "", tmp_path, task_dir)
    assert seen["existed_when_judge_ran"] is True


def test_ledger_row_appended_when_both_judge_and_summariser_fail(
    monkeypatch, task_dir, tmp_path
):
    """Total enrichment failure still yields a durable, complete ledger row."""
    from track3 import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(
        judge, "grade_attempt",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("judge unreachable")))
    monkeypatch.setattr(
        summariser, "summarize_iteration",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("summariser gone")))
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    rows = refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    written = load_ledger(tmp_path / "ledger.jsonl")
    assert len(written) == 1
    assert written[0]["reward"] == 0.5
    assert "SystemExit" in written[0]["rubric_verdicts"]["error"]
    assert "summariser_failed" in written[0]["summary"]["_degraded"]
    # What was returned is what was persisted. Compared key-by-key rather than with
    # `==` because JSON has no tuple: `parent_curve`'s (step, loss) pairs come back
    # as lists, which is a serialisation detail and not a discrepancy.
    assert set(written[0]) == set(rows[0])
    for key in set(written[0]) - {"parent_curve"}:
        assert written[0][key] == rows[0][key], key


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
    from track3 import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    _seed_ledger(tmp_path / "ledger.jsonl", 2)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    rows = refine(str(task_dir), iterations=1, harbor_bin="/nonexistent/harbor")

    assert len(rows) == 3
    assert rows[-1]["iteration"] == 3
    assert rows[-1]["job_name"].endswith("_iter03")


def test_start_at_overrides_ledger_length(monkeypatch, task_dir, tmp_path):
    from track3 import loop

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
    from track3 import loop

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
        from track3.harbor_config import _load_job_config_cls

        _load_job_config_cls()
    except HarborUnavailable as exc:  # pragma: no cover - host without harbor
        pytest.skip(f"harbor not importable: {exc}")

    bad = dict(base_cfg, modle_name="typo")
    from track3 import loop

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
    from track3 import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run",
                        fake_harbor([], produce_trial=False, returncode=1))
    rows = refine(str(task_dir), iterations=2, harbor_bin="/nonexistent/harbor")

    assert [r["iteration"] for r in rows] == [1, 2]
    assert (tmp_path / "history" / "iter02_history.md").is_file()


def test_harbor_timeout_is_recorded_rather_than_raised(
    monkeypatch, base_cfg, tmp_path, task_dir
):
    from track3 import loop

    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(loop.subprocess, "run", _timeout)
    row = run_iteration(1, base_cfg, "", run_root=tmp_path, task_dir=task_dir,
                        harbor_bin="/nonexistent/harbor", timeout=1)
    assert row["outcome"] == "harness_incomplete"
    assert row["harbor_returncode"] == 124


# --------------------------------------------------------------------------- #
# ISOLATION -- never write into the read-only production tree
# --------------------------------------------------------------------------- #


def test_written_cfg_never_points_at_the_production_tree(
    monkeypatch, task_dir, task_slug, tmp_path
):
    """track3-pipeline is a 65GB read-only reference. Nothing here may write to it."""
    real_run_root = resolve_run_root(task_dir)
    cfg = build_base_cfg(task_dir, real_run_root, job_name="isolationprobe")

    from track3 import loop

    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([], produce_trial=False))
    run_iteration(1, cfg, "", run_root=tmp_path, task_dir=task_dir,
                  harbor_bin="/nonexistent/harbor")

    raw = (tmp_path / ".cfg_iter01.json").read_text()
    written = json.loads(raw)

    assert "track3-pipeline" not in raw
    jobs_dir = Path(written["jobs_dir"])
    assert "runs" in jobs_dir.parts and "track3" in jobs_dir.parts
    assert jobs_dir.parts.index("runs") + 1 == jobs_dir.parts.index("track3")
    assert task_slug in jobs_dir.parts


def test_run_root_is_under_runs_track3(task_dir, task_slug):
    parts = resolve_run_root(task_dir).parts
    assert parts[-3:] == ("runs", "track3", task_slug)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_main_with_zero_iterations_is_a_noop(monkeypatch, task_dir, tmp_path, capsys):
    """`--iterations 0` inspects an empty ledger; it must not IndexError on rows[-1]."""
    from track3 import loop

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
    from track3 import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    rc = main(["--task", str(task_dir), "--iterations", "1",
               "--harbor-bin", "/nonexistent/harbor"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "iterations in ledger : 1" in out
    assert "best" in out and "0.5000" in out
    assert len(load_ledger(tmp_path / "ledger.jsonl")) == 1


def test_main_flags_disable_enrichment(monkeypatch, task_dir, tmp_path):
    from track3 import loop

    monkeypatch.setattr(loop, "resolve_run_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(loop.subprocess, "run", fake_harbor([]))

    main(["--task", str(task_dir), "--iterations", "1",
          "--no-judge", "--no-summarise", "--harbor-bin", "/nonexistent/harbor"])

    row = load_ledger(tmp_path / "ledger.jsonl")[0]
    assert "rubric_verdicts" not in row
    assert "summary" not in row
    assert row["reward"] == 0.5  # facts are recorded regardless
