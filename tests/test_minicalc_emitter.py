"""Tests for tasks/minicalc/tests/emit_verifier_artifacts.py and the test.sh rewiring.

The emitter is bundle code: it ships inside the task's tests/ and runs in the graded
container, so it is stdlib-only and is NOT importable as a package. It is loaded here
from its path rather than added to sys.path, because `tasks/nanogpt-speedrun/tests` is
already on sys.path (see tests/conftest.py) and a second bundle's tests/ directory on
the same path is how two bundles' modules start shadowing each other.

The load-bearing assertion in this file is not about the new artifacts at all: it is
that `verifier/reward.json` and `verifier/reward_full.json` keep their exact names and
shapes. `runner/agentloop/trial_io.py` reads reward_full.json for `reason` and
`metrics.n_seeds`, and those two fields are the only source of the reason string the
classifier runs on. The delivery-format files are strictly additive.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

HARNESS_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE_TESTS = HARNESS_ROOT / "tasks" / "minicalc" / "tests"
EMITTER_PATH = BUNDLE_TESTS / "emit_verifier_artifacts.py"
TEST_SH = BUNDLE_TESTS / "test.sh"
GRADE_PY = BUNDLE_TESTS / "grade.py"


def _load_emitter():
    spec = importlib.util.spec_from_file_location(
        "minicalc_emit_verifier_artifacts", EMITTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    # No __pycache__ in the bundle: harbor copies the whole tests/ dir to /tests, and
    # importing the emitter here must not leave build droppings inside a shipped task.
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


@pytest.fixture(scope="module")
def em():
    if not EMITTER_PATH.is_file():
        pytest.fail(f"emitter not found at {EMITTER_PATH}")
    return _load_emitter()


# The reason vocabulary minicalc actually emits. Every entry is asserted below to
# appear verbatim (as a prefix) in grade.py or test.sh, so this list cannot drift into
# describing a vocabulary the bundle does not produce.
MINICALC_REASONS = {
    "graded_step=2653": 0,
    "graded_step=3499": 0,
    "verifier_produced_no_reward": 10,
    "verifier_produced_no_score": 10,
    "no_step_clears_target_loss": 20,
    "no_step_clears_baseline_reached_target_at_step_3480": 21,
    "telemetry_absent": 30,
    "need_at_least_2_seeds_got_1": 40,
}

# `verifier_produced_no_score` is the bia/track3nov spelling of the same event
# minicalc's test.sh calls `verifier_produced_no_reward` -- the verifier wrote no
# result file at all. It is carried at the same code so a run graded by either
# lineage's test.sh resolves, and it is exempted from the groundedness check below
# precisely because minicalc's own bundle does not emit it.
CROSS_BUNDLE_ALIASES = {"verifier_produced_no_score"}


# --------------------------------------------------------------------------- reasons

@pytest.mark.parametrize("reason,code", sorted(MINICALC_REASONS.items()))
def test_every_minicalc_reason_maps_to_its_code(em, reason, code):
    assert em._reason_code(reason) == code


def test_unrecognised_reason_is_99(em):
    assert em._reason_code("chain_break_at_step_7") == em.REASON_CODE_UNRECOGNISED == 99


def test_missing_reason_is_minus_one(em):
    assert em._reason_code("") == em.REASON_CODE_MISSING == -1
    assert em._reason_code(None) == -1


def test_no_reason_code_entry_is_a_prefix_of_another(em):
    """First match must be the only match, or the table's order becomes load-bearing."""
    prefixes = [p for p, _ in em.REASON_CODES]
    assert len(prefixes) == len(set(prefixes)), "duplicate prefix in REASON_CODES"
    for a in prefixes:
        for b in prefixes:
            if a is not b:
                assert not b.startswith(a), f"{a!r} is a prefix of {b!r}"


def test_reason_table_has_no_track3nov_only_families(em):
    """minicalc has no HMAC telemetry chain and no novelty corpus; codes for them would
    advertise gates this verifier does not run."""
    prefixes = [p for p, _ in em.REASON_CODES]
    for dead in ("chain_", "novelty_", "corpus_", "frozen_violation_", "fwd_bwd_"):
        assert not any(p.startswith(dead) for p in prefixes), f"{dead} has no gate here"


def test_reason_vocabulary_is_grounded_in_the_bundle():
    """Each reason this file claims minicalc emits must really appear in the bundle."""
    src = GRADE_PY.read_text() + TEST_SH.read_text()
    for reason in MINICALC_REASONS:
        if reason in CROSS_BUNDLE_ALIASES:
            continue
        stem = re.split(r"\d", reason, maxsplit=1)[0].rstrip("=_")
        assert stem in src, f"{reason!r} (stem {stem!r}) is in no bundle source file"


def test_every_bundle_reason_is_recognised(em):
    """No reason the bundle emits may fall through to 99."""
    for reason in MINICALC_REASONS:
        assert em._reason_code(reason) != em.REASON_CODE_UNRECOGNISED


# ----------------------------------------------------------------------------- build

GRADED_RECORD = {
    "reward": 1.0,
    "reason": "graded_step=2653",
    "detail": {"first_crossing": {"0": 2635, "1": 2653}, "graded_step": 2653},
    "metrics": {"n_seeds": 2},
}


def _run(tmp_path, record=None, *, test_stdout=None, seed_logs=None,
         rubrics=None) -> pathlib.Path:
    """Materialise a minimal run dir shaped like a collected minicalc trial."""
    run = tmp_path / "run"
    v = run / "verifier"
    v.mkdir(parents=True)
    if record is not None:
        (v / "grade-stdout.md").write_text(json.dumps(record, sort_keys=True) + "\n")
    if test_stdout is not None:
        (v / "test-stdout.md").write_text(test_stdout)
    if rubrics is not None:
        (run / "rubric_verdicts.json").write_text(json.dumps({"verdicts": rubrics}))
    if seed_logs:
        # Where harbor actually collects them for minicalc, three levels below
        # artifacts/ -- not the artifacts/full_seed*.log the track3nov emitter globs.
        d = run / "artifacts" / "workspace" / "submission" / "logs"
        d.mkdir(parents=True)
        for seed, text in seed_logs.items():
            (d / f"full_seed{seed}.log").write_text(text)
    return run


def _curve(points) -> str:
    return "".join(f"step:{s}/4200 val_loss:{v:.6f}\n" for s, v in points)


def test_build_reads_score_from_the_reward_key(em, tmp_path):
    numeric, full = em.build(_run(tmp_path, GRADED_RECORD))
    assert numeric["score"] == 1.0
    assert numeric["graded_score"] == 1.0
    assert numeric["reason_code"] == 0
    assert numeric["graded_step"] == 2653


def test_build_takes_the_first_json_line(em, tmp_path):
    run = _run(tmp_path, GRADED_RECORD)
    p = run / "verifier" / "grade-stdout.md"
    p.write_text("some preamble\n" + p.read_text() + '{"reward": 0.0}\n')
    numeric, _ = em.build(run)
    assert numeric["score"] == 1.0


def test_build_on_an_empty_run_is_zero_and_missing_reason(em, tmp_path):
    numeric, _ = em.build(_run(tmp_path))
    assert numeric["score"] == 0.0
    assert numeric["reason_code"] == em.REASON_CODE_MISSING
    assert "graded_step" not in numeric


def test_composite_equals_score_without_any_evidence(em, tmp_path):
    numeric, _ = em.build(_run(tmp_path, GRADED_RECORD))
    assert numeric["composite"] == 1.0


def test_composite_scales_with_the_pytest_fraction(em, tmp_path):
    run = _run(tmp_path, GRADED_RECORD,
               test_stdout="========= 1 failed, 3 passed in 0.10s =========\n")
    numeric, _ = em.build(run)
    assert numeric["pytests_passed"] == 3
    assert numeric["pytests_failed"] == 1
    assert numeric["pytests_executed"] == 4
    assert numeric["pytests_fraction"] == 0.75
    assert numeric["composite"] == 0.75


def test_composite_never_exceeds_score(em, tmp_path):
    run = _run(tmp_path, {"reward": 0.5, "reason": "graded_step=3200"},
               test_stdout="===== 2 passed in 0.1s =====\n")
    numeric, _ = em.build(run)
    assert numeric["composite"] <= numeric["score"]


def test_pytest_keys_absent_when_the_transcript_is_a_skip_notice(em, tmp_path):
    """Absent evidence must be absent keys, never zeros: a pytests_passed of 0 next to
    a pytests_executed of 0 reads as a suite that ran and asserted nothing."""
    notice = ("SKIPPED: pytest is unavailable in this image.\n"
              "Nothing was asserted here.\n")
    numeric, _ = em.build(_run(tmp_path, GRADED_RECORD, test_stdout=notice))
    assert not [k for k in numeric if k.startswith("pytests_")]
    assert numeric["composite"] == numeric["score"]


def test_pytest_keys_absent_when_there_is_no_transcript_at_all(em, tmp_path):
    numeric, _ = em.build(_run(tmp_path, GRADED_RECORD))
    assert not [k for k in numeric if k.startswith("pytests_")]


def test_rubric_keys_never_appear_because_the_judge_is_unwired(em, tmp_path):
    numeric, full = em.build(_run(tmp_path, GRADED_RECORD))
    assert not [k for k in numeric if k.startswith("rubrics_")]
    assert full["rubrics"] is None


def test_loss_keys_absent_without_seed_logs(em, tmp_path):
    numeric, full = em.build(_run(tmp_path, GRADED_RECORD))
    assert not [k for k in numeric if k.startswith("loss_")]
    assert full["loss"] is None


def test_loss_keys_come_from_the_collected_artifact_tree(em, tmp_path):
    run = _run(tmp_path, GRADED_RECORD, seed_logs={
        "0": _curve([(2600, 3.30), (2653, 3.20), (2700, 3.10)]),
        "1": _curve([(2600, 3.40), (2653, 3.24), (2700, 3.11)]),
    })
    numeric, _ = em.build(run)
    assert numeric["loss_steps"] == 2653
    assert numeric["loss_at_graded_step"] == round((3.20 + 3.24) / 2, 6)
    assert numeric["loss_per_seed_seed0"] == 3.2
    assert numeric["loss_per_seed_seed1"] == 3.24


def test_loss_falls_back_to_the_last_common_step(em, tmp_path):
    run = _run(tmp_path, {"reward": 0.0, "reason": "no_step_clears_target_loss"},
               seed_logs={"0": _curve([(10, 9.0), (20, 8.0)]),
                          "1": _curve([(10, 9.5), (20, 8.5)])})
    numeric, _ = em.build(run)
    assert numeric["loss_steps"] == 20
    assert numeric["reason_code"] == 20


def test_seed_source_override_finds_the_in_container_tree(em, tmp_path):
    """In the container the logs live under /workspace/submission, not under /logs."""
    run = _run(tmp_path, GRADED_RECORD)
    sub = tmp_path / "submission" / "logs"
    sub.mkdir(parents=True)
    (sub / "full_seed0.log").write_text(_curve([(2653, 3.1)]))
    (sub / "full_seed1.log").write_text(_curve([(2653, 3.3)]))
    numeric, _ = em.build(run, tmp_path / "submission")
    assert numeric["loss_at_graded_step"] == 3.2


# --------------------------------------------------------------- write_full_record

def test_write_full_record_is_idempotent(em, tmp_path):
    run = _run(tmp_path, GRADED_RECORD)
    v = run / "verifier"
    _, full = em.build(run)
    em.write_full_record(v, full)
    first = (v / "grade-stdout.md").read_text()
    em.write_full_record(v, full)
    second = (v / "grade-stdout.md").read_text()
    assert first == second
    assert second.count(em.FULL_RECORD_MARKER) == 1


def test_write_full_record_preserves_the_graded_line(em, tmp_path):
    run = _run(tmp_path, GRADED_RECORD)
    v = run / "verifier"
    _, full = em.build(run)
    em.write_full_record(v, full)
    head = (v / "grade-stdout.md").read_text().split(em.FULL_RECORD_MARKER)[0]
    assert json.loads(head.strip())["reason"] == "graded_step=2653"


def test_rebuilding_after_write_full_record_is_stable(em, tmp_path):
    """The appended record must not shadow the graded line on a second pass."""
    run = _run(tmp_path, GRADED_RECORD)
    n1, full = em.build(run)
    em.write_full_record(run / "verifier", full)
    n2, _ = em.build(run)
    assert n1 == n2


# ------------------------------------------------------------------------ main / io

def test_main_writes_numeric_only_score_json(em, tmp_path, monkeypatch, capsys):
    run = _run(tmp_path, GRADED_RECORD, seed_logs={
        "0": _curve([(2653, 3.1)]), "1": _curve([(2653, 3.3)])})
    monkeypatch.setattr(sys, "argv", ["emit", str(run)])
    assert em.main() == 0
    d = json.loads((run / "verifier" / "score.json").read_text())
    assert d, "score.json is empty"
    for k, v in d.items():
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f"{k}={v!r}"


def test_score_json_is_sorted_and_indented(em, tmp_path, monkeypatch, capsys):
    run = _run(tmp_path, GRADED_RECORD)
    monkeypatch.setattr(sys, "argv", ["emit", str(run)])
    em.main()
    text = (run / "verifier" / "score.json").read_text()
    assert text.startswith("{\n \"")
    assert text.endswith("}\n")
    assert list(json.loads(text)) == sorted(json.loads(text))


def test_main_accepts_a_seed_source_argument(em, tmp_path, monkeypatch, capsys):
    run = _run(tmp_path, GRADED_RECORD)
    sub = tmp_path / "submission" / "logs"
    sub.mkdir(parents=True)
    (sub / "full_seed0.log").write_text(_curve([(2653, 3.1)]))
    (sub / "full_seed1.log").write_text(_curve([(2653, 3.3)]))
    monkeypatch.setattr(sys, "argv", ["emit", str(run), str(tmp_path / "submission")])
    em.main()
    d = json.loads((run / "verifier" / "score.json").read_text())
    assert d["loss_at_graded_step"] == 3.2


def test_emitter_never_touches_the_files_the_loop_reads(em, tmp_path, monkeypatch,
                                                        capsys):
    """HARD CONSTRAINT: reward.json / reward_full.json are trial_io's inputs."""
    run = _run(tmp_path, GRADED_RECORD)
    v = run / "verifier"
    (v / "reward.json").write_text('{"reward": 1.0}')
    (v / "reward_full.json").write_text(json.dumps(GRADED_RECORD))
    before = {p.name: p.read_bytes() for p in v.glob("reward*.json")}
    monkeypatch.setattr(sys, "argv", ["emit", str(run)])
    em.main()
    after = {p.name: p.read_bytes() for p in v.glob("reward*.json")}
    assert before == after


def test_emitter_is_stdlib_only():
    """The image is bare python3: an import of anything third-party is a hard failure
    inside the container, where it would be reported as a verifier crash."""
    src = EMITTER_PATH.read_text()
    imported = set(re.findall(r"^\s*(?:import|from)\s+([\w.]+)", src, re.M))
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    assert {m.split(".")[0] for m in imported} <= allowed


def test_emitter_runs_under_bare_python(tmp_path):
    """Executes the module as a script the way test.sh does, with -I so nothing on the
    developer's sys.path can mask a missing stdlib-only guarantee."""
    run = tmp_path / "run"
    (run / "verifier").mkdir(parents=True)
    (run / "verifier" / "grade-stdout.md").write_text(json.dumps(GRADED_RECORD) + "\n")
    r = subprocess.run([sys.executable, "-I", str(EMITTER_PATH), str(run)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (run / "verifier" / "score.json").is_file()


# -------------------------------------------------------------------------- test.sh

def test_test_sh_writes_the_delivery_format_names():
    src = TEST_SH.read_text()
    for name in ("grade-stdout.md", "test-stdout.md", "score.md",
                 "emit_verifier_artifacts.py"):
        assert name in src, f"test.sh does not mention {name}"


def test_test_sh_keeps_the_files_the_loop_reads():
    src = TEST_SH.read_text()
    assert "/logs/verifier/reward.json" in src
    assert "/logs/verifier/reward_full.json" in src
    assert "grade-stdout.txt" not in src, "the .txt name was replaced by .md"


def test_test_sh_takes_the_grade_return_code_from_pipestatus():
    src = TEST_SH.read_text()
    assert "GRADE_RC=${PIPESTATUS[0]}" in src


def test_test_sh_calls_the_emitter_non_fatally():
    src = TEST_SH.read_text()
    line = next(l for l in src.splitlines() if "emit_verifier_artifacts.py" in l)
    assert "|| true" in line, "a broken emitter must never fail a legitimate grade"


def test_test_sh_probes_for_pytest_and_says_nothing_was_asserted():
    src = TEST_SH.read_text()
    assert "python3 -m pytest --version" in src
    assert "Nothing was asserted" in src


def test_bundle_ships_no_dead_pytest_file():
    """pytest is not installed in bia/minicalc:v1, so a test_output.py in the bundle
    would be permanently unexecutable code claiming to be a check."""
    assert not (BUNDLE_TESTS / "test_output.py").exists()


# ------------------------------------------------------------------- in the container

TASK_IMAGE = "bia/minicalc:v1"
TRIAL_FIXTURE_LOGS = {
    "0": "# bia/minicalc seed=0\n" + "".join(
        f"step:{s}/4200 val_loss:{v:.6f}\n" for s, v in
        [(2600, 3.40), (2635, 3.27), (2653, 3.20), (2700, 3.10)]),
    "1": "# bia/minicalc seed=1\n" + "".join(
        f"step:{s}/4200 val_loss:{v:.6f}\n" for s, v in
        [(2600, 3.50), (2635, 3.29), (2653, 3.27), (2700, 3.11)]),
}


def _docker_ok() -> bool:
    try:
        r = subprocess.run(["docker", "image", "inspect", TASK_IMAGE],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.fixture(scope="module")
def graded_in_container(tmp_path_factory):
    """Run the real verifier entrypoint in the real image and return its verifier dir.

    Unlike tests/test_agentloop_integration.py this is not env-gated, because it starts
    no harbor job, needs no GPU and no network, mounts only tmp_path plus the bundle's
    own tests/ read-only, and finishes in about a second. It auto-skips when the image
    is not on the host.
    """
    if not _docker_ok():
        pytest.skip(f"{TASK_IMAGE} not available")
    root = tmp_path_factory.mktemp("minicalc_container")
    logs, sub = root / "logs", root / "submission" / "logs"
    logs.mkdir()
    sub.mkdir(parents=True)
    for seed, text in TRIAL_FIXTURE_LOGS.items():
        (sub / f"full_seed{seed}.log").write_text(text)
    for _ in range(2):  # twice: the second run proves the emitter is idempotent
        r = subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{logs}:/logs",
             "-v", f"{root / 'submission'}:/workspace/submission",
             "-v", f"{BUNDLE_TESTS}:/tests:ro",
             "--entrypoint", "/bin/bash", TASK_IMAGE, "/tests/test.sh"],
            capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
    yield logs / "verifier"
    # The verifier runs as root, so everything under /logs comes back root-owned and
    # pytest's own tmp_path cleanup cannot remove it -- it warns and leaves the tree in
    # /tmp forever. Hand ownership back from inside the image, which is the only place
    # with the privilege to do it.
    subprocess.run(["docker", "run", "--rm", "-v", f"{logs}:/logs",
                    "--entrypoint", "/bin/bash", TASK_IMAGE,
                    "-c", f"chown -R {os.getuid()}:{os.getgid()} /logs"],
                   capture_output=True, timeout=120)


def test_container_emits_the_delivery_format_files(graded_in_container):
    for name in ("score.json", "score.md", "grade-stdout.md", "test-stdout.md"):
        assert (graded_in_container / name).is_file(), f"{name} was not emitted"


def test_container_keeps_the_files_the_loop_reads(graded_in_container):
    """HARD CONSTRAINT: trial_io reads reward_full.json for `reason` and
    `metrics.n_seeds`, and reward.json must stay numeric-only for harbor."""
    reward = json.loads((graded_in_container / "reward.json").read_text())
    assert set(reward) == {"reward"}
    assert isinstance(reward["reward"], float)

    full = json.loads((graded_in_container / "reward_full.json").read_text())
    assert full["reason"] == "graded_step=2653"
    assert full["metrics"]["n_seeds"] == 2
    assert full["reward"] == reward["reward"]


def test_container_score_json_is_numeric_only(graded_in_container):
    d = json.loads((graded_in_container / "score.json").read_text())
    for k, v in d.items():
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f"{k}={v!r}"
    assert d["reason_code"] == 0
    assert d["graded_step"] == 2653
    assert d["loss_at_graded_step"] == round((3.20 + 3.27) / 2, 6)


def test_container_score_md_is_the_bare_score(graded_in_container):
    d = json.loads((graded_in_container / "score.json").read_text())
    assert (graded_in_container / "score.md").read_text() == f"{d['score']}\n"


def test_container_test_stdout_cannot_be_mistaken_for_a_passing_suite(
        graded_in_container):
    text = (graded_in_container / "test-stdout.md").read_text()
    assert "Nothing was asserted here" in text


def test_container_full_record_appended_once(graded_in_container):
    """The fixture ran test.sh twice; a second marker would mean the record stacks."""
    text = (graded_in_container / "grade-stdout.md").read_text()
    assert text.count("--- full record ---") == 1
    assert json.loads(text.split("--- full record ---")[0].strip())["reward"] == 1.0
