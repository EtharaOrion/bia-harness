"""Tests for the agentloop campaign -> client delivery converter.

Most assertions run against a SYNTHETIC run root built by `build_run_root` below,
because the converter must be exercised on shapes the one real campaign does not
contain: a trial whose verifier wrote `score.json`/`*.md`, a config carrying an
unmasked secret, and an output tree that already exists. The real campaign under
`runs/agentloop/` is read-only and is covered by the integration test at the end,
which skips when that directory is absent.

The fixture writes BOTH verifier vocabularies because the bundle emits both
depending on when it was graded:

    "txt" style -> verifier/{reward.json,reward_full.json,grade-stdout.txt,test-stdout.txt}
    "md"  style -> verifier/{score.json,score.md,grade-stdout.md,test-stdout.md}

The txt style is what the three existing real trials have; the md style is what the
rewired `tasks/minicalc/tests/test.sh` now produces. Both must convert.

IDEMPOTENCY AND THE TIMESTAMP. `manifest.json` records the wall-clock time of the
conversion, so two runs cannot be byte-identical in that one field. `tree_checksum`
therefore drops `generated_at` from the manifest before hashing and hashes every
other byte of every other file. That is the strongest statement that still admits a
timestamp: nothing but the clock may move between runs.
"""

import collections
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS_ROOT / "tools"))

import package_delivery as pd

REAL_RUN_ROOT = HARNESS_ROOT / "runs" / "agentloop" / "17d66d37-7b3f-57d3-93ad-0263fc495147"

# A value shaped like a credential actually issued by the model provider. No test
# may let this string reach the output tree.
FAKE_SECRET = "sk-ant-api03-REALLOOKINGKEY0000000000000000000000000000AA"


# --------------------------------------------------------------------------
# fixture construction
# --------------------------------------------------------------------------

def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_json(path, obj):
    _write(path, json.dumps(obj, indent=4))


def build_task_dir(root):
    """A minimal harbor task bundle: the five paths the delivery root carries."""
    task = root / "task_bundle"
    _write(task / "task.toml", 'schema_version = "1.4"\n\n[task]\nname = "bia/minicalc"\n')
    _write(task / "instruction.md", "# Write an optimizer\n")
    _write(task / "environment" / "Dockerfile", "FROM python:3.11-slim\n")
    _write(task / "solution" / "reference_optimizer.py", "def build_optimizer(dim):\n    return None\n")
    _write(task / "tests" / "grade.py", "print('graded')\n")
    return task


def _model_for(model_name, i):
    """The model that ran iteration `i`.

    A campaign is no longer single-model: `model_name` may be one name for the
    whole run root, or a per-iteration mapping/sequence when iterations were run
    by different agents.
    """
    if isinstance(model_name, dict):
        return model_name[i]
    if isinstance(model_name, (list, tuple)):
        return model_name[i - 1]
    return model_name


def build_run_root(root, n_iters=3, verifier_style="txt", secret=FAKE_SECRET,
                   task_dir=None, model_name="claude-opus-5", job_name="agentic"):
    """Build a synthetic agentloop run root with `n_iters` completed iterations.

    `model_name` is one name, or a `{iteration: name}` mapping / per-iteration
    sequence for a campaign whose iterations were run by different models.

    `job_name` is the campaign's configured job name, which loop.py extends into
    `<job_name>_iterNN` per iteration. It defaults to harbor_config's own default
    so every caller that does not care keeps the shape it had.
    """
    task_dir = task_dir or build_task_dir(root)
    run_root = root / "runs" / "agentloop" / "17d66d37-7b3f-57d3-93ad-0263fc495147"
    trial_ids = {}
    ledger_rows = []

    for i in range(1, n_iters + 1):
        job = f"{job_name}_iter{i:02d}"
        trial_id = f"minicalc__TRIAL{i:02d}"
        trial_ids[i] = trial_id
        job_dir = run_root / "jobs" / job
        trial = job_dir / trial_id

        history_path = run_root / "history" / f"iter{i:02d}_history.md"
        if i > 1:
            _write(history_path, f"# Attempt {i} - your previous attempts at this task\n")
        _write_json(run_root / "history" / f"iter{i:02d}_facts.json", {"iteration": i})

        model = _model_for(model_name, i)
        env = {"ANTHROPIC_BASE_URL": "http://172.17.0.1:8765", "ANTHROPIC_API_KEY": secret}
        agent = {"name": "claude-code", "model_name": model, "env": dict(env)}

        base_cfg = {
            "job_name": job,
            "jobs_dir": str(run_root / "jobs"),
            "agent_setup_timeout_multiplier": 5.0,
            "n_concurrent_trials": 1,
            "agents": [dict(agent)],
            "tasks": [{"path": str(task_dir)}],
        }
        if i > 1:
            base_cfg["extra_instruction_paths"] = [str(history_path)]
        _write_json(run_root / f".cfg_iter{i:02d}.json", base_cfg)
        _write_json(job_dir / "config.json", base_cfg)
        _write(job_dir / "job.log", "job log\n")
        _write_json(job_dir / "lock.json", {"env": dict(env)})
        _write_json(job_dir / "result.json", {"job_name": job})

        trial_cfg = {
            "task": {"path": str(task_dir)},
            "trial_name": trial_id,
            "trials_dir": str(job_dir),
            "install_only": False,
            "timeout_multiplier": 1.0,
            "agent_setup_timeout_multiplier": 5.0,
            "agent": dict(agent),
            "environment": {"type": "docker", "delete": True},
            "verifier": {"disable": False},
            "artifacts": [],
            "job_id": f"0000000{i}-eead-4e89-b7b5-01d5cfc25b54",
            "source_trial": None,
        }
        if i > 1:
            trial_cfg["extra_instruction_paths"] = [str(history_path)]
        _write_json(trial / "config.json", trial_cfg)

        # agent/ -- one keeper, three droppers
        _write(trial / "agent" / "trajectory.json", json.dumps({"steps": [f"iter{i}"]}))
        _write(trial / "agent" / "claude-code.txt", "raw agent stdout\n")
        _write(trial / "agent" / "sessions" / "session-1.jsonl", '{"x": 1}\n')
        _write(trial / "agent" / "setup" / "setup.log", "setup\n")

        # artifacts/ -- one keeper, nested; the rest dropped
        _write(trial / "artifacts" / "workspace" / "submission" / "optimizer.py",
               f"# optimizer for iteration {i}\ndef build_optimizer(dim):\n    return None\n")
        _write(trial / "artifacts" / "workspace" / "submission" / "logs" / "full_seed0.log", "step:1/10 val_loss:9.0\n")
        _write(trial / "artifacts" / "workspace" / "submission" / "logs" / "full_seed1.log", "step:1/10 val_loss:9.1\n")
        _write_json(trial / "artifacts" / "manifest.json", {"artifacts": []})
        _write(trial / "artifacts" / "logs" / "verifier" / "reward.json", '{"reward": 1.0}')

        reward = 1.0 - 0.1 * i
        full = {"detail": {"graded_step": 2600 + i}, "metrics": {"n_seeds": 2},
                "reason": f"graded_step={2600 + i}", "reward": reward}
        if verifier_style == "txt":
            _write(trial / "verifier" / "reward.json", json.dumps({"reward": reward}))
            _write_json(trial / "verifier" / "reward_full.json", full)
            _write(trial / "verifier" / "grade-stdout.txt", json.dumps(full) + "\n")
            _write(trial / "verifier" / "test-stdout.txt", json.dumps(full) + "\n")
            verifier_result = {"rewards": {"reward": reward}}
        else:
            _write(trial / "verifier" / "score.json", json.dumps({"score": reward, "graded_step": 2600 + i}))
            _write(trial / "verifier" / "score.md", f"{reward}\n")
            _write(trial / "verifier" / "grade-stdout.md", json.dumps(full) + "\n")
            _write(trial / "verifier" / "test-stdout.md", "# pytest\ncollected 3 items\n")
            verifier_result = {"scores": {"score": reward}}

        _write_json(trial / "result.json", {
            "id": f"result-id-{i}",
            "task_name": "bia/minicalc",
            "trial_name": trial_id,
            "started_at": f"2026-08-18T1{i}:00:00.000000Z",
            "finished_at": f"2026-08-18T1{i}:30:45.123456Z",
            "config": trial_cfg,
            "agent_info": {"name": "claude-code", "version": "2.1.234",
                           "model_info": {"name": model, "provider": "anthropic"}},
            "agent_result": {"cost_usd": 1.0 * i},
            "verifier_result": verifier_result,
            "exception_info": None,
        })
        _write_json(trial / "lock.json", {"env": dict(env)})
        _write(trial / "trial.log", "trial log\n")

        ledger_rows.append({"iteration": i, "job_name": job, "reward": reward,
                            "trial_dir": str(trial), "outcome": "graded_pass"})

    _write(run_root / "ledger.jsonl", "".join(json.dumps(r) + "\n" for r in ledger_rows))
    (run_root / ".lock").write_text("")
    return run_root, task_dir, trial_ids


def tree_checksum(root):
    """sha256 over every relative path and its bytes, with the manifest's clock removed."""
    h = hashlib.sha256()
    root = Path(root)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode() + b"\0")
        if rel == "manifest.json":
            obj = json.loads(path.read_text())
            obj.pop("generated_at", None)
            h.update(json.dumps(obj, sort_keys=True).encode())
        else:
            h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def rel_files(root):
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def convert_bundle(run_root, root, **kw):
    """convert() and return the bundle dir, which is nested under the task uuid.

    Tests assert on bundle contents, so returning the delivery root instead
    would put every path one level too high.
    """
    pd.convert(run_root, root, **kw)
    return Path(root) / Path(run_root).name


@pytest.fixture
def converted(tmp_path):
    """The `out` slot is this task's bundle dir, not the delivery root.

    convert() nests under the task uuid so one root holds many tasks, so the
    bundle -- not its parent -- is what assertions about bundle contents mean.
    """
    run_root, task_dir, trial_ids = build_run_root(tmp_path)
    root = tmp_path / "delivery"
    manifest = pd.convert(run_root, root)
    return run_root, task_dir, trial_ids, root / run_root.name, manifest


# --------------------------------------------------------------------------
# mapping
# --------------------------------------------------------------------------

def test_iter_to_run_mapping(converted):
    """agentic_iterNN becomes run_N under trajectories/<model-slug>/."""
    _, _, trial_ids, out, manifest = converted
    traj = out / "trajectories" / "claude-opus-5"
    assert traj.is_dir()
    assert sorted(p.name for p in traj.iterdir()) == ["run_1", "run_2", "run_3"]
    by_run = {r["run"]: r for r in manifest["runs"]}
    for i in (1, 2, 3):
        assert by_run[f"run_{i}"]["source_job"] == f"agentic_iter{i:02d}"
        assert by_run[f"run_{i}"]["source_trial_dir"].endswith(trial_ids[i])


def test_model_slug_is_derived_not_hardcoded(tmp_path):
    """The slug comes from result.json agent_info.model_info.name."""
    run_root, _, _ = build_run_root(tmp_path, n_iters=1, model_name="Claude Opus 4.5 (preview)")
    manifest = pd.convert(run_root, tmp_path / "delivery")
    out = tmp_path / "delivery" / run_root.name
    assert manifest["model_slugs"] == ["claude-opus-4-5-preview"]
    assert manifest["runs"][0]["model_slug"] == "claude-opus-4-5-preview"
    assert manifest["runs"][0]["model_name"] == "Claude Opus 4.5 (preview)"
    assert (out / "trajectories" / "claude-opus-4-5-preview" / "run_1").is_dir()


# --------------------------------------------------------------------------
# which job dirs are iterations
# --------------------------------------------------------------------------
#
# A job dir is named `<job_name>_iterNN`, and `job_name` is a configurable field
# of the harbor config (harbor_config.build_base_cfg defaults it to "agentic";
# loop.py appends `_iterNN`). So the prefix is campaign data, not a constant.
#
# The anchor after the digits is load-bearing rather than decorative. A campaign
# may hold a `<job_name>_iterNN-retest` dir next to its rollout iterations: that
# is an oracle-replay job whose task bundle ships the answer under `solution/`,
# and delivering it would put the answer in a client bundle. `_iterNN` must end
# at the digits.

JOB_NAMES_ACCEPTED = [
    ("agentic_iter01", 1),          # the default, and the one real campaign on disk
    ("agentic-gpt_iter01", 1),
    ("agentic-gpt_iter02", 2),
    ("agentic_gpt_iter03", 3),      # a job_name may itself carry an underscore
    ("agentic.v2_iter10", 10),
]

JOB_NAMES_REJECTED = [
    "agentic-gpt_iter01-retest",    # oracle replay -- must never reach a bundle
    "agentic_iter01-retest",
    "agentic-gpt_iter01.bak",
    "agentic-gpt_iter",
    "agentic-gpt",
    "history",
]


@pytest.mark.parametrize("name,number", JOB_NAMES_ACCEPTED)
def test_iteration_job_name_accepted(tmp_path, name, number):
    (tmp_path / name).mkdir()
    assert pd._iteration_jobs(tmp_path) == [(number, tmp_path / name)]


@pytest.mark.parametrize("name", JOB_NAMES_REJECTED)
def test_iteration_job_name_rejected(tmp_path, name):
    (tmp_path / name).mkdir()
    assert pd._iteration_jobs(tmp_path) == []


def test_campaign_with_another_job_name_converts(tmp_path):
    """The prefix is read from the dir name, not assumed to be `agentic`."""
    run_root, _, trial_ids = build_run_root(tmp_path, n_iters=2, job_name="agentic-gpt")
    manifest = pd.convert(run_root, tmp_path / "delivery")
    assert [(r["iteration"], r["source_job"], r["run"]) for r in manifest["runs"]] == [
        (1, "agentic-gpt_iter01", "run_1"),
        (2, "agentic-gpt_iter02", "run_2"),
    ]
    out = tmp_path / "delivery" / run_root.name
    assert sorted(p.name for p in (out / "trajectories" / "claude-opus-5").iterdir()) == [
        "run_1", "run_2"]


def test_retest_job_dir_is_not_delivered(tmp_path):
    """A `-retest` sibling of a real iteration is left out of the bundle.

    It is an oracle-replay job: its task bundle ships `solution/`, so converting
    it would deliver the answer alongside the rollout runs it sits next to.
    """
    run_root, _, _ = build_run_root(tmp_path, n_iters=2, job_name="agentic-gpt")
    jobs_dir = run_root / "jobs"
    retest = jobs_dir / "agentic-gpt_iter01-retest"
    shutil.copytree(jobs_dir / "agentic-gpt_iter01", retest)
    manifest = pd.convert(run_root, tmp_path / "delivery")

    assert [r["source_job"] for r in manifest["runs"]] == [
        "agentic-gpt_iter01", "agentic-gpt_iter02"]
    assert not any(Path(r["source_trial_dir"]).is_relative_to(retest)
                   for r in manifest["runs"])
    assert retest.name not in [s["source_job"] for s in manifest["skipped"]]
    assert len(manifest["runs"]) == 2


def test_run_root_with_only_a_retest_job_is_an_error(tmp_path):
    """Nothing convertible is left once the retest is excluded, and that is loud."""
    run_root, _, _ = build_run_root(tmp_path, n_iters=1, job_name="agentic-gpt")
    jobs_dir = run_root / "jobs"
    (jobs_dir / "agentic-gpt_iter01").rename(jobs_dir / "agentic-gpt_iter01-retest")
    with pytest.raises(pd.DeliveryError, match="no convertible iteration"):
        pd.convert(run_root, tmp_path / "delivery")


# --------------------------------------------------------------------------
# several models in one campaign
# --------------------------------------------------------------------------
#
# A campaign may switch agents mid-way: the real run root has agentic_iter01..04
# under claude-opus-5 and agentic_iter05 under an OpenAI codex agent. One bundle
# holds both, one `trajectories/<slug>/` per model, and run numbers restart at 1
# inside each of them -- so `run_N` no longer names the iteration and the manifest
# row is the only thing that maps a delivered run back to its source job.

MIXED_MODELS = {1: "claude-opus-5", 2: "claude-opus-5", 3: "claude-opus-5",
                4: "claude-opus-5", 5: "gpt-5.6-sol"}
GPT_SLUG = "gpt-5-6-sol"


@pytest.fixture
def multi_model(tmp_path):
    run_root, _, trial_ids = build_run_root(tmp_path, n_iters=5, model_name=MIXED_MODELS)
    root = tmp_path / "delivery"
    manifest = pd.convert(run_root, root)
    return run_root, trial_ids, root / run_root.name, manifest


def test_two_models_get_one_directory_each(multi_model):
    """Disagreement about the model is grouping, not a refusal."""
    _, _, out, _ = multi_model
    traj = out / "trajectories"
    assert sorted(p.name for p in traj.iterdir()) == ["claude-opus-5", GPT_SLUG]
    assert sorted(p.name for p in (traj / "claude-opus-5").iterdir()) == [
        "run_1", "run_2", "run_3", "run_4"]
    assert sorted(p.name for p in (traj / GPT_SLUG).iterdir()) == ["run_1"]


def test_run_numbers_restart_inside_each_model(multi_model):
    """Numbering is per model, assigned in ledger-iteration order."""
    _, _, _, manifest = multi_model
    assert [(r["model_slug"], r["run"], r["source_job"]) for r in manifest["runs"]] == [
        ("claude-opus-5", "run_1", "agentic_iter01"),
        ("claude-opus-5", "run_2", "agentic_iter02"),
        ("claude-opus-5", "run_3", "agentic_iter03"),
        ("claude-opus-5", "run_4", "agentic_iter04"),
        (GPT_SLUG, "run_1", "agentic_iter05"),
    ]


def test_manifest_names_every_model_present(multi_model):
    """`model_slug` described one model; a multi-model bundle needs the whole list."""
    _, _, out, manifest = multi_model
    assert "model_slug" not in manifest
    assert manifest["model_slugs"] == ["claude-opus-5", GPT_SLUG]
    # self-consistent: exactly the directories on disk, and exactly what the rows say
    assert manifest["model_slugs"] == sorted(
        p.name for p in (out / "trajectories").iterdir())
    assert set(manifest["model_slugs"]) == {r["model_slug"] for r in manifest["runs"]}


def test_run_row_traces_a_delivered_run_back_to_its_job(multi_model):
    """run_N no longer implies agentic_iterNN, so the row must carry the mapping."""
    _, trial_ids, out, manifest = multi_model
    rows = {(r["model_slug"], r["run"]): r for r in manifest["runs"]}
    assert len(rows) == len(manifest["runs"])
    gpt = rows[(GPT_SLUG, "run_1")]
    assert gpt["source_job"] == "agentic_iter05"
    assert gpt["source_trial_dir"].endswith(trial_ids[5])
    assert gpt["destination"] == f"trajectories/{GPT_SLUG}/run_1"
    for row in manifest["runs"]:
        assert row["destination"] == f"trajectories/{row['model_slug']}/{row['run']}"
        assert (out / row["destination"]).is_dir()
        assert Path(row["source_trial_dir"]).is_dir()
        assert row["source_result_sha256"] == hashlib.sha256(
            (Path(row["source_trial_dir"]) / "result.json").read_bytes()).hexdigest()
    # one delivered run per source job, no job delivered twice
    jobs = [r["source_job"] for r in manifest["runs"]]
    assert sorted(jobs) == sorted(set(jobs))


@pytest.mark.parametrize("section", ["renamed", "fallbacks", "headers_added"])
def test_transform_rows_say_which_model_they_belong_to(multi_model, section):
    """`run_1` alone is ambiguous once two models each have one."""
    _, _, _, manifest = multi_model
    rows = manifest[section]
    assert rows
    for row in rows:
        assert row["model_slug"] in {"claude-opus-5", GPT_SLUG}
    pairs = {(r["model_slug"], r["run"]) for r in rows}
    assert ("claude-opus-5", "run_1") in pairs
    assert (GPT_SLUG, "run_1") in pairs


def test_per_model_slugification_is_unchanged(tmp_path):
    """Each name is slugified exactly as a single-model bundle would slugify it."""
    run_root, _, _ = build_run_root(
        tmp_path, n_iters=2,
        model_name={1: "Claude Opus 4.5 (preview)", 2: "GPT-5.6 Sol"})
    manifest = pd.convert(run_root, tmp_path / "delivery")
    out = tmp_path / "delivery" / run_root.name
    assert manifest["model_slugs"] == ["claude-opus-4-5-preview", "gpt-5-6-sol"]
    assert (out / "trajectories" / "claude-opus-4-5-preview" / "run_1").is_dir()
    assert (out / "trajectories" / "gpt-5-6-sol" / "run_1").is_dir()


def test_each_model_run_carries_its_own_iterations_content(multi_model):
    """The renumbering must not cross-wire run bodies between models."""
    _, _, out, _ = multi_model
    gpt = out / "trajectories" / GPT_SLUG / "run_1"
    assert "iteration 5" in (gpt / "artifacts" / "optimizer.py").read_text()
    assert json.loads((gpt / "agent" / "trajectory.json").read_text()) == {"steps": ["iter5"]}
    # iteration 5 has an injected history; the first iteration of the campaign does not
    assert (gpt / "agent" / "history.md").is_file()
    assert not (out / "trajectories" / "claude-opus-5" / "run_1"
                / "agent" / "history.md").exists()
    header = (gpt / "verifier" / "test-stdout.md").read_text().split("\n")[0]
    assert f"  target: trajectories/{GPT_SLUG}/run_1" in header
    # run_4 of the claude group is iteration 4, not iteration 5
    assert "iteration 4" in (out / "trajectories" / "claude-opus-5" / "run_4"
                             / "artifacts" / "optimizer.py").read_text()


def test_multi_model_bundle_is_idempotent(tmp_path):
    run_root, _, _ = build_run_root(tmp_path, n_iters=5, model_name=MIXED_MODELS)
    root = tmp_path / "delivery"
    out = convert_bundle(run_root, root)
    first = tree_checksum(out)
    files_first = rel_files(out)
    pd.convert(run_root, root, force=True)
    assert tree_checksum(out) == first
    assert rel_files(out) == files_first


def test_multi_model_secret_never_reaches_the_output(tmp_path):
    run_root, _, _ = build_run_root(tmp_path, n_iters=5, model_name=MIXED_MODELS)
    out = convert_bundle(run_root, tmp_path / "delivery")
    for path in Path(out).rglob("*"):
        if path.is_file():
            assert FAKE_SECRET not in path.read_bytes().decode("utf-8", "replace"), path


def test_cli_reports_both_model_directories(tmp_path):
    run_root, _, _ = build_run_root(tmp_path, n_iters=5, model_name=MIXED_MODELS)
    r = _cli("--run-root", run_root, "--out", tmp_path / "d", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "trajectories/claude-opus-5/run_4" in r.stdout
    assert f"trajectories/{GPT_SLUG}/run_1" in r.stdout


def test_task_bundle_copied_to_delivery_root(converted):
    _, _, _, out, _ = converted
    for rel in ("task.toml", "instruction.md", "environment/Dockerfile",
                "solution/reference_optimizer.py", "tests/grade.py"):
        assert (out / rel).is_file(), rel


# --------------------------------------------------------------------------
# per-run file set
# --------------------------------------------------------------------------

def test_agent_dir_pruned(converted):
    """run_1 keeps only trajectory.json; later runs add the injected history."""
    _, _, _, out, _ = converted
    traj = out / "trajectories" / "claude-opus-5"
    assert rel_files(traj / "run_1" / "agent") == ["trajectory.json"]
    assert rel_files(traj / "run_2" / "agent") == ["history.md", "trajectory.json"]
    assert rel_files(traj / "run_3" / "agent") == ["history.md", "trajectory.json"]


def test_optimizer_flattened(converted):
    _, _, _, out, _ = converted
    for i in (1, 2, 3):
        run = out / "trajectories" / "claude-opus-5" / f"run_{i}"
        assert rel_files(run / "artifacts") == ["optimizer.py"]
        assert f"iteration {i}" in (run / "artifacts" / "optimizer.py").read_text()


def test_dropped_sources_are_absent(converted):
    _, _, _, out, manifest = converted
    everything = rel_files(out)
    for banned in ("claude-code.txt", "sessions", "setup", "lock.json", "trial.log",
                   "manifest.json/", "workspace", "full_seed0.log", "reward_full.json",
                   "job.log", "ledger.jsonl"):
        assert not any(banned in f for f in everything), banned
    # the drops are declared, not silent
    dropped = set(manifest["dropped"])
    for name in ("agent/claude-code.txt", "agent/sessions/", "agent/setup/",
                 "artifacts/manifest.json", "artifacts/logs/", "lock.json", "trial.log"):
        assert name in dropped, name


def test_run_file_set_matches_delivery_format(converted):
    _, _, _, out, _ = converted
    run = out / "trajectories" / "claude-opus-5" / "run_3"
    assert rel_files(run) == [
        "agent/history.md",
        "agent/trajectory.json",
        "artifacts/optimizer.py",
        "config.json",
        "result.json",
        "verifier/grade-stdout.md",
        "verifier/score.json",
        "verifier/score.md",
        "verifier/test-stdout.md",
    ]


def test_rubric_verdicts_not_emitted(converted):
    """The LLM judge is unwired in this harness, so there is nothing to write."""
    _, _, _, out, manifest = converted
    assert not any("rubric_verdicts" in f for f in rel_files(out))
    assert any("rubric_verdicts.json" in n for n in manifest["not_produced"])


# --------------------------------------------------------------------------
# config rewrite
# --------------------------------------------------------------------------

def test_config_pruned_to_delivery_keys(converted):
    _, _, _, out, _ = converted
    cfg = json.loads((out / "trajectories" / "claude-opus-5" / "run_3" / "config.json").read_text())
    assert set(cfg) == {"task", "trial_name", "trials_dir", "agent_setup_timeout_multiplier",
                        "agent", "extra_instruction_paths", "job_id"}
    assert set(cfg["agent"]) == {"name", "model_name", "env"}
    assert cfg["task"] == {"path": pytest.approx(cfg["task"]["path"])} or set(cfg["task"]) == {"path"}
    assert "install_only" not in cfg
    assert "environment" not in cfg


def test_run_1_config_has_no_extra_instruction_paths(converted):
    _, _, _, out, _ = converted
    cfg = json.loads((out / "trajectories" / "claude-opus-5" / "run_1" / "config.json").read_text())
    assert "extra_instruction_paths" not in cfg


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------

def test_secret_redacted_in_config_and_result(converted):
    _, _, _, out, manifest = converted
    for i in (1, 2, 3):
        run = out / "trajectories" / "claude-opus-5" / f"run_{i}"
        cfg = json.loads((run / "config.json").read_text())
        assert cfg["agent"]["env"]["ANTHROPIC_API_KEY"] == "${ANTHROPIC_API_KEY}"
        assert cfg["agent"]["env"]["ANTHROPIC_BASE_URL"] == "http://172.17.0.1:8765"
        res = json.loads((run / "result.json").read_text())
        assert res["config"]["agent"]["env"]["ANTHROPIC_API_KEY"] == "${ANTHROPIC_API_KEY}"
    assert any(r["env_var"] == "ANTHROPIC_API_KEY" for r in manifest["redacted"])


def test_secret_value_absent_from_entire_output(converted):
    _, _, _, out, _ = converted
    for path in Path(out).rglob("*"):
        if path.is_file():
            assert FAKE_SECRET not in path.read_bytes().decode("utf-8", "replace"), path


@pytest.mark.parametrize("name", [
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "GITHUB_TOKEN",
    "CLIENT_SECRET", "DB_PASSWORD", "PASSWORD", "AWS_SECRET_ACCESS_KEY",
])
def test_secret_name_shapes_detected(name):
    assert pd.is_secret_name(name)


@pytest.mark.parametrize("name", ["ANTHROPIC_BASE_URL", "PATH", "MODEL_NAME", "KEYBOARD_LAYOUT"])
def test_non_secret_names_not_redacted(name):
    assert not pd.is_secret_name(name)


def test_fail_loudly_on_unredacted_secret(tmp_path):
    """The output audit must raise, not warn, if a secret survives."""
    out = tmp_path / "out"
    (out / "trajectories").mkdir(parents=True)
    (out / "trajectories" / "config.json").write_text(
        json.dumps({"agent": {"env": {"ANTHROPIC_API_KEY": FAKE_SECRET}}}))
    with pytest.raises(pd.SecretLeakError) as e:
        pd.audit_output(out, forbidden_values={FAKE_SECRET})
    assert "ANTHROPIC_API_KEY" in str(e.value)


def test_audit_detects_leaked_literal_in_any_file(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "trajectory.json").write_text(json.dumps({"text": f"my key is {FAKE_SECRET}"}))
    with pytest.raises(pd.SecretLeakError):
        pd.audit_output(out, forbidden_values={FAKE_SECRET})


def test_audit_accepts_templatized_placeholder(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text(
        json.dumps({"agent": {"env": {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}}}))
    pd.audit_output(out, forbidden_values={FAKE_SECRET})


def test_convert_raises_when_audit_would_fail(tmp_path, monkeypatch):
    """A redaction gap must abort the conversion rather than ship the tree."""
    run_root, _, _ = build_run_root(tmp_path, n_iters=1)
    monkeypatch.setattr(pd, "redact_secrets", lambda obj, out: (obj, []))
    with pytest.raises(pd.SecretLeakError):
        pd.convert(run_root, tmp_path / "delivery")


# --------------------------------------------------------------------------
# rewards -> scores normalization
# --------------------------------------------------------------------------

def test_rewards_normalized_to_scores_and_recorded(converted):
    _, _, _, out, manifest = converted
    res = json.loads((out / "trajectories" / "claude-opus-5" / "run_3" / "result.json").read_text())
    assert "scores" in res["verifier_result"]
    assert "rewards" not in res["verifier_result"]
    assert res["verifier_result"]["scores"] == {"score": pytest.approx(0.7)}
    renames = [r for r in manifest["renamed"] if r["from"].endswith("rewards")]
    assert renames, manifest["renamed"]
    r = renames[0]
    assert r["from"] == "verifier_result.rewards"
    assert r["to"] == "verifier_result.scores"
    assert r["run"] in {"run_1", "run_2", "run_3"}
    assert "reason" in r and r["reason"]


def test_scores_source_is_left_alone(tmp_path):
    """A trial already speaking `scores` must not be renamed, and nothing recorded."""
    run_root, _, _ = build_run_root(tmp_path, n_iters=1, verifier_style="md")
    out = convert_bundle(run_root, tmp_path / "delivery")
    manifest = json.loads((out / "manifest.json").read_text())
    res = json.loads((out / "trajectories" / "claude-opus-5" /
                      "run_1" / "result.json").read_text())
    assert "scores" in res["verifier_result"]
    assert not [r for r in manifest["renamed"] if "verifier_result" in r["from"]]


def test_inner_reward_key_also_renamed(converted):
    """The inner {"reward": x} becomes {"score": x} to match the delivery shape."""
    _, _, _, out, manifest = converted
    res = json.loads((out / "trajectories" / "claude-opus-5" / "run_1" / "result.json").read_text())
    assert list(res["verifier_result"]["scores"]) == ["score"]


def test_source_result_checksum_recorded(converted):
    """A reviewer can re-derive what the source said before the rename."""
    run_root, _, trial_ids, out, manifest = converted
    row = [r for r in manifest["runs"] if r["run"] == "run_3"][0]
    src = Path(row["source_trial_dir"]) / "result.json"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    assert row["source_result_sha256"] == digest


# --------------------------------------------------------------------------
# verifier md/txt fallback
# --------------------------------------------------------------------------

def test_md_style_verifier_copied_verbatim(tmp_path):
    run_root, _, _ = build_run_root(tmp_path, n_iters=1, verifier_style="md")
    manifest = pd.convert(run_root, tmp_path / "delivery")
    out = tmp_path / "delivery" / run_root.name
    v = out / "trajectories" / "claude-opus-5" / "run_1" / "verifier"
    assert rel_files(v) == ["grade-stdout.md", "score.json", "score.md", "test-stdout.md"]
    assert json.loads((v / "score.json").read_text())["score"] == pytest.approx(0.9)
    assert (v / "score.md").read_text().strip() == "0.9"
    assert not [f for f in manifest["fallbacks"] if f["run"] == "run_1"]


def test_txt_style_verifier_falls_back_and_is_recorded(converted):
    _, _, _, out, manifest = converted
    v = out / "trajectories" / "claude-opus-5" / "run_1" / "verifier"
    assert rel_files(v) == ["grade-stdout.md", "score.json", "score.md", "test-stdout.md"]
    kinds = {(f["run"], f["destination"]): f for f in manifest["fallbacks"]}
    assert kinds[("run_1", "verifier/score.json")]["source"].endswith("verifier/reward.json")
    assert kinds[("run_1", "verifier/grade-stdout.md")]["source"].endswith("grade-stdout.txt")
    assert kinds[("run_1", "verifier/test-stdout.md")]["source"].endswith("test-stdout.txt")
    assert kinds[("run_1", "verifier/score.md")]["synthesised"] is True


def test_fallback_score_json_renames_reward_key(converted):
    _, _, _, out, manifest = converted
    v = out / "trajectories" / "claude-opus-5" / "run_1" / "verifier"
    assert json.loads((v / "score.json").read_text()) == {"score": pytest.approx(0.9)}
    assert any(r["from"] == "verifier/score.json:reward" for r in manifest["renamed"])


def test_no_verifier_file_is_empty(converted):
    _, _, _, out, _ = converted
    for path in Path(out).rglob("verifier/*"):
        assert path.stat().st_size > 0, path


def test_synthesised_score_md_matches_score_json(converted):
    _, _, _, out, _ = converted
    for i in (1, 2, 3):
        v = out / "trajectories" / "claude-opus-5" / f"run_{i}" / "verifier"
        score = json.loads((v / "score.json").read_text())["score"]
        assert (v / "score.md").read_text().strip() == str(score)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def test_manifest_written_and_complete(converted):
    run_root, _, _, out, manifest = converted
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    assert on_disk["source_run_root"] == str(run_root)
    assert on_disk["task_uuid"] == "17d66d37-7b3f-57d3-93ad-0263fc495147"
    assert on_disk["converter_version"] == pd.CONVERTER_VERSION
    assert on_disk["generated_at"].endswith("Z")
    for key in ("runs", "renamed", "redacted", "dropped", "fallbacks", "not_produced"):
        assert key in on_disk, key


def test_manifest_run_rows_map_source_to_destination(converted):
    _, _, _, out, manifest = converted
    for row in manifest["runs"]:
        assert Path(row["source_trial_dir"]).is_dir()
        assert (Path(out) / row["destination"]).is_dir()
        assert row["destination"] == f"trajectories/claude-opus-5/{row['run']}"


# --------------------------------------------------------------------------
# idempotency, clobbering, dry run
# --------------------------------------------------------------------------

def test_idempotent(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    root = tmp_path / "delivery"
    out = convert_bundle(run_root, root)
    first = tree_checksum(out)
    files_first = rel_files(out)
    pd.convert(run_root, root, force=True)
    assert tree_checksum(out) == first
    assert rel_files(out) == files_first


def test_stale_files_removed_on_reconvert(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    root = tmp_path / "delivery"
    out = convert_bundle(run_root, root)
    junk = out / "trajectories" / "claude-opus-5" / "run_9" / "leftover.txt"
    junk.parent.mkdir(parents=True)
    junk.write_text("stale")
    pd.convert(run_root, root, force=True)
    assert not junk.exists()


def test_refuses_to_clobber_a_non_empty_bundle(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    root = tmp_path / "delivery"
    bundle = root / run_root.name
    bundle.mkdir(parents=True)
    (bundle / "existing.txt").write_text("keep me")
    with pytest.raises(pd.DeliveryError) as e:
        pd.convert(run_root, root)
    assert "--force" in str(e.value)


def test_a_sibling_task_bundle_does_not_block(tmp_path):
    """The delivery root is shared, so another task's bundle is not a clobber."""
    run_root, _, _ = build_run_root(tmp_path)
    root = tmp_path / "delivery"
    sibling = root / "99999999-0000-0000-0000-000000000000"
    sibling.mkdir(parents=True)
    (sibling / "manifest.json").write_text('{"task_uuid": "other"}')
    out = convert_bundle(run_root, root)
    assert (out / "manifest.json").is_file()
    assert (sibling / "manifest.json").read_text() == '{"task_uuid": "other"}'


def test_force_on_one_task_leaves_a_sibling_intact(tmp_path):
    """--force clears this task's bundle only; a sibling's files survive."""
    run_root, _, _ = build_run_root(tmp_path)
    root = tmp_path / "delivery"
    out = convert_bundle(run_root, root)
    sibling = root / "99999999-0000-0000-0000-000000000000"
    (sibling / "trajectories").mkdir(parents=True)
    (sibling / "manifest.json").write_text('{"task_uuid": "other"}')
    pd.convert(run_root, root, force=True)
    assert (out / "manifest.json").is_file()
    assert (sibling / "manifest.json").is_file()
    assert (sibling / "trajectories").is_dir()


def test_empty_bundle_dir_is_not_a_clobber(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    root = tmp_path / "delivery"
    (root / run_root.name).mkdir(parents=True)
    pd.convert(run_root, root)
    assert (root / run_root.name / "manifest.json").is_file()


def test_dry_run_writes_nothing(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    out = tmp_path / "delivery"
    manifest = pd.convert(run_root, out, dry_run=True)
    assert not out.exists()
    assert len(manifest["runs"]) == 3


def test_missing_run_root_is_an_error(tmp_path):
    with pytest.raises(pd.DeliveryError):
        pd.convert(tmp_path / "nope", tmp_path / "out")


def test_run_root_without_jobs_is_an_error(tmp_path):
    rr = tmp_path / "runs" / "agentloop" / "abc"
    rr.mkdir(parents=True)
    with pytest.raises(pd.DeliveryError):
        pd.convert(rr, tmp_path / "out")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cli(*args):
    return subprocess.run(
        [sys.executable, str(HARNESS_ROOT / "tools" / "package_delivery.py"), *map(str, args)],
        capture_output=True, text=True)


def test_cli_help():
    r = _cli("--help")
    assert r.returncode == 0
    for flag in ("--run-root", "--out", "--dry-run", "--force"):
        assert flag in r.stdout, flag


def test_cli_converts(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    out = tmp_path / "delivery"
    r = _cli("--run-root", run_root, "--out", out)
    assert r.returncode == 0, r.stderr
    assert (out / run_root.name / "manifest.json").is_file()
    assert "run_3" in r.stdout


def test_cli_dry_run_writes_nothing(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    out = tmp_path / "delivery"
    r = _cli("--run-root", run_root, "--out", out, "--dry-run")
    assert r.returncode == 0, r.stderr
    assert not out.exists()


def test_cli_force_required(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    out = tmp_path / "delivery"
    bundle = out / run_root.name
    bundle.mkdir(parents=True)
    (bundle / "x.txt").write_text("x")
    r = _cli("--run-root", run_root, "--out", out)
    assert r.returncode != 0
    assert "--force" in r.stderr
    r = _cli("--run-root", run_root, "--out", out, "--force")
    assert r.returncode == 0, r.stderr


def test_cli_reports_which_verifier_names_were_used(tmp_path):
    run_root, _, _ = build_run_root(tmp_path, n_iters=1, verifier_style="txt")
    r = _cli("--run-root", run_root, "--out", tmp_path / "d", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "reward.json" in r.stdout and "score.json" in r.stdout


# --------------------------------------------------------------------------
# the real campaign
# --------------------------------------------------------------------------

def _campaign_is_running(run_root=None):
    """True while a loop holds the run root's flock; those tests need a quiet campaign."""
    lock = (run_root or REAL_RUN_ROOT) / ".lock"
    if not lock.exists():
        return False
    try:
        with open(lock, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        return True
    return False


@pytest.mark.skipif(not REAL_RUN_ROOT.is_dir(), reason="real campaign not present")
@pytest.mark.skipif(_campaign_is_running(), reason="a campaign is running against the real run root")
def test_real_campaign_converts(tmp_path):
    manifest = pd.convert(REAL_RUN_ROOT, tmp_path / "delivery")
    out = tmp_path / "delivery" / REAL_RUN_ROOT.name
    # Asserted as invariants, not as a snapshot: the campaign grows by one run
    # every time an iteration is added, and a size-pinned test would fail for
    # that alone. It is also no longer single-model -- iteration 5 was run by a
    # codex agent -- so the shape asserted is per model.
    assert manifest["model_slugs"] == sorted(
        p.name for p in (out / "trajectories").iterdir())
    assert "claude-opus-5" in manifest["model_slugs"]
    delivered = [r["run"] for r in manifest["runs"]]
    assert len(delivered) >= 3
    per_model = collections.defaultdict(list)
    for row in manifest["runs"]:
        per_model[row["model_slug"]].append(row["run"])
    for slug, runs in per_model.items():
        assert runs == [f"run_{n}" for n in range(1, len(runs) + 1)], slug
    # the campaign's own iteration order, unbroken and unduplicated across models
    assert [r["iteration"] for r in manifest["runs"]] == sorted(
        r["iteration"] for r in manifest["runs"])
    for row in manifest["runs"]:
        expected = [
            "agent/trajectory.json", "artifacts/optimizer.py", "config.json",
            "result.json", "verifier/grade-stdout.md", "verifier/score.json",
            "verifier/score.md", "verifier/test-stdout.md",
        ]
        # iteration 1 is the only run with no prior history to carry
        if row["iteration"] != 1:
            expected.append("agent/history.md")
        assert rel_files(out / row["destination"]) == sorted(expected), row["destination"]
    # every run's verifier_result speaks `rewards` until harbor is told otherwise
    renames = [r for r in manifest["renamed"] if r["from"] == "verifier_result.rewards"]
    assert len(renames) == len(delivered)
    # a trial graded before the score.* rewiring needs all four fallbacks; one
    # graded after needs none. Nothing in between is expected.
    per_run = collections.Counter(f["run"] for f in manifest["fallbacks"])
    assert set(per_run.values()) <= {4}
    assert len(manifest["headers_added"]) == len(delivered)


@pytest.mark.skipif(not REAL_RUN_ROOT.is_dir(), reason="real campaign not present")
@pytest.mark.skipif(_campaign_is_running(), reason="a campaign is running against the real run root")
def test_real_campaign_has_no_secret_in_output(tmp_path):
    out = convert_bundle(REAL_RUN_ROOT, tmp_path / "delivery")
    leaked = []
    for path in Path(out).rglob("*"):
        if path.is_file():
            text = path.read_bytes().decode("utf-8", "replace")
            for needle in ("oauth-placeholder", "oaut****der"):
                if needle in text:
                    leaked.append((path, needle))
    assert leaked == []


@pytest.mark.skipif(not REAL_RUN_ROOT.is_dir(), reason="real campaign not present")
@pytest.mark.skipif(_campaign_is_running(), reason="a campaign is running against the real run root")
def test_real_campaign_source_untouched(tmp_path):
    before = tree_checksum(REAL_RUN_ROOT)
    pd.convert(REAL_RUN_ROOT, tmp_path / "delivery")
    assert tree_checksum(REAL_RUN_ROOT) == before


# --------------------------------------------------------------------------
# verifier/test-stdout.md provenance header
# --------------------------------------------------------------------------

def _stdout_md(out, run="run_1"):
    return (out / "trajectories" / "claude-opus-5" / run
            / "verifier" / "test-stdout.md").read_text()


def test_test_stdout_gets_a_two_line_provenance_header(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    out = convert_bundle(run_root, tmp_path / "delivery")
    lines = _stdout_md(out).split("\n")
    assert lines[0].startswith("# pytest ")
    assert lines[1].startswith("# ")
    assert lines[2] == ""


def test_header_names_the_destination_relpath(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    out = convert_bundle(run_root, tmp_path / "delivery")
    for n in (1, 2, 3):
        first = _stdout_md(out, f"run_{n}").split("\n")[0]
        assert f"  target: trajectories/claude-opus-5/run_{n}" in first


def test_header_timestamp_is_the_trials_finish_time_not_the_wall_clock(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    out = convert_bundle(run_root, tmp_path / "delivery")
    # fixture finished_at for run_1 is 2026-08-18T11:30:45.123456Z
    assert _stdout_md(out, "run_1").startswith("# pytest 2026-08-18T11:30:45Z  target: ")
    assert _stdout_md(out, "run_2").startswith("# pytest 2026-08-18T12:30:45Z  target: ")


def test_header_survives_reconvert_without_doubling(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    out = convert_bundle(run_root, tmp_path / "delivery")
    first = _stdout_md(out)
    pd.convert(run_root, out, force=True)
    assert _stdout_md(out) == first
    assert first.count("  target: ") == 1


def test_header_not_added_when_source_already_carries_one(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    trial = run_root / "jobs" / "agentic_iter01" / "minicalc__TRIAL01"
    (trial / "verifier" / "test-stdout.txt").write_text(
        "# pytest 2020-01-01T00:00:00Z  target: trajectories/x/run_1\n"
        "# already carried\n\nbody\n")
    out = convert_bundle(run_root, tmp_path / "delivery")
    body = _stdout_md(out)
    assert body.count("  target: ") == 1
    assert body.startswith("# pytest 2020-01-01T00:00:00Z")


def test_header_says_pytest_was_unavailable_when_it_did_not_run(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    trial = run_root / "jobs" / "agentic_iter01" / "minicalc__TRIAL01"
    (trial / "verifier" / "test-stdout.txt").write_text(
        "pytest not available in this image; SKIPPED\n"
        "Nothing was asserted here -- do not read this as assertions passing.\n")
    out = convert_bundle(run_root, tmp_path / "delivery")
    second = _stdout_md(out).split("\n")[1]
    assert "pytest unavailable" in second
    assert "sole verifier" in second


def test_header_credits_pytest_when_a_summary_line_is_present(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    trial = run_root / "jobs" / "agentic_iter01" / "minicalc__TRIAL01"
    (trial / "verifier" / "test-stdout.txt").write_text(
        "============================= test session starts ====================\n"
        "collected 17 items\n"
        "======================== 16 passed, 1 failed in 0.4s =================\n")
    out = convert_bundle(run_root, tmp_path / "delivery")
    second = _stdout_md(out).split("\n")[1]
    assert "outcomes emitted by grade.py" in second
    assert "pytest unavailable" not in second


def test_grade_stdout_is_delivered_without_a_header(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    out = convert_bundle(run_root, tmp_path / "delivery")
    grade = (out / "trajectories" / "claude-opus-5" / "run_1"
             / "verifier" / "grade-stdout.md").read_text()
    assert not grade.startswith("# pytest ")
    assert "  target: " not in grade


def test_original_transcript_is_preserved_below_the_header(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    trial = run_root / "jobs" / "agentic_iter01" / "minicalc__TRIAL01"
    original = (trial / "verifier" / "test-stdout.txt").read_text()
    out = convert_bundle(run_root, tmp_path / "delivery")
    assert _stdout_md(out).endswith(original)


def test_manifest_records_every_header_added(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    manifest = pd.convert(run_root, tmp_path / "delivery")
    out = tmp_path / "delivery" / run_root.name
    added = manifest["headers_added"]
    assert [h["run"] for h in added] == ["run_1", "run_2", "run_3"]
    assert {h["destination"] for h in added} == {"verifier/test-stdout.md"}
    assert {h["timestamp_source"] for h in added} == {"result.json:finished_at"}


# --------------------------------------------------------------------------
# refusing / skipping runs that cannot be delivered
# --------------------------------------------------------------------------

def test_has_submission_is_false_when_nothing_was_submitted(tmp_path):
    """`Path.glob` returns a generator, which is truthy even when it yields nothing."""
    trial = tmp_path / "trial"
    (trial / "artifacts").mkdir(parents=True)
    assert pd._has_submission(trial) is False
    _write(trial / "artifacts" / "workspace" / "submission" / "optimizer.py", "x = 1\n")
    assert pd._has_submission(trial) is True


def test_iteration_that_submitted_nothing_is_skipped_not_fatal(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    trial = run_root / "jobs" / "agentic_iter02" / "minicalc__TRIAL02"
    shutil.rmtree(trial / "artifacts")
    (trial / "artifacts").mkdir()
    manifest = pd.convert(run_root, tmp_path / "delivery")
    out = tmp_path / "delivery" / run_root.name
    # Numbering is over DELIVERED runs, so the undeliverable iteration leaves no
    # gap; the row is what says run_2 came from agentic_iter03.
    assert [r["run"] for r in manifest["runs"]] == ["run_1", "run_2"]
    assert [r["source_job"] for r in manifest["runs"]] == ["agentic_iter01", "agentic_iter03"]
    assert [r["iteration"] for r in manifest["runs"]] == [1, 3]
    assert not (out / "trajectories" / "claude-opus-5" / "run_3").exists()
    assert "iteration 3" in (out / "trajectories" / "claude-opus-5" / "run_2"
                             / "artifacts" / "optimizer.py").read_text()


def test_skipped_iteration_is_named_in_the_manifest(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    trial = run_root / "jobs" / "agentic_iter02" / "minicalc__TRIAL02"
    shutil.rmtree(trial / "artifacts")
    (trial / "artifacts").mkdir()
    manifest = pd.convert(run_root, tmp_path / "delivery")
    assert len(manifest["skipped"]) == 1
    row = manifest["skipped"][0]
    assert row["iteration"] == 2
    assert row["source_job"] == "agentic_iter02"
    assert "no optimizer submitted" in row["reason"]


def test_no_skips_recorded_for_a_clean_campaign(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    manifest = pd.convert(run_root, tmp_path / "delivery")
    assert manifest["skipped"] == []


def test_refuses_to_package_a_campaign_that_is_still_running(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    lock = run_root / ".lock"
    lock.write_text("")
    holder = open(lock, "a")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(pd.DeliveryError) as exc:
            pd.convert(run_root, tmp_path / "delivery")
        assert "locked by a running campaign" in str(exc.value)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()


def test_packages_normally_once_the_lock_is_released(tmp_path):
    run_root, _, _ = build_run_root(tmp_path)
    lock = run_root / ".lock"
    lock.write_text("")
    holder = open(lock, "a")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(holder, fcntl.LOCK_UN)
    holder.close()
    manifest = pd.convert(run_root, tmp_path / "delivery")
    assert [r["run"] for r in manifest["runs"]] == ["run_1", "run_2", "run_3"]


def _flatten_jobs(run_root):
    """Move job dirs out of `jobs/` and up beside the ledger, then drop `jobs/`.

    Reproduces the layout ship-task/tools/refine.py wrote, where jobs_dir was
    the run root itself rather than a `jobs/` subdirectory.
    """
    jobs = run_root / "jobs"
    for child in list(jobs.iterdir()):
        shutil.move(str(child), str(run_root / child.name))
    jobs.rmdir()
    return run_root


def test_flat_layout_converts_when_jobs_sit_in_the_run_root(tmp_path):
    run_root, _, _ = build_run_root(tmp_path, n_iters=2, job_name="agentic-gpt")
    _flatten_jobs(run_root)
    manifest = pd.convert(run_root, tmp_path / "delivery")
    assert [r["run"] for r in manifest["runs"]] == ["run_1", "run_2"]
    assert [r["source_job"] for r in manifest["runs"]] == [
        "agentic-gpt_iter01", "agentic-gpt_iter02"]


def test_nested_jobs_dir_outranks_a_stray_top_level_job_dir(tmp_path):
    run_root, _, _ = build_run_root(tmp_path, n_iters=1, job_name="agentic")
    decoy = run_root / "agentic_iter99"
    shutil.copytree(run_root / "jobs" / "agentic_iter01", decoy)
    manifest = pd.convert(run_root, tmp_path / "delivery")
    assert [r["source_job"] for r in manifest["runs"]] == ["agentic_iter01"]


def test_neither_layout_raises_and_names_both_places_searched(tmp_path):
    run_root, _, _ = build_run_root(tmp_path, n_iters=1)
    shutil.rmtree(run_root / "jobs")
    with pytest.raises(pd.DeliveryError) as exc:
        pd.convert(run_root, tmp_path / "delivery")
    assert str(run_root) in str(exc.value)
    assert "jobs" in str(exc.value)


def test_retest_job_dir_is_excluded_under_the_flat_layout(tmp_path):
    run_root, _, _ = build_run_root(tmp_path, n_iters=2, job_name="agentic-gpt")
    _flatten_jobs(run_root)
    shutil.copytree(run_root / "agentic-gpt_iter01",
                    run_root / "agentic-gpt_iter01-retest")
    manifest = pd.convert(run_root, tmp_path / "delivery")
    rows = manifest["runs"] + manifest["skipped"]
    assert not any("retest" in str(r.get("source_job", "")) for r in rows)
