import tomllib
from pathlib import Path

import pytest

TASKS_ROOT = Path(__file__).resolve().parent.parent / "tasks"
TASK_DIRS = sorted(p for p in TASKS_ROOT.iterdir() if p.is_dir())
TASK_IDS = [p.name for p in TASK_DIRS]

NATIVE_TASK_DIRS = [p for p in TASK_DIRS if (p / "tests" / "grader.py").is_file()]
NATIVE_IDS = [p.name for p in NATIVE_TASK_DIRS]


def _load(task_dir):
    with (task_dir / "task.toml").open("rb") as f:
        return tomllib.load(f)


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=TASK_IDS)
def test_task_toml_parses(task_dir):
    data = _load(task_dir)
    schema = data.get("version") or data.get("harbor_schema_version")
    assert schema == "1.0", f"task.toml must declare version or harbor_schema_version = '1.0'; got {schema!r}"
    name = data.get("task", {}).get("name") or data.get("name")
    assert isinstance(name, str) and name, "task name must be present in [task].name or top-level name"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=TASK_IDS)
def test_task_dir_minimum_harbor_files(task_dir):
    assert (task_dir / "task.toml").is_file()
    assert (task_dir / "tests" / "test.sh").is_file(), "Harbor requires tests/test.sh entrypoint"
    env = _load(task_dir).get("environment", {})
    dockerfile = task_dir / "environment" / "Dockerfile"
    image_ref = env.get("docker_image") or env.get("image")
    assert dockerfile.is_file() or image_ref, "task must provide environment/Dockerfile or [environment].docker_image/image"


@pytest.mark.parametrize("task_dir", NATIVE_TASK_DIRS, ids=NATIVE_IDS)
def test_native_agent_and_verifier_timeouts_declared(task_dir):
    data = _load(task_dir)
    assert data["agent"]["timeout_sec"] > 0
    assert data["verifier"]["timeout_sec"] > 0
    assert data["verifier"]["environment_mode"] in {"shared", "separate"}


@pytest.mark.parametrize("task_dir", NATIVE_TASK_DIRS, ids=NATIVE_IDS)
def test_native_environment_shape(task_dir):
    env = _load(task_dir)["environment"]
    assert env["gpus"] >= 0
    assert isinstance(env["gpu_types"], list)
    assert env["memory_mb"] >= 1024
    assert env["storage_mb"] >= 1024
    assert env.get("network_mode") in {"public", "no-network", "allowlist", None}


@pytest.mark.parametrize("task_dir", NATIVE_TASK_DIRS, ids=NATIVE_IDS)
def test_native_task_dir_contains_extended_files(task_dir):
    assert (task_dir / "instruction.md").is_file()
    assert (task_dir / "environment" / "Dockerfile").is_file()
    assert (task_dir / "solution" / "solve.sh").is_file()
    assert (task_dir / "tests" / "grader.py").is_file()


def test_multiple_tasks_present():
    assert len(TASK_DIRS) >= 2, "harness must support multi-task; found <2 tasks"


def test_at_least_one_native_task_present():
    assert len(NATIVE_TASK_DIRS) >= 1, "at least one native (grader.py) task required for dry/local backends"
