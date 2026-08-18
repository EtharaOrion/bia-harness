import tomllib
from pathlib import Path

import pytest

TASKS_ROOT = Path(__file__).resolve().parents[3] / "tasks"
HARNESS_ROOT = Path(__file__).resolve().parents[3]
TASK_DIRS = sorted(p for p in TASKS_ROOT.iterdir() if p.is_dir())
TASK_IDS = [p.name for p in TASK_DIRS]


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=TASK_IDS)
def test_mount_toml_parses_if_present(task_dir):
    mt = task_dir / "mount.toml"
    if not mt.is_file():
        pytest.skip(f"{task_dir.name} has no mount.toml (allowed)")
    with mt.open("rb") as f:
        data = tomllib.load(f)
    assert data.get("version") == "1.0"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=TASK_IDS)
def test_mount_toml_shared_assets_exist_in_harness(task_dir):
    mt = task_dir / "mount.toml"
    if not mt.is_file():
        pytest.skip("no mount.toml")
    with mt.open("rb") as f:
        data = tomllib.load(f)
    for entry in data.get("shared", []):
        src = HARNESS_ROOT / entry["src"]
        assert src.is_file(), \
            f"{task_dir.name}/mount.toml references missing shared asset: {entry['src']}"
        assert "dst" in entry and entry["dst"].startswith(("environment/", "tests/", "solution/")), \
            f"{task_dir.name}/mount.toml [shared] dst should stay inside task subdirs"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=TASK_IDS)
def test_mount_toml_variant_block_is_valid_if_present(task_dir):
    mt = task_dir / "mount.toml"
    if not mt.is_file():
        pytest.skip("no mount.toml")
    with mt.open("rb") as f:
        data = tomllib.load(f)
    variant = data.get("variant")
    if variant is None:
        pytest.skip("task does not accept a variant")
    assert "dst" in variant
    assert variant["dst"].startswith("environment/")
    assert isinstance(variant.get("required", False), bool)
