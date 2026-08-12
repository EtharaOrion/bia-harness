from __future__ import annotations

from pathlib import Path

import pytest

from mount_variant import _resolve_mount_config_path, mount_task


HARNESS_ROOT = Path(__file__).resolve().parent.parent


def _make_task(tmp_path: Path, name: str) -> Path:
    task_dir = tmp_path / "tasks" / name
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task_dir / "task.toml").write_text('version = "1.0"\n[task]\nname = "t/x"\n')
    return task_dir


def _make_variant(tmp_path: Path) -> Path:
    v = tmp_path / "variants" / "iter0.py"
    v.parent.mkdir(parents=True, exist_ok=True)
    v.write_text("# variant\n")
    return v


def test_no_config_returns_none(tmp_path):
    task_dir = _make_task(tmp_path, "task-a")
    assert _resolve_mount_config_path(task_dir, tmp_path) is None


def test_in_tree_config_wins_over_fallback(tmp_path):
    task_dir = _make_task(tmp_path, "task-a")
    in_tree = task_dir / "mount.toml"
    in_tree.write_text('version = "1.0"\n')
    fallback = tmp_path / "mount-configs" / "task-a.toml"
    fallback.parent.mkdir()
    fallback.write_text('version = "1.0"\n')
    assert _resolve_mount_config_path(task_dir, tmp_path) == in_tree


def test_fallback_used_when_no_in_tree(tmp_path):
    task_dir = _make_task(tmp_path, "task-a")
    fallback = tmp_path / "mount-configs" / "task-a.toml"
    fallback.parent.mkdir()
    fallback.write_text('version = "1.0"\n')
    assert _resolve_mount_config_path(task_dir, tmp_path) == fallback


def test_mount_task_uses_fallback_variant_dst(tmp_path):
    task_dir = _make_task(tmp_path, "bia-slug")
    (task_dir / "submission").mkdir()
    fallback = tmp_path / "mount-configs" / "bia-slug.toml"
    fallback.parent.mkdir()
    fallback.write_text(
        'version = "1.0"\n[variant]\ndst = "submission/optimizer.py"\nrequired = false\n'
    )
    variant = _make_variant(tmp_path)
    out = tmp_path / "runs" / "work"
    mount_task(task_dir, out, variant=variant, harness_root=tmp_path)
    assert (out / "submission" / "optimizer.py").read_text() == "# variant\n"


def test_repo_bia_task_resolves_via_out_of_tree_config():
    task_dir = HARNESS_ROOT / "tasks" / "b92e1502-7efe-5004-af4a-a7715da77b41"
    if not task_dir.exists():
        pytest.skip("bia task not present")
    cfg = HARNESS_ROOT / "mount-configs" / f"{task_dir.name}.toml"
    if not cfg.is_file():
        pytest.skip("bia mount-config not present")
    resolved = _resolve_mount_config_path(task_dir, HARNESS_ROOT)
    assert resolved == cfg
