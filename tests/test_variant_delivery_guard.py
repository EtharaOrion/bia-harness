"""Guard: --variant must fail loudly when it cannot physically reach the container.

Harbor uploads only the workdir, _mount_targets(), /tests, /solution and the
task's environment/ dir into a trial container (harbor/environments/islo.py,
~L918-922). The task root is never mapped to /workspace. For tasks that ship
/workspace inside a prebuilt image (bia/track3nov:v2), copying a variant into
the mounted task tree is a silent no-op: the run "succeeds" while grading the
optimizer baked into the image. These tests pin the loud failure.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

import harness


HARNESS_ROOT = Path(__file__).resolve().parent.parent
BIA_TASK = HARNESS_ROOT / "tasks" / "2739a678-1759-516d-8ba7-1cd023267ea8"
NANOGPT = HARNESS_ROOT / "tasks" / "nanogpt-speedrun"


def _variant(tmp_path: Path) -> Path:
    v = tmp_path / "iter0.py"
    v.write_text("# variant sentinel\n")
    return v


# --------------------------------------------------------------------------
# _workspace_ships_in_image
# --------------------------------------------------------------------------

def test_workspace_ships_in_image_true_for_bia_task():
    assert harness._workspace_ships_in_image(BIA_TASK) is True


def test_workspace_ships_in_image_false_for_nanogpt_speedrun():
    # Has [environment].docker_image but declares no /workspace artifacts,
    # so the mounted tree really is the workspace -> variant delivery works.
    assert harness._workspace_ships_in_image(NANOGPT) is False


def test_workspace_ships_in_image_false_without_docker_image(tmp_path):
    task = tmp_path / "synthetic"
    task.mkdir()
    (task / "task.toml").write_text(
        'schema_version = "1.4"\n'
        'artifacts = ["/workspace/submission/optimizer.py"]\n\n'
        '[task]\nname = "synthetic/no-image"\n'
    )
    assert harness._workspace_ships_in_image(task) is False


def test_workspace_ships_in_image_false_without_workspace_artifacts(tmp_path):
    task = tmp_path / "synthetic"
    task.mkdir()
    (task / "task.toml").write_text(
        'schema_version = "1.4"\n'
        'artifacts = ["/logs/verifier/reward.json"]\n\n'
        '[task]\nname = "synthetic/no-ws"\n\n'
        '[environment]\ndocker_image = "some/image:v1"\n'
    )
    assert harness._workspace_ships_in_image(task) is False


def test_workspace_ships_in_image_tolerates_missing_task_toml(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert harness._workspace_ships_in_image(empty) is False


def test_workspace_ships_in_image_tolerates_malformed_task_toml(tmp_path):
    task = tmp_path / "broken"
    task.mkdir()
    (task / "task.toml").write_text("this is [not valid toml\n")
    assert harness._workspace_ships_in_image(task) is False


# --------------------------------------------------------------------------
# harness.run() guard
# --------------------------------------------------------------------------

def test_harbor_variant_on_image_shipped_workspace_raises(tmp_path):
    with pytest.raises(harness.VariantDeliveryImpossible) as exc:
        harness.run(
            task="2739a678-1759-516d-8ba7-1cd023267ea8",
            seeds=1,
            backend="harbor",
            out_root=tmp_path / "work",
            variant=_variant(tmp_path),
            ledger=tmp_path / "runs.jsonl",
        )

    msg = str(exc.value)
    assert "bia/track3nov" in msg
    assert "/workspace/submission/optimizer.py" in msg
    assert "in-container" in msg


def test_guard_raises_before_any_mount_or_subprocess(tmp_path):
    """No docker, no harbor, no mounting -- the guard fires first."""
    def _boom(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("work was performed before the guard fired")

    with patch.object(harness, "mount_task", _boom), \
         patch.object(harness.subprocess, "run", _boom), \
         patch.dict(harness.DISPATCHERS, {"harbor": _boom}):
        with pytest.raises(harness.VariantDeliveryImpossible):
            harness.run(
                task="2739a678-1759-516d-8ba7-1cd023267ea8",
                seeds=1,
                backend="harbor",
                out_root=tmp_path / "work",
                variant=_variant(tmp_path),
                ledger=tmp_path / "runs.jsonl",
            )

    # Nothing was created on the way out.
    assert not (tmp_path / "work").exists()


def test_no_variant_on_bia_task_is_not_guarded(tmp_path):
    """Guard is about variant delivery only; a variant-less harbor run is legal.

    run() absorbs dispatch failures into an error row, so reaching the stubbed
    dispatcher at all proves the guard did not fire.
    """
    reached = []

    def _dispatch(*a, **k):
        reached.append(True)
        raise RuntimeError("reached dispatch")

    with patch.dict(harness.DISPATCHERS, {"harbor": _dispatch}):
        rows = harness.run(
            task="2739a678-1759-516d-8ba7-1cd023267ea8",
            seeds=1,
            backend="harbor",
            out_root=tmp_path / "work",
            variant=None,
            ledger=tmp_path / "runs.jsonl",
        )

    assert reached, "guard fired on a variant-less run"
    assert rows[0]["variant"] == "(none)"


def test_dry_backend_with_variant_does_not_raise(tmp_path):
    """The guard is harbor-only: dry mounts locally, so delivery is real."""
    rows = harness.run(
        task="nanogpt-speedrun",
        seeds=1,
        backend="dry",
        out_root=tmp_path / "work",
        variant=_variant(tmp_path),
        ledger=tmp_path / "runs.jsonl",
    )
    assert len(rows) == 1
    assert rows[0]["variant"] == "iter0.py"


def test_variant_delivery_impossible_is_a_runtime_error():
    assert issubclass(harness.VariantDeliveryImpossible, RuntimeError)
