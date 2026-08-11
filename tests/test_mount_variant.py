from pathlib import Path

import pytest

import mount_variant

HARNESS_ROOT = Path(__file__).resolve().parent.parent
SPEEDRUN_TASK = HARNESS_ROOT / "tasks" / "nanogpt-speedrun"
SMOKE_TASK = HARNESS_ROOT / "tasks" / "nanogpt-smoke"


def test_mount_task_injects_shared_and_swaps_variant(tmp_path):
    variant = tmp_path / "custom_variant.py"
    variant.write_text("# custom variant sentinel\nprint('hello')\n")

    out = mount_variant.mount_task(SPEEDRUN_TASK, tmp_path / "mounted", variant=variant)

    swapped = out / "environment" / "train_gpt_simple.py"
    assert swapped.read_text().startswith("# custom variant sentinel")

    assert (out / "environment" / "cached_fineweb10B.py").is_file()
    assert (out / "environment" / "track3_README.md").is_file()

    assert (out / "task.toml").is_file()
    assert (out / "mount.toml").is_file()
    assert (out / "environment" / "Dockerfile").is_file()
    assert (out / "solution" / "solve.sh").is_file()
    assert (out / "tests" / "grader.py").is_file()
    assert (out / "tests" / "test.sh").is_file()


def test_mount_task_shared_without_variant(tmp_path):
    out = mount_variant.mount_task(SPEEDRUN_TASK, tmp_path / "mounted")

    trainer = out / "environment" / "train_gpt_simple.py"
    assert trainer.is_file()
    shared_trainer = HARNESS_ROOT / "shared" / "train_gpt_simple.py"
    assert trainer.read_bytes() == shared_trainer.read_bytes()


def test_mount_task_no_shared_no_variant_task_still_mounts(tmp_path):
    out = mount_variant.mount_task(SMOKE_TASK, tmp_path / "mounted")

    assert (out / "task.toml").is_file()
    assert (out / "mount.toml").is_file()
    assert (out / "environment" / "Dockerfile").is_file()
    assert (out / "tests" / "grader.py").is_file()


def test_mount_task_rejects_variant_on_task_without_variant_block(tmp_path):
    variant = tmp_path / "v.py"
    variant.write_text("pass\n")
    with pytest.raises(ValueError, match="does not accept a --variant"):
        mount_variant.mount_task(SMOKE_TASK, tmp_path / "mounted", variant=variant)


def test_mount_task_preserves_task_toml_bytes(tmp_path):
    original = (SPEEDRUN_TASK / "task.toml").read_bytes()
    out = mount_variant.mount_task(SPEEDRUN_TASK, tmp_path / "mounted")
    assert (out / "task.toml").read_bytes() == original


def test_mount_task_overwrites_existing_out_dir(tmp_path):
    variant = tmp_path / "v.py"
    variant.write_text("first\n")
    dest = tmp_path / "mounted"

    mount_variant.mount_task(SPEEDRUN_TASK, dest, variant=variant)
    assert (dest / "environment" / "train_gpt_simple.py").read_text() == "first\n"

    variant.write_text("second\n")
    mount_variant.mount_task(SPEEDRUN_TASK, dest, variant=variant)
    assert (dest / "environment" / "train_gpt_simple.py").read_text() == "second\n"


def test_mount_task_rejects_missing_task(tmp_path):
    with pytest.raises(FileNotFoundError):
        mount_variant.mount_task(tmp_path / "nope", tmp_path / "mounted")


def test_mount_task_rejects_missing_variant(tmp_path):
    with pytest.raises(FileNotFoundError):
        mount_variant.mount_task(SPEEDRUN_TASK, tmp_path / "mounted",
                                 variant=tmp_path / "nope.py")
