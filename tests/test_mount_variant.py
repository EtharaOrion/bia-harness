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


def _bia_style_task(root: Path, artifacts: list[str], mount_toml: str | None = None) -> Path:
    task = root / "bia-style"
    (task / "tests").mkdir(parents=True)
    rendered = ", ".join(f'"{a}"' for a in artifacts)
    (task / "task.toml").write_text(
        f'schema_version = "1.4"\nartifacts = [{rendered}]\n\n'
        '[task]\nname = "bia/track3nov"\n'
    )
    if mount_toml is not None:
        (task / "mount.toml").write_text(mount_toml)
    return task


def test_variant_dst_inferred_from_task_toml_artifacts(tmp_path):
    task = _bia_style_task(tmp_path, [
        "/telemetry/run_record.jsonl",
        "/workspace/submission/optimizer.py",
        "/workspace/submission/logs",
        "/logs/verifier/reward.json",
    ])
    variant = tmp_path / "iter0.py"
    variant.write_text("# inferred-dst sentinel\n")

    out = mount_variant.mount_task(task, tmp_path / "mounted", variant=variant)

    assert (out / "submission" / "optimizer.py").read_text() == "# inferred-dst sentinel\n"


def test_inferred_dst_prefers_submission_over_other_py_artifacts(tmp_path):
    task = _bia_style_task(tmp_path, [
        "/workspace/tools/helper.py",
        "/workspace/submission/optimizer.py",
    ])
    assert mount_variant._variant_dst_from_task_toml(task) == "submission/optimizer.py"


def test_mount_toml_takes_precedence_over_task_toml_inference(tmp_path):
    task = _bia_style_task(
        tmp_path,
        ["/workspace/submission/optimizer.py"],
        mount_toml='version = "1.0"\n\n[variant]\ndst = "environment/explicit.py"\nrequired = false\n',
    )
    variant = tmp_path / "iter0.py"
    variant.write_text("# precedence sentinel\n")

    out = mount_variant.mount_task(task, tmp_path / "mounted", variant=variant)

    assert (out / "environment" / "explicit.py").is_file()
    assert not (out / "submission" / "optimizer.py").exists()


def test_no_py_artifact_still_rejects_variant(tmp_path):
    task = _bia_style_task(tmp_path, ["/logs/verifier/reward.json", "/workspace/submission/logs"])
    variant = tmp_path / "iter0.py"
    variant.write_text("pass\n")
    with pytest.raises(ValueError, match="does not accept a --variant"):
        mount_variant.mount_task(task, tmp_path / "mounted", variant=variant)
