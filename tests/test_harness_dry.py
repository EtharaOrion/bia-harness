import json
from pathlib import Path

import harness


def test_dry_backend_end_to_end_with_variant(tmp_path):
    variant = tmp_path / "dry_variant.py"
    variant.write_text("# no-op variant for dry run\n")
    ledger = tmp_path / "runs.jsonl"

    rows = harness.run(
        task="nanogpt-speedrun",
        seeds=2,
        backend="dry",
        out_root=tmp_path / "work",
        variant=variant,
        ledger=ledger,
    )

    assert len(rows) == 2
    for row in rows:
        assert row["backend"] == "dry"
        assert row["variant"] == "dry_variant.py"
        assert row["task_id"] == "nanogpt/track3-speedrun"
        assert row["status"] == "success"
        assert row["reward"] is not None
        assert Path(row["log_path"]).is_file()

    ledger_rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(ledger_rows) == 2
    assert ledger_rows[0]["seed"] == 0
    assert ledger_rows[1]["seed"] == 1


def test_dry_backend_second_task_without_variant(tmp_path):
    ledger = tmp_path / "runs.jsonl"

    rows = harness.run(
        task="nanogpt-smoke",
        seeds=1,
        backend="dry",
        out_root=tmp_path / "work",
        ledger=ledger,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["backend"] == "dry"
    assert row["task_id"] == "nanogpt/env-smoke"
    assert row["variant"] == "(none)"
    assert Path(row["log_path"]).is_file()


def test_task_resolution_by_name_and_path(tmp_path):
    from pathlib import Path as P
    named = harness.resolve_task("nanogpt-speedrun")
    by_path = harness.resolve_task(str(harness.TASKS_ROOT / "nanogpt-speedrun"))
    assert named == by_path
    assert (named / "task.toml").is_file()


def test_task_resolution_missing_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        harness.resolve_task("does-not-exist")
