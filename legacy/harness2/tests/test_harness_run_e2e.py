"""End-to-end `harness.run` dry-backend guards: synthetic marking + canonical ledger row.

Split out of `tests/test_synthetic_marking.py`; the pure `agentloop.marking` unit
tests stayed there because agentloop imports that module. These drive
`harness.run` / `ingest_result.CANONICAL_FIELDS`, which agentloop never touches.
"""
import json
from pathlib import Path

import ingest_result
import harness


def test_dry_run_writes_is_synthetic_into_per_seed_reward_json(tmp_path):
    variant = tmp_path / "dry_variant.py"
    variant.write_text("# no-op variant for dry run\n")
    ledger = tmp_path / "runs.jsonl"
    attempt_dir = tmp_path / "work" / "attempt_01"

    rows = harness.run(
        task="nanogpt-speedrun",
        seeds=2,
        backend="dry",
        out_root=tmp_path / "work",
        variant=variant,
        ledger=ledger,
        attempt_dir=attempt_dir,
        verifier_enabled=False,
    )

    assert len(rows) == 2
    for seed in range(2):
        reward_json = attempt_dir / f"seed_{seed}" / "reward.json"
        assert reward_json.is_file(), f"missing {reward_json}"
        payload = json.loads(reward_json.read_text())
        assert payload["is_synthetic"] is True, (
            f"seed {seed} reward.json must self-identify as synthetic: {payload}"
        )

    # dry-run semantics are unchanged
    for row in rows:
        assert row["backend"] == "dry"
        assert row["status"] == "success"
        assert row["reward"] is not None
        assert row["variant"] == "dry_variant.py"


def test_ledger_row_schema_is_untouched_by_marking():
    assert len(ingest_result.CANONICAL_FIELDS) == 33
    assert "is_synthetic" not in ingest_result.CANONICAL_FIELDS


def test_dry_ledger_rows_still_have_exactly_canonical_fields(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    harness.run(
        task="nanogpt-smoke",
        seeds=1,
        backend="dry",
        out_root=tmp_path / "work",
        ledger=ledger,
        verifier_enabled=False,
    )
    row = json.loads(Path(ledger).read_text().splitlines()[0])
    assert list(sorted(row)) == sorted(ingest_result.CANONICAL_FIELDS)
