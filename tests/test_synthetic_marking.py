"""Synthetic-result marking: dry-backend numbers are fabricated and must say so.

`harness.dispatch_dry` invents a val_loss curve from a hash of the seed. Those
numbers must never be mistaken for real training output, so every artifact they
touch carries an explicit `is_synthetic` flag and the reporting layer can render
a loud banner.
"""
import json
from pathlib import Path

import ingest_result
import harness
from track3.marking import (
    SYNTHETIC_BACKENDS,
    is_synthetic_backend,
    mark_reward_payload,
    synthetic_banner,
)


def test_dry_is_the_synthetic_backend():
    assert is_synthetic_backend("dry") is True
    assert is_synthetic_backend("harbor") is False
    assert is_synthetic_backend("local") is False
    assert "dry" in SYNTHETIC_BACKENDS


def test_mark_reward_payload_marks_dry_true_without_mutating_input():
    payload = {"reward": 1.0}
    marked = mark_reward_payload(payload, "dry")

    assert marked["is_synthetic"] is True
    assert marked["reward"] == 1.0
    assert payload == {"reward": 1.0}, "input payload must not be mutated"
    assert marked is not payload


def test_mark_reward_payload_marks_real_backends_false():
    marked = mark_reward_payload({"reward": 0.4}, "harbor")
    assert marked["is_synthetic"] is False


def test_synthetic_banner_fires_on_dry_rows():
    banner = synthetic_banner([{"backend": "dry"}])
    assert banner is not None
    assert "SYNTHETIC" in banner
    lowered = banner.lower()
    assert "fabricated" in lowered
    assert "not" in lowered and "real training" in lowered
    assert "\n" not in banner, "banner must be a single line"


def test_synthetic_banner_fires_on_is_synthetic_flag():
    banner = synthetic_banner([{"backend": "harbor", "is_synthetic": True}])
    assert banner is not None and "SYNTHETIC" in banner


def test_synthetic_banner_none_for_real_rows():
    assert synthetic_banner([{"backend": "harbor"}]) is None
    assert synthetic_banner([{"backend": "local"}, {"backend": "harbor"}]) is None
    assert synthetic_banner([]) is None


def test_synthetic_banner_fires_when_any_row_is_synthetic():
    banner = synthetic_banner([{"backend": "harbor"}, {"backend": "dry"}])
    assert banner is not None and "SYNTHETIC" in banner


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
