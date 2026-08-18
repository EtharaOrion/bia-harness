"""Synthetic-result marking: dry-backend numbers are fabricated and must say so.

`harness.dispatch_dry` invents a val_loss curve from a hash of the seed. Those
numbers must never be mistaken for real training output, so every artifact they
touch carries an explicit `is_synthetic` flag and the reporting layer can render
a loud banner.

The end-to-end `harness.run` half of these guards lives in
`legacy/harness2/tests/test_harness_run_e2e.py`; agentloop uses only `marking`.
"""
from agentloop.marking import (
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
