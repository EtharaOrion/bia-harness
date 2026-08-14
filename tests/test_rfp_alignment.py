import json
import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS_ROOT / "runner"))

import harness  # noqa: E402
from ingest_result import CANONICAL_FIELDS, normalize, append  # noqa: E402


def test_canonical_fields_include_rfp_alignment_fields():
    for name in ("attempt", "input_tokens", "output_tokens", "total_tokens"):
        assert name in CANONICAL_FIELDS, f"{name!r} missing from CANONICAL_FIELDS"


def test_canonical_fields_length_is_33():
    assert len(CANONICAL_FIELDS) == 33


def test_canonical_fields_include_work_dir():
    assert "work_dir" in CANONICAL_FIELDS


def test_canonical_fields_include_harbor_trial_dir():
    assert "harbor_trial_dir" in CANONICAL_FIELDS


def test_canonical_fields_include_bia_fields():
    for name in ("verifier_dir", "consolidated_reward", "verification_complete", "process_score"):
        assert name in CANONICAL_FIELDS, f"{name!r} missing from CANONICAL_FIELDS"


def test_canonical_fields_include_model_name_and_retry_count():
    for name in ("model_name", "retry_count"):
        assert name in CANONICAL_FIELDS, f"{name!r} missing from CANONICAL_FIELDS"


def test_normalize_model_name_and_retry_count_defaults():
    row = normalize(
        variant="one_shot.py", seed=0, backend="dry",
        task_id="nanogpt/env-smoke", trial=0, status="success",
        reward_payload={"reward": 0.5, "hit_target": True},
        error=None, log_path=None,
    )
    assert row["model_name"] is None
    assert row["retry_count"] == 0


def test_normalize_model_name_and_retry_count_propagate():
    row = normalize(
        variant="iter3.py", attempt=3, seed=1, backend="harbor",
        task_id="nanogpt/track3-speedrun", trial=1, status="success",
        reward_payload={"reward": 0.7, "hit_target": True},
        error=None, log_path=None,
        model_name="anthropic/claude-opus-4-8",
        retry_count=2,
    )
    assert row["model_name"] == "anthropic/claude-opus-4-8"
    assert row["retry_count"] == 2


def test_normalize_populates_new_fields():
    row = normalize(
        variant="iter5.py", attempt=5, seed=0, backend="dry",
        task_id="nanogpt/env-smoke", trial=0, status="success",
        reward_payload={"reward": 0.7, "hit_target": True, "step_to_3_28": 3000},
        error=None, log_path="/tmp/x.log",
        input_tokens=1234, output_tokens=567, total_tokens=1801,
    )
    assert row["attempt"] == 5
    assert row["input_tokens"] == 1234
    assert row["output_tokens"] == 567
    assert row["total_tokens"] == 1801


def test_normalize_defaults_new_fields_to_none():
    row = normalize(
        variant="one_shot.py", seed=0, backend="dry",
        task_id="nanogpt/env-smoke", trial=0, status="success",
        reward_payload={"reward": 0.5, "hit_target": True},
        error=None, log_path=None,
    )
    for field in ("attempt", "input_tokens", "output_tokens", "total_tokens"):
        assert row[field] is None, f"{field} should default to None"


def test_iterations_alias_still_parses():
    args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry", "--iterations", "3"])
    assert args.attempts == 3


def test_attempts_new_name_parses():
    args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry", "--attempts", "7"])
    assert args.attempts == 7


def test_llm_retries_default_is_3():
    args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry"])
    assert args.llm_retries == 3


def test_llm_retries_flag_parses():
    args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry", "--llm-retries", "7"])
    assert args.llm_retries == 7


def test_seeds_default_is_2():
    args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry"])
    assert args.seeds == 2


def test_rfp_flag_defaults_to_11aug_requirements():
    args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry"])
    assert args.rfp.name == "11aug requirements.md"


def test_rfp_flag_override(tmp_path):
    args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry",
                               "--rfp", str(tmp_path / "custom-rfp.md")])
    assert args.rfp == tmp_path / "custom-rfp.md"


def test_stop_condition_flags_default_to_none():
    args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry"])
    assert args.target_reward is None
    assert args.max_wall_clock_sec is None
    assert args.max_input_tokens is None
    assert args.k_consecutive_incomplete is None


def test_stop_condition_flags_parse():
    args = harness.parse_args([
        "--task", "nanogpt-smoke", "--backend", "dry",
        "--target-reward", "0.9",
        "--max-wall-clock-sec", "3600",
        "--max-input-tokens", "500000",
        "--k-consecutive-incomplete", "3",
    ])
    assert args.target_reward == 0.9
    assert args.max_wall_clock_sec == 3600.0
    assert args.max_input_tokens == 500000
    assert args.k_consecutive_incomplete == 3


def test_retries_flag_is_removed():
    with pytest.raises(SystemExit):
        harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry", "--retries", "10"])


def test_final_flag_choices():
    for policy in ("none", "last", "best"):
        args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry", "--final", policy])
        assert args.final == policy


def test_final_flag_rejects_bad_value():
    with pytest.raises(SystemExit):
        harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry", "--final", "middle"])


def _write_rows(ledger: Path, rows: list[dict]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        append(ledger, r)


def test_write_final_selection_best(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    rows = [
        normalize(variant="a.py", attempt=1, seed=0, backend="dry", task_id="t/x",
                  trial=0, status="success",
                  reward_payload={"reward": 0.3, "step_to_3_28": 3400, "hit_target": True},
                  error=None, log_path="/l/a0.log"),
        normalize(variant="b.py", attempt=2, seed=0, backend="dry", task_id="t/x",
                  trial=0, status="success",
                  reward_payload={"reward": 0.9, "step_to_3_28": 2900, "hit_target": True},
                  error=None, log_path="/l/b0.log"),
        normalize(variant="c.py", attempt=3, seed=0, backend="dry", task_id="t/x",
                  trial=0, status="success",
                  reward_payload={"reward": 0.7, "step_to_3_28": 3100, "hit_target": True},
                  error=None, log_path="/l/c0.log"),
    ]
    _write_rows(ledger, rows)

    out = harness.write_final_selection(ledger, "best", "t/x")
    assert out is not None and out.is_file()
    payload = json.loads(out.read_text())
    assert payload["final_selection_policy"] == "best"
    assert payload["selected_variant"] == "b.py"
    assert payload["selected_attempt"] == 2
    assert payload["selected_reward"] == 0.9


def test_write_final_selection_last(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    rows = [
        normalize(variant="a.py", attempt=1, seed=0, backend="dry", task_id="t/x",
                  trial=0, status="success",
                  reward_payload={"reward": 0.9, "hit_target": True},
                  error=None, log_path=None),
        normalize(variant="z.py", attempt=2, seed=0, backend="dry", task_id="t/x",
                  trial=0, status="success",
                  reward_payload={"reward": 0.4, "hit_target": True},
                  error=None, log_path=None),
    ]
    _write_rows(ledger, rows)

    out = harness.write_final_selection(ledger, "last", "t/x")
    payload = json.loads(out.read_text())
    assert payload["final_selection_policy"] == "last"
    assert payload["selected_variant"] == "z.py"
    assert payload["selected_attempt"] == 2


def test_write_final_selection_none_returns_none(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    ledger.write_text("")
    assert harness.write_final_selection(ledger, "none", "t/x") is None


def test_write_final_selection_filters_by_task_id(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    rows = [
        normalize(variant="a.py", attempt=1, seed=0, backend="dry", task_id="t/other",
                  trial=0, status="success",
                  reward_payload={"reward": 0.99, "hit_target": True},
                  error=None, log_path=None),
        normalize(variant="b.py", attempt=1, seed=0, backend="dry", task_id="t/x",
                  trial=0, status="success",
                  reward_payload={"reward": 0.2, "hit_target": True},
                  error=None, log_path=None),
    ]
    _write_rows(ledger, rows)

    payload = json.loads(harness.write_final_selection(ledger, "best", "t/x").read_text())
    assert payload["selected_variant"] == "b.py", "other-task row must not leak into selection"


def test_write_final_selection_skips_error_rows(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    rows = [
        normalize(variant="a.py", attempt=1, seed=0, backend="dry", task_id="t/x",
                  trial=0, status="error",
                  reward_payload={"reward": 0.99},
                  error="boom", log_path=None),
        normalize(variant="b.py", attempt=1, seed=0, backend="dry", task_id="t/x",
                  trial=0, status="success",
                  reward_payload={"reward": 0.3, "hit_target": True},
                  error=None, log_path=None),
    ]
    _write_rows(ledger, rows)

    payload = json.loads(harness.write_final_selection(ledger, "best", "t/x").read_text())
    assert payload["selected_variant"] == "b.py"


def test_write_final_selection_empty_ledger_returns_none(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    assert harness.write_final_selection(ledger, "best", "t/x") is None
