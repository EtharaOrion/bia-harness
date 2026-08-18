import json
from pathlib import Path

import ingest_result


def _payload(**over):
    base = {
        "reward": 0.78,
        "hit_target": True,
        "step_to_3_28": 2720,
        "final_val_loss": 3.27978,
        "min_val_loss": 3.27978,
        "mean_val_loss": 3.4,
        "n_val_points": 26,
        "final_step": 2720,
        "total_steps_planned": 2900,
        "train_time_s": 872.4,
        "stat_sig_at_n1": 0.00022,
    }
    base.update(over)
    return base


def test_normalize_emits_all_canonical_fields():
    row = ingest_result.normalize(
        variant="v1.py", seed=0, backend="dry",
        task_id="nanogpt/track3-speedrun", trial=0, status="success",
        reward_payload=_payload(), error=None, log_path="/tmp/x.log",
    )
    for field in ingest_result.CANONICAL_FIELDS:
        assert field in row
    assert row["variant"] == "v1.py"
    assert row["reward"] == 0.78
    assert row["hit_target"] is True


def test_normalize_handles_error_case():
    row = ingest_result.normalize(
        variant="broken.py", seed=1, backend="local",
        task_id="nanogpt/track3-speedrun", trial=1, status="error",
        reward_payload={"reward": 0.0, "hit_target": False, "reason": "cuda oom"},
        error="RuntimeError: CUDA OOM", log_path=None,
    )
    assert row["reward"] == 0.0
    assert row["hit_target"] is False
    assert row["error"] == "RuntimeError: CUDA OOM"
    assert row["log_path"] is None
    assert row["step_to_3_28"] is None


def test_append_writes_jsonl_line(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    row = ingest_result.normalize(
        variant="v.py", seed=0, backend="dry",
        task_id="t", trial=0, status="success",
        reward_payload=_payload(), error=None, log_path="/tmp/x",
    )
    ingest_result.append(ledger, row)
    ingest_result.append(ledger, row)

    lines = ledger.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert parsed["reward"] == 0.78


def test_append_creates_parent_dir(tmp_path):
    ledger = tmp_path / "nested" / "deep" / "runs.jsonl"
    row = ingest_result.normalize(
        variant="v.py", seed=0, backend="dry",
        task_id="t", trial=0, status="success",
        reward_payload=_payload(), error=None, log_path=None,
    )
    ingest_result.append(ledger, row)
    assert ledger.is_file()
