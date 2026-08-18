from pathlib import Path

import grader

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_reaches_target_gives_expected_reward():
    result = grader.grade(FIXTURES / "log_reaches_target.txt")
    assert result["hit_target"] is True
    assert result["step_to_3_28"] == 2720
    assert result["final_val_loss"] == 3.27978
    assert result["min_val_loss"] == 3.27978
    assert result["n_val_points"] == 26
    assert result["final_step"] == 2720
    assert result["total_steps_planned"] == 2900
    assert 0 < result["reward"] <= 1
    assert result["reward"] == round((3500 - 2720) / (3500 - 2500), 6)


def test_no_target_returns_zero_reward():
    result = grader.grade(FIXTURES / "log_no_target.txt")
    assert result["hit_target"] is False
    assert result["reward"] == 0.0
    assert result["step_to_3_28"] is None
    assert result["n_val_points"] == 8
    assert result["final_val_loss"] == 3.63118


def test_empty_log_returns_zero_reward_with_reason():
    result = grader.grade(FIXTURES / "log_empty.txt")
    assert result["hit_target"] is False
    assert result["reward"] == 0.0
    assert result["n_val_points"] == 0
    assert "reason" in result


def test_reward_is_clipped_to_zero_one():
    fake = grader.compute_reward({
        "checkpoints": [{"step": 100, "val_loss": 3.0}],
        "total_steps": 3000,
        "final_train_time_s": 12.0,
    })
    assert fake["reward"] == 1.0
    assert fake["hit_target"] is True


def test_grader_writes_json_via_cli(tmp_path):
    import json
    import subprocess
    import sys

    out = tmp_path / "reward.json"
    proc = subprocess.run(
        [sys.executable, str(Path(grader.__file__)), "--write",
         str(FIXTURES / "log_reaches_target.txt"), str(out)],
        capture_output=True, text=True, check=True,
    )
    assert proc.returncode == 0
    payload = json.loads(out.read_text())
    assert payload["hit_target"] is True
    assert payload["step_to_3_28"] == 2720


def test_stat_sig_at_n1_matches_margin():
    result = grader.grade(FIXTURES / "log_reaches_target.txt")
    assert result["stat_sig_at_n1"] == round(3.28 - 3.27978, 6)


def test_target_constants_stable():
    assert grader.TARGET_VAL_LOSS == 3.28
    assert grader.BASELINE_STEPS == 3500
    assert grader.STRETCH_STEPS == 2500
