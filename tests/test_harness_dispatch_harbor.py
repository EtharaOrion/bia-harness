import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import harness


HARNESS_ROOT = Path(__file__).resolve().parent.parent
NANOGPT_SMOKE = HARNESS_ROOT / "tasks" / "nanogpt-smoke"


def test_build_harbor_cmd_minimal(tmp_path):
    cmd = harness.build_harbor_cmd(
        tmp_path / "mounted", seed=0, jobs_dir=tmp_path / "jobs",
        llm_config=None, agent="nop", attempts=1, concurrent=1,
    )
    assert cmd[:2] == ["harbor", "run"]
    assert "-a" in cmd and cmd[cmd.index("-a") + 1] == "nop"
    assert "-k" in cmd and cmd[cmd.index("-k") + 1] == "1"
    assert "-n" in cmd and cmd[cmd.index("-n") + 1] == "1"
    assert "-m" not in cmd
    assert "--allow-agent-host" not in cmd
    ae_pairs = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--ae"]
    ae_dict = dict(pair.split("=", 1) for pair in ae_pairs)
    assert ae_dict == {"BIA_HARNESS_SEED": "0", "SEED": "0", "PYTHONHASHSEED": "0"}
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "LITELLM_TIMEOUT", "LITELLM_NUM_RETRIES"):
        assert k not in ae_dict


def test_build_harbor_cmd_with_llm_config(tmp_path):
    llm = {
        "model": "anthropic/claude-opus-4-8",
        "base_url": "http://172.17.0.1:8765",
        "api_key": "stub-key-24-characters!",
        "timeout": 600,
        "num_retries": 2,
    }
    cmd = harness.build_harbor_cmd(
        tmp_path / "mounted", seed=3, jobs_dir=tmp_path / "jobs",
        llm_config=llm, agent="claude-code", attempts=100, concurrent=4,
    )
    assert cmd[cmd.index("-a") + 1] == "claude-code"
    assert cmd[cmd.index("-k") + 1] == "100"
    assert cmd[cmd.index("-n") + 1] == "4"
    assert cmd[cmd.index("-m") + 1] == "claude-opus-4-8"
    assert "--seed" not in cmd

    ae_pairs = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--ae"]
    ae_dict = dict(pair.split("=", 1) for pair in ae_pairs)
    assert "ANTHROPIC_BASE_URL" in ae_dict
    assert ae_dict["ANTHROPIC_API_KEY"] == "stub-key-24-characters!"
    assert ae_dict["LITELLM_TIMEOUT"] == "600"
    assert ae_dict["LITELLM_NUM_RETRIES"] == "2"
    assert ae_dict["SEED"] == "3"
    assert ae_dict["BIA_HARNESS_SEED"] == "3"
    assert "--allow-agent-host" in cmd


def test_load_llm_config_missing_field(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": "x", "base_url": "y"}))
    with pytest.raises(ValueError, match="api_key"):
        harness.load_llm_config(bad)


def test_load_llm_config_ok(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "model": "anthropic/claude-opus-4-8",
        "base_url": "http://172.17.0.1:8765",
        "api_key": "stub",
    }))
    cfg = harness.load_llm_config(good)
    assert cfg["model"] == "anthropic/claude-opus-4-8"


def test_reward_read_from_verifier_result_not_top_level():
    trial = {
        "trial_name": "t__abc",
        "verifier_result": {"rewards": {"reward": 0.87, "graded_steps": 2955}},
    }
    got = harness._reward_from_harbor(trial)
    assert got["reward"] == 0.87
    assert got["hit_target"] is True
    assert got["graded_steps"] == 2955


def test_reward_defaults_to_zero_when_verifier_absent():
    got = harness._reward_from_harbor({"trial_name": "t", "verifier_result": None})
    assert got["reward"] == 0.0
    assert got["hit_target"] is False


def test_trial_result_preferred_over_job_result(tmp_path):
    job = tmp_path / "2026-01-01__00-00-00"
    (job / "trial__x").mkdir(parents=True)
    (job / "result.json").write_text(json.dumps({"id": "job", "stats": {}}))
    trial = job / "trial__x" / "result.json"
    trial.write_text(json.dumps({"trial_name": "trial__x", "verifier_result": {"rewards": {"reward": 1.0}}}))
    found = sorted(tmp_path.rglob("result.json"))
    picked = [p for p in found if harness._is_trial_result(p)][-1]
    assert picked == trial


def test_harbor_model_flag_strips_provider_prefix(tmp_path):
    llm = {
        "model": "anthropic/claude-opus-5",
        "base_url": "http://172.17.0.1:8765",
        "api_key": "stub",
    }
    cmd = harness.build_harbor_cmd(
        tmp_path / "mounted", seed=0, jobs_dir=tmp_path / "jobs",
        llm_config=llm, agent="claude-code", attempts=1, concurrent=1,
    )
    assert cmd[cmd.index("-m") + 1] == "claude-opus-5"


def test_default_agent_is_a_real_harbor_agent():
    args = harness.parse_args(["--task", "nanogpt-smoke", "--backend", "dry"])
    assert args.agent == "claude-code"
