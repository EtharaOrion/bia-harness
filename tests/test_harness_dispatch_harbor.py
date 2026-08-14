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
        llm_config=None, agent="bash", attempts=1, concurrent=1,
    )
    assert cmd[:2] == ["harbor", "run"]
    assert "-a" in cmd and cmd[cmd.index("-a") + 1] == "bash"
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
        llm_config=llm, agent="claude_code", attempts=100, concurrent=4,
    )
    assert cmd[cmd.index("-a") + 1] == "claude_code"
    assert cmd[cmd.index("-k") + 1] == "100"
    assert cmd[cmd.index("-n") + 1] == "4"
    assert cmd[cmd.index("-m") + 1] == "anthropic/claude-opus-4-8"
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


def test_container_base_url_darwin_rewrite():
    with patch("harness.platform.system", return_value="Darwin"):
        got = harness._container_base_url("http://172.17.0.1:8765")
        assert got == "http://host.docker.internal:8765"


def test_container_base_url_linux_unchanged():
    with patch("harness.platform.system", return_value="Linux"):
        got = harness._container_base_url("http://172.17.0.1:8765")
        assert got == "http://172.17.0.1:8765"


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


def test_preflight_bridge_unreachable_raises():
    with pytest.raises(RuntimeError, match="claude_code_bridge unreachable"):
        harness.preflight_bridge("http://127.0.0.1:1", timeout_sec=0.5)


def test_task_uuid_resolution():
    task_dir = harness.resolve_task("b92e1502-7efe-5004-af4a-a7715da77b41")
    assert task_dir.name == "b92e1502-7efe-5004-af4a-a7715da77b41"
    assert (task_dir / "task.toml").is_file()
