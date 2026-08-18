"""Validate the programmatic Harbor job-config against harbor's REAL schema.

Every assertion here is a guard against burning GPU hours on a typo: the config
this module builds is what `harbor run --config` consumes, and harbor's own
pydantic model is the only authority on what is legal.

The whole module skips if harbor is not importable, so the suite still passes
on a host without the harbor venv.
"""

import json

import pytest

from agentloop.harbor_config import (
    HarborConfigError,
    HarborUnavailable,
    agent_env,
    build_base_cfg,
    resolve_run_root,
    validate_cfg,
)

try:
    from agentloop.harbor_config import _load_job_config_cls

    _load_job_config_cls()
    HARBOR_OK = True
    HARBOR_WHY = ""
except HarborUnavailable as exc:  # pragma: no cover - host without harbor
    HARBOR_OK = False
    HARBOR_WHY = str(exc)

pytestmark = pytest.mark.skipif(
    not HARBOR_OK, reason=f"harbor not importable: {HARBOR_WHY}"
)

TASK_UUID = "2739a678-1759-516d-8ba7-1cd023267ea8"


@pytest.fixture
def task_dir():
    from harness import HARNESS_ROOT

    d = HARNESS_ROOT / "tasks" / TASK_UUID
    if not (d / "task.toml").is_file():
        pytest.skip(f"task dir not present: {d}")
    return d


@pytest.fixture
def run_root(task_dir):
    return resolve_run_root(task_dir)


@pytest.fixture
def cfg(task_dir, run_root):
    return build_base_cfg(task_dir, run_root)


def test_base_cfg_validates_against_real_job_config(cfg):
    """The happy path must clear harbor's own JobConfig without complaint."""
    validate_cfg(cfg)  # must not raise


def test_validate_uses_the_real_harbor_class():
    """Guard against a stub sneaking in: this must be harbor's JobConfig."""
    cls = _load_job_config_cls()
    assert cls.__module__ == "harbor.models.job.config"
    assert cls.__name__ == "JobConfig"
    # Fields we depend on must actually exist upstream.
    assert {"job_name", "jobs_dir", "agents", "tasks", "retry"} <= set(
        cls.model_fields
    )


def test_export_traces_is_rejected(cfg):
    """export_traces is a CLI flag, not a config field. In the JSON it is a no-op,
    so it must be caught here rather than silently ignored at run time."""
    with pytest.raises(HarborConfigError) as exc:
        validate_cfg({**cfg, "export_traces": True})
    assert "export_traces" in str(exc.value)


def test_top_level_model_name_is_rejected(cfg):
    """model_name belongs to agents[], not the top level."""
    with pytest.raises(HarborConfigError) as exc:
        validate_cfg({**cfg, "model_name": "x"})
    assert "model_name" in str(exc.value)


def test_nested_unknown_agent_key_is_rejected(cfg):
    """A typo inside agents[] is just as expensive as one at the top level."""
    bad = json.loads(json.dumps(cfg))
    bad["agents"][0]["modle_name"] = "typo"
    with pytest.raises(HarborConfigError) as exc:
        validate_cfg(bad)
    assert "modle_name" in str(exc.value)


def test_task_without_path_or_name_is_rejected(cfg):
    """TaskConfig requires exactly one of path/name -- pydantic must catch this."""
    from pydantic import ValidationError

    bad = json.loads(json.dumps(cfg))
    bad["tasks"] = [{}]
    with pytest.raises(ValidationError):
        validate_cfg(bad)


def test_agent_carries_model_name_env_and_bridge(cfg):
    agent = cfg["agents"][0]
    assert agent["name"] == "claude-code"
    assert agent["model_name"] == "claude-opus-5"
    assert agent["env"]["ANTHROPIC_BASE_URL"] == "http://172.17.0.1:8765"
    assert agent["env"]["ANTHROPIC_API_KEY"] == "oauth-placeholder"
    # These must be nested, never top-level.
    assert "model_name" not in cfg
    assert "env" not in cfg


def test_retry_block_mirrors_reference_minus_the_no_op(cfg):
    retry = cfg["retry"]
    assert retry["max_retries"] == 1
    assert retry["min_wait_sec"] == 600.0
    assert retry["max_wait_sec"] == 2400.0
    assert retry["wait_multiplier"] == 2.0
    assert retry["include_exceptions"] == ["ApiRateLimitError", "ApiOverloadedError"]
    # ApiUsageLimitError is in harbor's DEFAULT exclude_exceptions, and exclude
    # beats include -- listing it would be a no-op, so it must stay out.
    assert "ApiUsageLimitError" not in retry["include_exceptions"]
    cls = _load_job_config_cls()
    defaults = cls().retry.exclude_exceptions or set()
    assert "ApiUsageLimitError" in defaults


def test_job_shape_matches_reference(cfg, task_dir):
    assert cfg["job_name"] == "agentic"
    assert cfg["n_attempts"] == 1
    assert cfg["n_concurrent_trials"] == 1
    assert cfg["agent_setup_timeout_multiplier"] == 5.0
    assert cfg["tasks"] == [{"path": str(task_dir)}]


def test_jobs_dir_is_under_runs_agentloop(cfg, run_root, task_dir):
    from harness import resolve_task_uuid

    uuid = resolve_task_uuid(task_dir)
    assert cfg["jobs_dir"] == str(run_root / "jobs")
    assert "/runs/agentloop/" in cfg["jobs_dir"]
    assert cfg["jobs_dir"].endswith(f"/runs/agentloop/{uuid}/jobs")


def test_run_root_is_inside_this_harness(task_dir, run_root):
    """The uuid comes from resolve_task_uuid, which falls back to a slug of
    [task].name when task.toml carries no uuid -- as this task's does not."""
    from harness import HARNESS_ROOT, resolve_task_uuid

    assert run_root == HARNESS_ROOT / "runs" / "agentloop" / resolve_task_uuid(task_dir)
    assert run_root.is_relative_to(HARNESS_ROOT / "runs" / "agentloop")


def test_run_root_honours_explicit_harness_root(task_dir, tmp_path):
    from harness import resolve_task_uuid

    root = resolve_run_root(task_dir, harness_root=tmp_path)
    assert root == tmp_path / "runs" / "agentloop" / resolve_task_uuid(task_dir)


def test_cfg_never_points_at_the_production_pipeline_tree(cfg):
    """track3-pipeline is a 65GB read-only production dir. Nothing may write there."""
    assert "track3-pipeline" not in json.dumps(cfg)


def test_cfg_round_trips_through_json_unchanged(cfg):
    """No Path objects, no sets -- it must survive json.dumps/loads untouched."""
    assert json.loads(json.dumps(cfg)) == cfg


def test_serialized_cfg_still_validates(cfg):
    """What we write to disk is what harbor reads; validate that exact object."""
    validate_cfg(json.loads(json.dumps(cfg)))


def test_overrides_flow_through(task_dir, run_root):
    cfg = build_base_cfg(
        task_dir,
        run_root,
        job_name="probe",
        model_name="claude-sonnet-4",
        agent_name="claude-code",
        bridge_url="http://127.0.0.1:9999",
    )
    assert cfg["job_name"] == "probe"
    assert cfg["agents"][0]["model_name"] == "claude-sonnet-4"
    assert cfg["agents"][0]["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"
    validate_cfg(cfg)


def test_build_base_cfg_returns_a_fresh_mutable_cfg(task_dir, run_root):
    """Callers layer overrides on top; shared mutable state would leak between runs."""
    a = build_base_cfg(task_dir, run_root)
    b = build_base_cfg(task_dir, run_root)
    a["agents"][0]["env"]["ANTHROPIC_API_KEY"] = "mutated"
    a["retry"]["include_exceptions"].append("Nope")
    assert b["agents"][0]["env"]["ANTHROPIC_API_KEY"] == "oauth-placeholder"
    assert "Nope" not in b["retry"]["include_exceptions"]


# --------------------------------------------------------------------------- #
# agent selection -- claude-code vs openhands-sdk over the SAME OAuth bridge
# --------------------------------------------------------------------------- #

BRIDGE = "http://172.17.0.1:8765"


def test_agent_env_for_claude_code_is_exactly_the_anthropic_pair():
    """harbor/agents/installed/claude_code.py reads ANTHROPIC_BASE_URL + a key."""
    assert agent_env("claude-code", BRIDGE) == {
        "ANTHROPIC_BASE_URL": BRIDGE,
        "ANTHROPIC_API_KEY": "oauth-placeholder",
    }


def test_agent_env_for_openhands_sdk_is_exactly_the_llm_pair():
    """harbor/agents/installed/openhands_sdk.py reads LLM_BASE_URL + LLM_API_KEY,
    and RAISES ValueError when LLM_API_KEY is unset -- so it is not optional."""
    assert agent_env("openhands-sdk", BRIDGE) == {
        "LLM_BASE_URL": BRIDGE,
        "LLM_API_KEY": "oauth-placeholder",
    }


def test_agent_env_rejects_an_unknown_agent_and_names_the_supported_ones():
    with pytest.raises(ValueError) as exc:
        agent_env("bogus", BRIDGE)
    msg = str(exc.value)
    assert "bogus" in msg
    assert "claude-code" in msg and "openhands-sdk" in msg


def test_agent_env_honours_an_explicit_api_key():
    assert agent_env("openhands-sdk", BRIDGE, api_key="sk-real")["LLM_API_KEY"] == "sk-real"


@pytest.fixture
def openhands_cfg(task_dir, run_root):
    return build_base_cfg(task_dir, run_root, agent_name="openhands-sdk")


def test_openhands_cfg_selects_the_openhands_agent(openhands_cfg):
    assert openhands_cfg["agents"][0]["name"] == "openhands-sdk"


def test_openhands_cfg_carries_llm_env_and_no_anthropic_key_anywhere(openhands_cfg):
    """LiteLLM reads LLM_*; a stray ANTHROPIC_* would be dead weight that reads as
    if the bridge were wired twice."""
    env = openhands_cfg["agents"][0]["env"]
    assert env == {"LLM_BASE_URL": BRIDGE, "LLM_API_KEY": "oauth-placeholder"}
    assert "ANTHROPIC" not in json.dumps(openhands_cfg)


def test_openhands_model_name_gets_the_litellm_provider_prefix(task_dir, run_root):
    """LiteLLM routes by provider prefix: `anthropic/` is what makes it speak the
    Anthropic Messages API, which is exactly what the bridge already serves."""
    cfg = build_base_cfg(task_dir, run_root, agent_name="openhands-sdk",
                         model_name="claude-opus-5")
    assert cfg["agents"][0]["model_name"] == "anthropic/claude-opus-5"


def test_an_already_prefixed_model_name_is_left_alone(task_dir, run_root):
    cfg = build_base_cfg(task_dir, run_root, agent_name="openhands-sdk",
                         model_name="anthropic/claude-sonnet-4")
    assert cfg["agents"][0]["model_name"] == "anthropic/claude-sonnet-4"


def test_claude_code_model_name_stays_bare(task_dir, run_root):
    """The claude CLI wants an unprefixed model; prefixing it would break it. This
    asymmetry with openhands-sdk is deliberate."""
    cfg = build_base_cfg(task_dir, run_root, agent_name="claude-code",
                         model_name="claude-opus-5")
    assert cfg["agents"][0]["model_name"] == "claude-opus-5"


def test_both_agent_configs_pass_the_real_harbor_schema(cfg, openhands_cfg):
    """THE load-bearing one: harbor's own JobConfig must accept both, serialized
    exactly as they will be written to disk."""
    validate_cfg(json.loads(json.dumps(cfg)))
    validate_cfg(json.loads(json.dumps(openhands_cfg)))


def test_the_bridge_url_is_identical_for_both_agents(cfg, openhands_cfg):
    """Same OAuth bridge, different env var names. If these ever diverge, one of
    the two agents is talking to something else."""
    assert cfg["agents"][0]["env"]["ANTHROPIC_BASE_URL"] == BRIDGE
    assert openhands_cfg["agents"][0]["env"]["LLM_BASE_URL"] == BRIDGE


def test_default_agent_is_still_claude_code(cfg):
    """Adding an option must not change what an existing caller gets."""
    assert cfg["agents"][0]["name"] == "claude-code"
    assert cfg["agents"][0]["model_name"] == "claude-opus-5"


def test_setup_timeout_multiplier_defaults_to_five_and_is_tunable(task_dir, run_root):
    """openhands-sdk pip-installs into /opt/openhands-sdk-venv at setup, which is
    slower than claude-code's npm install; raising this must not need an edit."""
    assert build_base_cfg(task_dir, run_root)["agent_setup_timeout_multiplier"] == 5.0
    cfg = build_base_cfg(task_dir, run_root, agent_name="openhands-sdk",
                         setup_timeout_multiplier=12.0)
    assert cfg["agent_setup_timeout_multiplier"] == 12.0
    validate_cfg(cfg)
