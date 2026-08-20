"""Build and validate the JSON job-config that drives Harbor programmatically.

`harbor run --config <file.json>` feeds the file straight into
`harbor.models.job.config:JobConfig` (pydantic v2). Validating against that real
class before we ever start a container is what keeps a one-character typo from
costing GPU hours.

Two things about that class are load-bearing and easy to get wrong:

1. `model_name`, `env` and `n_concurrent` live on `agents[]` (AgentConfig), NOT
   at the top level.
2. `JobConfig` declares no `model_config`, so pydantic's default `extra="ignore"`
   applies: harbor SILENTLY DROPS unknown keys instead of rejecting them. A
   misspelled or misplaced key therefore produces a perfectly "valid" config
   that quietly runs with the wrong settings. `validate_cfg` is deliberately
   stricter than harbor here -- it audits every key at every depth against the
   real field sets and raises `HarborConfigError`. See `_assert_no_unknown_keys`.

Shape mirrors the proven reference at track3-pipeline/jobs/agentic.json, except
`jobs_dir` points inside this harness (`runs/agentloop/<task-uuid>/jobs`) so nothing
is ever written into that read-only production tree.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

from harness import HARNESS_ROOT, resolve_task_uuid  # noqa: F401  (re-exported below)

__all__ = [
    "AGENT_ENV_KEYS",
    "HARBOR_PKG_PATH",
    "HarborConfigError",
    "HarborUnavailable",
    "agent_env",
    "build_base_cfg",
    "resolve_run_root",
    "validate_cfg",
]

# harbor lives in its own venv (v0.21.0); our interpreter cannot see it by default.
# Do NOT use `which harbor` -- that resolves to a stale 0.13.2 uv-tool install.
HARBOR_PKG_PATH = "/home/bia-gpu/oer/.venv-harbor/lib/python3.13/site-packages"


# Which (base_url, api_key) env pair each harbor agent actually reads. The first
# two point at the SAME Claude OAuth bridge; only the variable names differ.
#   claude-code   -> harbor/agents/installed/claude_code.py reads ANTHROPIC_*.
#   openhands-sdk -> harbor/agents/installed/openhands_sdk.py reads LLM_* and
#                    RAISES ValueError when LLM_API_KEY is unset, so the key must
#                    be present even though the bridge ignores its value.
#   codex         -> harbor/agents/installed/codex.py is
#                    ModelConnectionSpec(default_provider="openai"), so harbor's
#                    PROVIDERS["openai"] reads OPENAI_*. Its run() writes the key
#                    into $CODEX_HOME/auth.json unconditionally -- same posture as
#                    openhands-sdk, emit it even when the bridge ignores it -- and
#                    it needs its OWN OpenAI-compatible bridge, on port 8788.
AGENT_ENV_KEYS: dict[str, tuple[str, str]] = {
    "claude-code": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"),
    "openhands-sdk": ("LLM_BASE_URL", "LLM_API_KEY"),
    "codex": ("OPENAI_BASE_URL", "OPENAI_API_KEY"),
}

# LiteLLM (which openhands-sdk drives) routes by provider prefix, and `anthropic/`
# is what makes it speak the Anthropic Messages API the bridge already serves. The
# claude CLI wants the model BARE and breaks on a prefix. This asymmetry is
# deliberate -- do not "harmonize" the two.
_LITELLM_PROVIDER_PREFIX = "anthropic/"
_PREFIXED_MODEL_AGENTS = frozenset({"openhands-sdk"})


class HarborUnavailable(RuntimeError):
    """Harbor's pydantic models could not be imported, so nothing can be validated.

    Callers and tests are expected to catch this and skip rather than fall back to
    an unvalidated config.
    """


class HarborConfigError(ValueError):
    """The config carries keys harbor does not define.

    Harbor itself would ignore them; we refuse them, because an ignored key means
    the setting you thought you set is not in effect.
    """


_JOB_CONFIG_CLS: type | None = None


def _harbor_pkg_path() -> Path:
    """Locate harbor's site-packages, preferring the sibling venv layout."""
    explicit = Path(HARBOR_PKG_PATH)
    if explicit.is_dir():
        return explicit
    for root in (Path.home() / "oer" / ".venv-harbor", HARNESS_ROOT.parent / ".venv-harbor"):
        for lib in sorted(root.glob("lib/python3.*/site-packages")):
            if lib.is_dir():
                return lib
    return explicit


def _load_job_config_cls() -> type:
    """Import harbor's real `JobConfig`, putting its site-packages on sys.path.

    Idempotent: the path is inserted at most once and the class is cached.

    Raises:
        HarborUnavailable: if harbor cannot be imported.
    """
    global _JOB_CONFIG_CLS
    if _JOB_CONFIG_CLS is not None:
        return _JOB_CONFIG_CLS

    pkg_path = str(_harbor_pkg_path())
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

    try:
        from harbor.models.job.config import JobConfig
    except Exception as exc:  # ImportError, but also any transitive failure
        raise HarborUnavailable(
            f"cannot import harbor.models.job.config.JobConfig from {pkg_path!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    _JOB_CONFIG_CLS = JobConfig
    return JobConfig


def resolve_run_root(task_dir: Path, harness_root: Path = HARNESS_ROOT) -> Path:
    """Per-task run root: `<harness_root>/runs/agentloop/<task-uuid>`."""
    return Path(harness_root) / "runs" / "agentloop" / resolve_task_uuid(Path(task_dir))


def agent_env(
    agent_name: str, bridge_url: str, api_key: str = "oauth-placeholder"
) -> dict[str, str]:
    """The two env vars `agent_name` needs to reach the OAuth bridge.

    Raises:
        ValueError: `agent_name` is not one of `AGENT_ENV_KEYS`.
    """
    try:
        url_key, key_key = AGENT_ENV_KEYS[agent_name]
    except KeyError:
        raise ValueError(
            f"unsupported agent {agent_name!r}; "
            f"supported agents: {', '.join(sorted(AGENT_ENV_KEYS))}"
        ) from None
    return {url_key: bridge_url, key_key: api_key}


def _agent_model_name(agent_name: str, model_name: str) -> str:
    """Add LiteLLM's provider prefix for the agents that route through it."""
    if agent_name in _PREFIXED_MODEL_AGENTS and "/" not in model_name:
        return f"{_LITELLM_PROVIDER_PREFIX}{model_name}"
    return model_name


def build_base_cfg(
    task_dir: Path,
    run_root: Path,
    *,
    job_name: str = "agentic",
    model_name: str = "claude-opus-5",
    agent_name: str = "claude-code",
    bridge_url: str = "http://172.17.0.1:8765",
    setup_timeout_multiplier: float = 5.0,
) -> dict[str, Any]:
    """Build the base JobConfig dict, in the shape proven by agentic.json.

    The result is plain JSON-serializable data (str, not Path) and is freshly
    constructed on every call, so callers can layer overrides without leaking
    state between runs.
    """
    return {
        "job_name": job_name,
        "jobs_dir": str(Path(run_root) / "jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        # openhands-sdk pip-installs into /opt/openhands-sdk-venv at setup, slower
        # than claude-code's npm install; raise this rather than editing the module.
        "agent_setup_timeout_multiplier": setup_timeout_multiplier,
        "agents": [
            {
                "name": agent_name,
                # model_name/env belong to the agent, never to the job.
                "model_name": _agent_model_name(agent_name, model_name),
                "env": agent_env(agent_name, bridge_url),
            }
        ],
        "tasks": [{"path": str(task_dir)}],
        "retry": {
            "max_retries": 1,
            # The reference also lists ApiUsageLimitError, but harbor's DEFAULT
            # retry.exclude_exceptions already contains it and exclude takes
            # precedence over include -- listing it is a pure no-op. Omitted so
            # the config states only what actually takes effect.
            "include_exceptions": ["ApiRateLimitError", "ApiOverloadedError"],
            "min_wait_sec": 600.0,
            "max_wait_sec": 2400.0,
            "wait_multiplier": 2.0,
        },
    }


def _assert_no_unknown_keys(raw: Any, model: Any, path: str = "") -> None:
    """Recursively reject keys harbor does not define.

    Walks the raw dict alongside the *validated* model instance: known keys map to
    validated attributes (recursing into nested models and lists), and anything in
    the raw dict without a matching field is an unknown key. This catches typos at
    any depth -- `agents[0].modle_name` as well as top-level `export_traces`.
    """
    from pydantic import BaseModel

    if isinstance(raw, dict) and isinstance(model, BaseModel):
        fields = type(model).model_fields
        unknown = sorted(k for k in raw if k not in fields)
        if unknown:
            where = path or "<top-level>"
            raise HarborConfigError(
                f"unknown key(s) at {where}: {', '.join(unknown)}. "
                f"{type(model).__name__} defines no such field, so harbor would "
                f"silently ignore it. Valid fields: {', '.join(sorted(fields))}"
            )
        for key, value in raw.items():
            _assert_no_unknown_keys(
                value, getattr(model, key, None), f"{path}.{key}" if path else key
            )
    elif isinstance(raw, list) and isinstance(model, list):
        for i, (item, sub) in enumerate(zip(raw, model)):
            _assert_no_unknown_keys(item, sub, f"{path}[{i}]")


def validate_cfg(cfg: dict) -> None:
    """Validate `cfg` against harbor's real JobConfig.

    Pydantic's `ValidationError` propagates untouched (wrong types, a TaskConfig
    with neither `path` nor `name`, agent concurrency conflicts, ...).

    Additionally raises `HarborConfigError` for keys harbor does not define.
    Harbor would ignore those; an ignored key is a setting that silently never
    took effect, which is exactly the failure this module exists to prevent
    (e.g. `export_traces` is a CLI flag, and `model_name` is an agent field).

    Raises:
        HarborUnavailable: harbor is not importable.
        HarborConfigError: the config carries unknown keys.
        pydantic.ValidationError: the config violates harbor's schema.
    """
    job_config_cls = _load_job_config_cls()
    validated = job_config_cls.model_validate(cfg)
    _assert_no_unknown_keys(copy.deepcopy(cfg), validated)
