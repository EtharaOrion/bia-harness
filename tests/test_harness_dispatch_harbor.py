"""`harness.resolve_harbor_bin` -- one of the four symbols agentloop imports from harness.py.

The legacy dispatch_harbor / build_harbor_cmd / trial-parsing guards that used to
share this file now live in `legacy/harness2/tests/test_harness_dispatchers.py`.
"""
from pathlib import Path

import harness


HARNESS_ROOT = Path(__file__).resolve().parent.parent
NANOGPT_SMOKE = HARNESS_ROOT / "tasks" / "nanogpt-smoke"


def test_resolve_harbor_bin_prefers_explicit_over_path(monkeypatch):
    monkeypatch.setenv("HARBOR_BIN", "/custom/harbor")
    assert harness.resolve_harbor_bin() == "/custom/harbor"


def test_resolve_harbor_bin_avoids_bare_path_when_venv_present(monkeypatch):
    """A bare 'harbor' hits a stale uv-tool install lacking gpus/allowlist support."""
    monkeypatch.delenv("HARBOR_BIN", raising=False)
    resolved = harness.resolve_harbor_bin()
    assert resolved == "harbor" or resolved.endswith(".venv-harbor/bin/harbor")
