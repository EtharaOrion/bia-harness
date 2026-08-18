"""sys.path wiring for the legacy harness2 (planner-authors-the-variant) tests.

These tests moved here from the top-level ``tests/`` tree together with the modules
they exercise. They import those modules FLAT (``import orchestrator``,
``from summarize import ...``) and the modules import each other flat too, so
``legacy/harness2`` has to be on ``sys.path`` -- exactly as ``tests/conftest.py``
and ``runner/harness.py`` do it. Mirrors those inserts so this directory can be
collected on its own (``pytest legacy/harness2/tests``) as well as as part of the
full suite.
"""
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(HARNESS_ROOT / "runner"))
# legacy/harness2 modules are imported flat (`import orchestrator`), not as a package.
sys.path.insert(0, str(HARNESS_ROOT / "legacy" / "harness2"))
sys.path.insert(0, str(HARNESS_ROOT / "tasks" / "nanogpt-speedrun" / "tests"))
