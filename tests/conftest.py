import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(HARNESS_ROOT / "runner"))
# legacy/harness2 modules are imported flat (`import orchestrator`), not as a package.
sys.path.insert(0, str(HARNESS_ROOT / "legacy" / "harness2"))
sys.path.insert(0, str(HARNESS_ROOT / "tasks" / "nanogpt-speedrun" / "tests"))
