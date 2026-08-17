import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(HARNESS_ROOT / "runner"))
# legacy_planner modules are still imported flat, not as legacy_planner.*
sys.path.insert(0, str(HARNESS_ROOT / "runner" / "legacy_planner"))
sys.path.insert(0, str(HARNESS_ROOT / "tasks" / "nanogpt-speedrun" / "tests"))
