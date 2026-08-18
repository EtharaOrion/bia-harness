from __future__ import annotations

import tomllib
from pathlib import Path

try:
    from harness import HARNESS_ROOT, resolve_policy_root
except ImportError:
    from runner.harness import HARNESS_ROOT, resolve_policy_root


PLAN_TEMPLATE = """\
# Plan — {slug}

## 1. Current state

<!-- AUTO-GENERATED:CURRENT_STATE:START -->
_(no runs yet)_
<!-- AUTO-GENERATED:CURRENT_STATE:END -->

## 2. Active picklist (LLM-owned)

_Append candidates you want to try next, in priority order. Format: `- <slug> — <one-line rationale>`._

## 3. Lessons (LLM-owned, append-only)

_One line per iteration insight. Format: `- <YYYY-MM-DD> <slug> — <what you learned>`._
"""


GOAL_TEMPLATE = """\
# Goal — {slug}

{mission}

## Frontier

<!-- AUTO-GENERATED:FRONTIER:START -->
_(no target hits recorded yet)_
<!-- AUTO-GENERATED:FRONTIER:END -->

## Combos worth trying (LLM-owned)

_Append parent × modification stacks here as you rank them._

## Ruled down (LLM-owned, append-only)

_Add variants that failed under ≥ 2 seeds. Include N and margin so a future iteration doesn't retry._
"""


THREAD_TEMPLATE = """\
# Thread — {slug}

Durable narrative log. Append one entry per iteration so a fresh orchestrator can recover state after context compaction.

Entry format:

```
## <YYYY-MM-DD HH:MM UTC> — iter <N> — <slug>

- **Hypothesis**: <one-line>
- **Result**: <one-line — step_to_3_28 / reward / hit_target>
- **Insight**: <one-line — what this changes for next iteration>
```
"""


PICKLIST_TEMPLATE = "# Picklist — {slug}\n\nShort-list of the next 3-5 variants under active consideration. Overwritten each iteration.\n"


AUDITS_TEMPLATE = "# Audits — {slug}\n\nAppend-only log of rule-compliance checks (integrity/novelty/noise-floor pre-flights).\n"


def _read_task_meta(task_dir: Path) -> tuple[str, str]:
    with (task_dir / "task.toml").open("rb") as f:
        data = tomllib.load(f)
    task_block = data.get("task", {}) if isinstance(data.get("task"), dict) else {}
    slug = task_block.get("name") or data.get("name") or task_dir.name
    description = (
        task_block.get("description")
        or data.get("description")
        or f"Run task `{slug}` and iterate toward the highest reward per the task's own instruction.md."
    )
    return slug, description


def ensure_policy_scaffolded(task_dir: Path, harness_root: Path = HARNESS_ROOT) -> Path:
    policy_dir = resolve_policy_root(task_dir, harness_root=harness_root)
    plan_path = policy_dir / "plan.md"
    if plan_path.is_file():
        return policy_dir

    slug, description = _read_task_meta(task_dir)
    scratchpad = policy_dir / "scratchpad"
    variants = scratchpad / "variants"
    variants.mkdir(parents=True, exist_ok=True)

    fmt = {"slug": slug, "mission": description}
    (policy_dir / "plan.md").write_text(PLAN_TEMPLATE.format(**fmt))
    (policy_dir / "goal.md").write_text(GOAL_TEMPLATE.format(**fmt))
    (scratchpad / "THREAD.md").write_text(THREAD_TEMPLATE.format(**fmt))
    (scratchpad / "picklist.md").write_text(PICKLIST_TEMPLATE.format(**fmt))
    (scratchpad / "audits.md").write_text(AUDITS_TEMPLATE.format(**fmt))
    (variants / ".gitkeep").touch()
    return policy_dir
