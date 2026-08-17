from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any


CURRENT_STATE = "CURRENT_STATE"
FRONTIER = "FRONTIER"


def _marker_re(name: str) -> re.Pattern:
    start = re.escape(f"<!-- AUTO-GENERATED:{name}:START -->")
    end = re.escape(f"<!-- AUTO-GENERATED:{name}:END -->")
    return re.compile(rf"({start})(.*?)({end})", re.DOTALL)


def write_between_markers(path: Path, marker_name: str, new_body: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"target file not found: {path}")
    text = path.read_text()
    pattern = _marker_re(marker_name)
    if not pattern.search(text):
        raise ValueError(f"markers for {marker_name} not found in {path}")
    body = new_body.strip("\n")
    updated = pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    if updated != text:
        path.write_text(updated)


def render_current_state(summary: dict[str, Any]) -> str:
    n = summary.get("n_rows", 0)
    stuck = summary.get("stuck_by_family", {})
    best = summary.get("best_per_variant", {})
    best_step = min(
        (r.get("step_to_3_28") for r in best.values() if r.get("step_to_3_28") is not None),
        default=None,
    )
    lines = [
        f"- **Total attempts recorded**: {n}",
        f"- **Best step_to_3_28**: {best_step if best_step is not None else '—'}",
        f"- **Distinct variants that hit target**: {len(best)}",
        f"- **Stuck counter (by slug-family)**: {dict(stuck) if stuck else '—'}",
    ]
    ruled = summary.get("ruled_down_suggestions", [])
    if ruled:
        joined = ", ".join(f"`{v}`" for v in ruled)
        lines.append(f"- **Suggested ruled-down** ({len(ruled)}): {joined}")
    nf = summary.get("noise_floor")
    if nf:
        lines.append(
            f"- **Noise floor** (n={nf['n']}): step_to_3.28 = {nf['mean_step_to_3_28']} "
            f"± {nf['std_step_to_3_28']}"
        )
    return "\n".join(lines)


def render_frontier_table(summary: dict[str, Any]) -> str:
    front = summary.get("frontier", [])
    if not front:
        return "_(no target hits recorded yet)_"
    lines = [
        "| Rank | Variant | step_to_3_28 | final_val_loss | seed |",
        "|------|---------|--------------|----------------|------|",
    ]
    for i, row in enumerate(front, 1):
        lines.append(
            f"| {i} | `{row.get('variant', '?')}` | "
            f"{row.get('step_to_3_28', '—')} | "
            f"{row.get('final_val_loss', '—')} | "
            f"{row.get('seed', '—')} |"
        )
    return "\n".join(lines)


def update_plan(plan_path: Path, summary: dict[str, Any]) -> None:
    write_between_markers(plan_path, CURRENT_STATE, render_current_state(summary))


def update_goal(goal_path: Path, summary: dict[str, Any]) -> None:
    write_between_markers(goal_path, FRONTIER, render_frontier_table(summary))


def freeze_run_plan(policy_dir: Path, run_num: int) -> Path:
    run_dir = policy_dir / f"run{run_num:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "variants").mkdir(parents=True, exist_ok=True)
    for name in ("plan.md", "goal.md"):
        src = policy_dir / name
        dst = run_dir / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    return run_dir


def link_top_level_plan_to_run(policy_dir: Path, run_num: int) -> None:
    src_plan = policy_dir / f"run{run_num:02d}" / "plan.md"
    src_goal = policy_dir / f"run{run_num:02d}" / "goal.md"
    if src_plan.is_file():
        (policy_dir / "plan.md").write_text(src_plan.read_text())
    if src_goal.is_file():
        (policy_dir / "goal.md").write_text(src_goal.read_text())
