#!/usr/bin/env python3
"""Full 7-stage feedback loop demo — no live LLM or Codex required.

Runs `orchestrator.run_loop` end-to-end on `nanogpt-speedrun` with:
  - a **scripted planner** (returns `write_variant` tool_use per turn; no HTTP)
  - **backend=dry** (synthetic training log; no GPU)
  - **precomputed rubric judgements** (both rub concerns -> pass; no Codex judge)

Produces the full on-disk tree under:
  work/nanogpt-track3-speedrun/run{01,02}/
    variant.py, plan.md, goal.md, summary.md, summary.json, attempt_meta.json
    diff/variant.patch
    planner/trajectory.jsonl
    seed_{0,1}/{log, reward.json, trajectory.json, task/, verifier/*}
  policy/nanogpt-track3-speedrun/run{01,02,03}/
    plan.md, goal.md, variants/iter{0,1}.py

Usage:
    python scripts/demo_flow.py                  # clean prior demo state, run 2x2
    python scripts/demo_flow.py --attempts 3     # 3 attempts
    python scripts/demo_flow.py --keep           # do not wipe prior work/ tree
    python scripts/demo_flow.py --no-verifier    # skip BIA (fastest)

Exit code 0 on success. Prints a directory tree summary at the end.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

HARNESS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(HARNESS_ROOT / "runner"))

from runner import orchestrator  # noqa: E402
from runner.harness import resolve_task, resolve_work_root, resolve_policy_root, resolve_ledger_path, read_task_id  # noqa: E402
from runner.llm_client import LLMResponse  # noqa: E402


DEFAULT_TASK = "nanogpt-speedrun"


class ScriptedPlanner:
    """Duck-typed LLMClient. Returns one write_variant tool_use per call."""

    def __init__(self) -> None:
        self.call_count = 0

    def messages(self, *, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None,
                 max_tokens: int = 4096,
                 temperature: float | None = None) -> LLMResponse:
        i = self.call_count
        self.call_count += 1
        filename = f"iter{i}.py"
        contents = (
            f"# Demo variant iter={i} produced by scripts/demo_flow.py ScriptedPlanner\n"
            f"# In production this file would be trainer source authored by the planner LLM.\n"
            f"def train():\n"
            f"    # backend=dry ignores this body and fabricates a synthetic log\n"
            f"    return {{'iter': {i}, 'source': 'demo_flow.ScriptedPlanner'}}\n"
        )
        tool_use = {
            "id": f"toolu_demo_{i:03d}",
            "name": "write_variant",
            "input": {"filename": filename, "contents": contents},
        }
        raw = {
            "content": [{"type": "tool_use", **tool_use}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 100 + i, "output_tokens": 50 + i},
            "model": "scripted-fake-planner-v1",
        }
        return LLMResponse(text_blocks=[], tool_uses=[tool_use],
                           stop_reason="tool_use", raw=raw)


def write_precomputed_judgements(path: Path) -> Path:
    """Write a dict-keyed judgements JSON that bia_verifier.cli grade will accept."""
    path.write_text(json.dumps({
        "rub.constraint_compliance": {
            "verdict": "pass",
            "rationale": (
                "Demo run: scripted planner produced trivial variant, no real "
                "modification to trainer sections; Track-3 hard constraints are "
                "vacuously satisfied since no forbidden edit occurred."
            ),
        },
        "rub.stat_sig_communicated": {
            "verdict": "pass",
            "rationale": (
                "Demo run: synthetic dry-backend log reports the standard "
                "step:S/N val_loss:V lines; stat-sig column is emitted by the "
                "task grader as stat_sig_at_n1 in reward.json."
            ),
        },
    }, indent=2, sort_keys=True) + "\n")
    return path


def clean_state(task_dir: Path) -> None:
    task_id = read_task_id(task_dir)
    work = resolve_work_root(task_dir)
    policy = resolve_policy_root(task_dir)
    ledger = resolve_ledger_path(task_id)
    for p in (work, policy, ledger):
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            print(f"[demo] cleaned: {p}")


def print_tree(root: Path, prefix: str = "", max_depth: int = 5, depth: int = 0) -> None:
    if depth >= max_depth or not root.exists():
        return
    entries = sorted(root.iterdir())
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if entry.is_dir() else ""
        print(f"{prefix}{connector}{entry.name}{suffix}")
        if entry.is_dir():
            ext = "    " if is_last else "│   "
            print_tree(entry, prefix + ext, max_depth, depth + 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default=DEFAULT_TASK, help="Task dir name (default nanogpt-speedrun)")
    p.add_argument("--attempts", type=int, default=2, help="Iterations to run (default 2)")
    p.add_argument("--seeds", type=int, default=2, help="Seeds per attempt (default 2)")
    p.add_argument("--keep", action="store_true", help="Do NOT wipe prior work/policy/ledger state")
    p.add_argument("--no-verifier", action="store_true", help="Skip BIA verification (fastest)")
    args = p.parse_args(argv)

    task_dir = resolve_task(args.task)
    print(f"[demo] task_dir  = {task_dir}")

    if not args.keep:
        clean_state(task_dir)

    judgements_path = HARNESS_ROOT / "work" / "_demo_judgements.json"
    judgements_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_verifier:
        write_precomputed_judgements(judgements_path)
        print(f"[demo] wrote judgements at {judgements_path}")

    planner = ScriptedPlanner()
    llm_config = {
        "base_url": "http://scripted.invalid",
        "api_key": "scripted-fake-key",
        "model": "scripted-fake-planner-v1",
    }

    print(f"[demo] running orchestrator.run_loop iterations={args.attempts} "
          f"seeds_per_iter={args.seeds} backend=dry "
          f"verifier_enabled={not args.no_verifier}")
    rows = orchestrator.run_loop(
        task_dir,
        iterations=args.attempts,
        backend="dry",
        llm_config=llm_config,
        agent="claude-code",
        seeds_per_iter=args.seeds,
        llm_client=planner,
        install_sigint=False,
        verifier_enabled=not args.no_verifier,
        precomputed_judgements=judgements_path if not args.no_verifier else None,
    )

    task_id = read_task_id(task_dir)
    work_root = resolve_work_root(task_dir)
    policy_root = resolve_policy_root(task_dir)
    ledger_path = resolve_ledger_path(task_id)

    print(f"\n[demo] DONE: {len(rows)} ledger row(s) written")
    print(f"[demo] planner LLM invocations: {planner.call_count}")
    print(f"[demo] ledger: {ledger_path}")
    for r in rows:
        status = r.get("status")
        reward = r.get("reward")
        cr = r.get("consolidated_reward")
        vc = r.get("verification_complete")
        print(f"[demo]   attempt={r.get('attempt')} seed={r.get('seed')} "
              f"status={status} reward={reward} consolidated_reward={cr} "
              f"verification_complete={vc}")

    print(f"\n[demo] work tree at {work_root}:")
    print_tree(work_root, max_depth=4)
    print(f"\n[demo] policy tree at {policy_root}:")
    print_tree(policy_root, max_depth=3)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
