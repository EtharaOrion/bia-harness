#!/usr/bin/env python3
"""Exercise the refinement loop end-to-end without spending LLM tokens.

Everything downstream of the model is real: scaffolding, ledger, summarize,
plan/goal rewriting, tool dispatch, mount, dispatch, grade, ingest. Only the
model itself is replaced, by a fixed script of tool calls that also drives the
loop through its rejection paths (ruled-down reuse, over-long slug stacks) and
the multi-turn subagent path.

    python tools/flow_check.py --task nanogpt-speedrun --workdir /tmp/flowcheck
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS_ROOT / "runner"))
# legacy/harness2 modules are imported flat (`import orchestrator`), not as a package.
sys.path.insert(0, str(HARNESS_ROOT / "legacy" / "harness2"))

from llm_client import LLMResponse  # noqa: E402
from orchestrator import run_loop  # noqa: E402


def _tool(call_id, name, **payload):
    return {"id": call_id, "name": name, "input": payload}


def _variant_src(tag: str) -> str:
    return f"# flow-check variant {tag}\nHYPERPARAM = 1.0\n"


SCRIPT: list[list[dict]] = [
    [
        _tool("a1", "write_variant", filename="iter0.py", contents=_variant_src("iter0")),
        _tool("a2", "append_thread", entry="## iter0\n- Hypothesis: baseline sanity."),
    ],
    [
        _tool("b1", "update_plan_section", section=3, contents="- iter0 established the baseline."),
        _tool("b2", "write_variant", filename="muon_warm_tilt.py", contents=_variant_src("stack3")),
    ],
    [
        _tool("c1", "add_ruled_down", entry="- `muon_warm_tilt` — 2 seeds, no improvement."),
        _tool("c2", "write_variant", filename="muon_warm_tilt.py", contents=_variant_src("dup")),
        _tool("c3", "write_variant", filename="a_b_c_d_e.py", contents=_variant_src("toolong")),
        _tool("c4", "write_variant", filename="iter2.py", contents=_variant_src("iter2")),
    ],
    [
        _tool("d1", "spawn_subagent", topic="Muon second-order preconditioning", kind="idea"),
    ],
    [
        _tool("e1", "write_variant", filename="iter3.py", contents=_variant_src("iter3")),
    ],
]


class ScriptedLLM:
    """Replays SCRIPT one turn per call; subagent calls return canned prose."""

    def __init__(self) -> None:
        self.turn = 0
        self.calls: list[dict] = []

    def messages(self, *, system, messages, tools=None, **kwargs):
        is_subagent = tools is None
        self.calls.append({
            "turn": self.turn,
            "subagent": is_subagent,
            "system_chars": len(system),
            "n_messages": len(messages),
        })
        if is_subagent:
            return LLMResponse(
                text_blocks=["(scripted subagent write-up)"],
                tool_uses=[], stop_reason="end_turn", raw={},
            )
        turn = self.turn
        self.turn += 1
        if turn >= len(SCRIPT):
            return LLMResponse(text_blocks=["done"], tool_uses=[], stop_reason="end_turn", raw={})
        uses = SCRIPT[turn]
        return LLMResponse(
            text_blocks=[], tool_uses=uses, stop_reason="tool_use",
            raw={"content": [{"type": "tool_use", **u} for u in uses]},
        )


def _slug(task_dir: Path) -> str:
    sys.path.insert(0, str(HARNESS_ROOT / "runner"))
    from harness import read_task_id
    return read_task_id(task_dir).replace("/", "-")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="nanogpt-speedrun")
    p.add_argument("--iterations", type=int, default=len(SCRIPT))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--workdir", type=Path, default=Path("/tmp/flowcheck"))
    p.add_argument("--keep", action="store_true", help="Do not wipe --workdir first.")
    args = p.parse_args(argv)

    from harness import resolve_task

    task_dir = resolve_task(args.task)
    root = args.workdir.resolve()
    if root.exists() and not args.keep:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    (root / "policy").mkdir(parents=True, exist_ok=True)
    shutil.copy2(HARNESS_ROOT / "policy" / "AGENTS.md", root / "policy" / "AGENTS.md")

    slug = _slug(task_dir)
    ledger = root / "runs" / slug / "runs.jsonl"
    client = ScriptedLLM()

    rows = run_loop(
        task_dir,
        iterations=args.iterations,
        backend="dry",
        llm_config={"base_url": "scripted://", "api_key": "none", "model": "scripted"},
        seeds_per_iter=args.seeds,
        out_root=root / "runs",
        ledger=ledger,
        harness_root=root,
        llm_client=client,
        install_sigint=False,
    )

    policy_dir = root / "policy" / slug
    variants = sorted((policy_dir / "scratchpad" / "variants").glob("*.py"))
    report = {
        "llm_calls_total": len(client.calls),
        "llm_calls_subagent": sum(1 for c in client.calls if c["subagent"]),
        "ledger_rows": len(rows),
        "distinct_variants_trained": sorted({r["variant"] for r in rows}),
        "variant_files_written": [v.name for v in variants],
        "rewards": [r["reward"] for r in rows],
        "all_success": all(r["status"] == "success" for r in rows),
    }
    print(json.dumps(report, indent=2))

    print("\n--- goal.md 'Ruled down' ---")
    goal = (policy_dir / "goal.md").read_text()
    print(goal[goal.find("## Ruled down"):].strip() or "(empty)")

    print("\n--- plan.md auto block ---")
    plan = (policy_dir / "plan.md").read_text()
    start = plan.find("<!-- AUTO-GENERATED:CURRENT_STATE:START -->")
    end = plan.find("<!-- AUTO-GENERATED:CURRENT_STATE:END -->")
    print(plan[start:end].strip() if start >= 0 else "(markers missing)")

    expected_rejected = {"a_b_c_d_e.py"}
    written = {v.name for v in variants}
    problems = []
    if written & expected_rejected:
        problems.append(f"slug-stack gate failed to reject: {written & expected_rejected}")
    if "muon_warm_tilt.py" not in written:
        problems.append("muon_warm_tilt.py should exist from iteration 1")
    if not report["all_success"]:
        problems.append("some ledger rows are not status=success")
    if report["llm_calls_subagent"] < 1:
        problems.append("subagent path never exercised")

    print("\n--- gate checks ---")
    if problems:
        for m in problems:
            print(f"FAIL: {m}")
        return 1
    print("all gates behaved as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
