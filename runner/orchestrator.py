from __future__ import annotations

import json
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness import (
    HARNESS_ROOT,
    read_task_id,
    resolve_ledger_path,
    resolve_policy_root,
    resolve_task,
)
from harness import run as harness_run
from llm_client import LLMClient
from plan_writer import update_goal, update_plan
from scaffold_policy import ensure_policy_scaffolded
from summarize import load_ledger, summarize


MAX_TURNS_PER_ITER = 4
STUCK_THRESHOLD = 15
MAX_SLUG_SEGMENTS = 4


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "write_variant",
        "description": "Write the Python source for the next attempt. Filename must be either 'iterN.py' (iteration-numbered, always allowed) or a slug-stack name with at most 4 underscore-segments (Rule 4). Cannot reuse a slug already in goal.md's 'Ruled down' list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "contents": {"type": "string"},
            },
            "required": ["filename", "contents"],
        },
    },
    {
        "name": "append_thread",
        "description": "Append one narrative entry to scratchpad/THREAD.md.",
        "input_schema": {
            "type": "object",
            "properties": {"entry": {"type": "string"}},
            "required": ["entry"],
        },
    },
    {
        "name": "update_plan_section",
        "description": "Append LLM-owned content to plan.md, labeled with section number (2=picklist, 3=lesson) and an ISO timestamp. Does not touch AUTO-GENERATED blocks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "section": {"type": "integer", "enum": [2, 3]},
                "contents": {"type": "string"},
            },
            "required": ["section", "contents"],
        },
    },
    {
        "name": "add_ruled_down",
        "description": "Append a one-line variant entry under 'Ruled down' in goal.md. After this, write_variant rejects that slug.",
        "input_schema": {
            "type": "object",
            "properties": {"entry": {"type": "string"}},
            "required": ["entry"],
        },
    },
    {
        "name": "spawn_subagent",
        "description": "Spawn a nested LLM for a deep write-up. kind='paper' does an 800-2000w structured read; kind='idea' does a 500-1500w 7-part optimizer-idea analysis. Result is returned as tool_result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "kind": {"type": "string", "enum": ["paper", "idea"]},
                "notes": {"type": "string"},
            },
            "required": ["topic", "kind"],
        },
    },
]


SUBAGENT_PROMPTS = {
    "paper": (
        "You are a subagent invoked by an autonomous optimizer researcher. "
        "Read the paper deeply, not skimmingly. Return 800-2000 words: "
        "key equations verbatim, exact hyperparameters, ablations, failure cases, "
        "and END with a paragraph titled '**To port to our setup:**' explaining "
        "concrete changes needed for the modded-nanogpt Track-3 substrate."
    ),
    "idea": (
        "You are a subagent invoked by an autonomous optimizer researcher. "
        "Analyze this optimizer idea in 500-1500 words using the 7-part template: "
        "(1) math derivation, (2) intuition for this regime, (3) web search of priors "
        "with >=3 references, (4) improvement vector, (5) ablation plan (baseline+kill-cell), "
        "(6) failure-mode prediction, (7) go/no-go recommendation."
    ),
}


_RULED_DOWN_HEADING_RE = re.compile(r"^##\s+Ruled\s+down.*$", re.IGNORECASE | re.MULTILINE)
_ITER_FILENAME_RE = re.compile(r"^iter\d+$")
_BACKTICK_SLUG_RE = re.compile(r"`([A-Za-z0-9_\-]+)`")
_BULLET_SLUG_RE = re.compile(r"^-\s+([A-Za-z0-9_\-]+)")


def _parse_ruled_down(goal_md_path: Path) -> set[str]:
    if not goal_md_path.is_file():
        return set()
    text = goal_md_path.read_text()
    m = _RULED_DOWN_HEADING_RE.search(text)
    if not m:
        return set()
    body = text[m.end():]
    next_h = re.search(r"^##\s+", body, re.MULTILINE)
    if next_h:
        body = body[: next_h.start()]
    slugs: set[str] = set()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("_"):
            continue
        slugs.update(_BACKTICK_SLUG_RE.findall(line))
        bm = _BULLET_SLUG_RE.match(line)
        if bm:
            slugs.add(bm.group(1))
    return slugs


def _validate_variant_filename(filename: str) -> None:
    if "/" in filename or "\\" in filename or filename.startswith(".") or ".." in filename:
        raise ValueError(f"variant filename must be a bare name with no path separators: {filename!r}")
    if not filename.endswith(".py"):
        raise ValueError(f"variant filename must end with .py: {filename!r}")
    stem = filename[:-3]
    if _ITER_FILENAME_RE.fullmatch(stem):
        return
    segments = stem.split("_")
    if len(segments) > MAX_SLUG_SEGMENTS:
        raise ValueError(
            f"variant filename exceeds slug-stack of {MAX_SLUG_SEGMENTS} underscore-segments: "
            f"{filename!r}. Start a new family or bump to iterN.py."
        )


def build_system_prompt(task_dir: Path, policy_dir: Path, harness_root: Path = HARNESS_ROOT) -> str:
    parts: list[str] = []
    agents_md = harness_root / "policy" / "AGENTS.md"
    if agents_md.is_file():
        parts.append("# Constitution (shared)\n\n" + agents_md.read_text())
    instr = task_dir / "instruction.md"
    if instr.is_file():
        parts.append("# Task instruction\n\n" + instr.read_text())
    goal = policy_dir / "goal.md"
    plan = policy_dir / "plan.md"
    if goal.is_file():
        parts.append("# Current goal (auto-refreshed)\n\n" + goal.read_text())
    if plan.is_file():
        parts.append("# Current plan (auto-refreshed)\n\n" + plan.read_text())
    return "\n\n---\n\n".join(parts)


def _stuck_signal(summary: dict) -> str | None:
    stuck = summary.get("stuck_by_family", {}) or {}
    hot = {f: n for f, n in stuck.items() if n >= STUCK_THRESHOLD}
    if not hot:
        return None
    return (
        f"STUCK-DETECTOR: families with {STUCK_THRESHOLD}+ runs without improvement: "
        f"{hot}. Pivot to a new slug-family this iteration."
    )


def build_user_message(iteration: int, prev_row: dict | None = None, stuck_signal: str | None = None) -> str:
    lines = [f"# Iteration {iteration}"]
    if stuck_signal:
        lines += ["", f"**{stuck_signal}**"]
    if prev_row is None:
        lines += [
            "",
            "This is the first attempt for this task in this run. Read the shared constitution and task instruction above.",
            f"Propose your first candidate and write it via the `write_variant` tool. Filename: `iter{iteration}.py`.",
        ]
    else:
        lines += [
            "",
            "## Previous iteration result",
            "",
            "```json",
            json.dumps(prev_row, indent=2, sort_keys=True),
            "```",
            "",
            "## Your tasks this iteration",
            "",
            "1. Reflect on the result. If insightful, call `append_thread` and/or `update_plan_section`.",
            "2. If a variant should be ruled down (>=2 seeds, hit_target=false), call `add_ruled_down`.",
            f"3. Propose the next candidate. Call `write_variant` with filename `iter{iteration}.py` (or a slug-stack name <= 4 underscore-segments).",
            "4. If you need to read a paper deeply or ablate an idea, call `spawn_subagent` first. Otherwise finalize with `write_variant`.",
        ]
    return "\n".join(lines)


def execute_spawn_subagent(tool_use: dict, client: LLMClient) -> str:
    inp = tool_use.get("input", {})
    kind = inp["kind"]
    if kind not in SUBAGENT_PROMPTS:
        raise ValueError(f"unknown subagent kind: {kind!r}")
    topic = inp["topic"]
    notes = inp.get("notes", "")
    user_msg = f"# Topic\n\n{topic}\n\n## Notes from parent\n\n{notes}"
    resp = client.messages(
        system=SUBAGENT_PROMPTS[kind],
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=8192,
    )
    return "\n\n".join(resp.text_blocks) or "(subagent returned no text content)"


def execute_tool_use(tool_use: dict, policy_dir: Path) -> Path:
    name = tool_use["name"]
    inp = tool_use.get("input", {})
    if name == "write_variant":
        _validate_variant_filename(inp["filename"])
        stem = Path(inp["filename"]).stem
        ruled = _parse_ruled_down(policy_dir / "goal.md")
        if stem in ruled:
            raise ValueError(
                f"variant slug {stem!r} was already ruled down; propose something different"
            )
        variants_dir = policy_dir / "scratchpad" / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)
        target = variants_dir / inp["filename"]
        target.write_text(inp["contents"])
        return target
    if name == "append_thread":
        thread = policy_dir / "scratchpad" / "THREAD.md"
        thread.parent.mkdir(parents=True, exist_ok=True)
        with thread.open("a") as f:
            f.write("\n" + inp["entry"].rstrip() + "\n")
        return thread
    if name == "update_plan_section":
        section = int(inp["section"])
        if section not in (2, 3):
            raise ValueError(f"update_plan_section: section must be 2 or 3, got {section}")
        plan_path = policy_dir / "plan.md"
        ts = datetime.now(timezone.utc).isoformat()
        with plan_path.open("a") as f:
            f.write(f"\n<!-- LLM \u00a7{section} @ {ts} -->\n{inp['contents'].rstrip()}\n")
        return plan_path
    if name == "add_ruled_down":
        goal_path = policy_dir / "goal.md"
        with goal_path.open("a") as f:
            f.write("\n" + inp["entry"].rstrip() + "\n")
        return goal_path
    raise ValueError(f"unknown tool: {name!r}")


def next_iteration_number(policy_dir: Path) -> int:
    variants_dir = policy_dir / "scratchpad" / "variants"
    if not variants_dir.is_dir():
        return 0
    return sum(1 for p in variants_dir.glob("iter*.py"))


def _install_sigint_handler() -> dict:
    halted = {"flag": False}

    def handler(_signum, _frame):
        halted["flag"] = True
        print("\n[orchestrator] SIGINT received; will halt after current iteration.", file=sys.stderr)

    signal.signal(signal.SIGINT, handler)
    return halted


def run_loop(
    task,
    iterations: int,
    backend: str,
    llm_config: dict,
    *,
    agent: str = "claude_code",
    seeds_per_iter: int = 1,
    out_root: Path | None = None,
    ledger: Path | None = None,
    harness_root: Path = HARNESS_ROOT,
    llm_client: LLMClient | None = None,
    install_sigint: bool = True,
) -> list[dict]:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    task_dir = resolve_task(str(task)) if not isinstance(task, Path) else task.resolve()
    task_id = read_task_id(task_dir)
    ensure_policy_scaffolded(task_dir, harness_root=harness_root)
    policy_dir = resolve_policy_root(task_dir, harness_root=harness_root)
    if ledger is None:
        ledger = resolve_ledger_path(task_id, harness_root=harness_root)
    if out_root is None:
        out_root = harness_root / "runs"

    client = llm_client or LLMClient(
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
        model=llm_config["model"],
        timeout_sec=float(llm_config.get("timeout", 300.0)),
        num_retries=int(llm_config.get("num_retries", 4)),
    )

    halted = _install_sigint_handler() if install_sigint else {"flag": False}

    all_rows: list[dict] = []
    for _ in range(iterations):
        if halted["flag"]:
            break
        i = next_iteration_number(policy_dir)
        summary = summarize(ledger)
        try:
            update_plan(policy_dir / "plan.md", summary)
        except (FileNotFoundError, ValueError):
            pass
        try:
            update_goal(policy_dir / "goal.md", summary)
        except (FileNotFoundError, ValueError):
            pass

        prev_rows = load_ledger(ledger)
        prev_row = prev_rows[-1] if prev_rows else None

        system_prompt = build_system_prompt(task_dir, policy_dir, harness_root=harness_root)
        user_msg = build_user_message(i, prev_row, stuck_signal=_stuck_signal(summary))

        conversation: list[dict] = [{"role": "user", "content": user_msg}]
        variant_path: Path | None = None

        for _turn in range(MAX_TURNS_PER_ITER):
            resp = client.messages(
                system=system_prompt,
                messages=conversation,
                tools=TOOL_SCHEMAS,
            )
            if not resp.tool_uses:
                break

            raw_content = resp.raw.get("content") if isinstance(resp.raw, dict) else None
            if raw_content:
                conversation.append({"role": "assistant", "content": raw_content})

            tool_results: list[dict] = []
            has_subagent = False
            for tu in resp.tool_uses:
                try:
                    if tu["name"] == "spawn_subagent":
                        has_subagent = True
                        result_text = execute_spawn_subagent(tu, client)
                    else:
                        path = execute_tool_use(tu, policy_dir)
                        if tu["name"] == "write_variant":
                            variant_path = path
                        result_text = f"OK: {tu['name']} -> {path.name}"
                except ValueError as e:
                    result_text = f"ERROR: {e}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.get("id", ""),
                    "content": result_text,
                })

            if raw_content:
                conversation.append({"role": "user", "content": tool_results})

            if not has_subagent:
                break

        if variant_path is None:
            print(f"[orchestrator] iter {i}: no write_variant emitted; skipping training call.", file=sys.stderr)
            continue

        rows = harness_run(
            task_dir,
            seeds_per_iter,
            backend,
            out_root,
            variant=variant_path,
            ledger=ledger,
            llm_config=llm_config,
            agent=agent,
        )
        all_rows.extend(rows)

    return all_rows
