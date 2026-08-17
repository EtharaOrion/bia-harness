from __future__ import annotations

import difflib
import json
import re
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness import (
    HARNESS_ROOT,
    read_task_id,
    resolve_ledger_path,
    resolve_policy_root,
    resolve_task,
    resolve_task_uuid,
    resolve_work_root,
)
from harness import run as harness_run
from llm_client import LLMClient
from plan_writer import freeze_run_plan, update_goal, update_plan
from scaffold_policy import ensure_policy_scaffolded
from summarize import load_ledger, summarize, write_attempt_summary


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
            "1. Reflect on the result. If insightful, call `append_thread` and/or `update_plan_section`. Keep entries small: THREAD <= 3 short lines (Hypothesis / Result / Insight), plan section 3 lesson <= 1 line.",
            "2. If a variant should be ruled down (>=2 seeds, hit_target=false), call `add_ruled_down`.",
            f"3. Propose the next candidate. Call `write_variant` with filename `iter{iteration}.py` (or a slug-stack name <= 4 underscore-segments).",
            "4. If you need to read a paper deeply or ablate an idea, call `spawn_subagent` first. Otherwise finalize with `write_variant`.",
        ]
    return "\n".join(lines)


def build_user_message_from_run(iteration: int,
                                prev_summary_md: str | None = None,
                                prev_variant_py: str | None = None,
                                stuck_signal: str | None = None) -> str:
    lines = [f"# Iteration {iteration}"]
    if stuck_signal:
        lines += ["", f"**{stuck_signal}**"]
    if prev_summary_md is None:
        lines += [
            "",
            "This is the first attempt for this task in this run. Read the shared constitution and task instruction above.",
            f"Propose your first candidate and write it via the `write_variant` tool. Filename: `iter{iteration}.py`.",
        ]
        return "\n".join(lines)
    lines += [
        "",
        "## Previous attempt summary (from work/<uuid>/run<N-1>/summary.md)",
        "",
        prev_summary_md.rstrip(),
        "",
    ]
    if prev_variant_py is not None:
        lines += [
            "## Previous attempt variant source (from work/<uuid>/run<N-1>/variant.py)",
            "",
            "```python",
            prev_variant_py.rstrip(),
            "```",
            "",
        ]
    lines += [
        "## Your tasks this iteration",
        "",
        "1. Reflect on the result. If insightful, call `append_thread` and/or `update_plan_section`. Keep entries small: THREAD <= 3 short lines (Hypothesis / Result / Insight), plan section 3 lesson <= 1 line.",
        "2. If a variant should be ruled down (>=2 seeds, hit_target=false), call `add_ruled_down`.",
        f"3. Propose the next candidate. Call `write_variant` with filename `iter{iteration}.py` (or a slug-stack name <= 4 underscore-segments).",
        "4. If you need to read a paper deeply or ablate an idea, call `spawn_subagent` first. Otherwise finalize with `write_variant`.",
    ]
    return "\n".join(lines)


def _append_trajectory(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str, sort_keys=True) + "\n")


def _read_prior_attempt(work_root: Path, run_num: int) -> tuple[str | None, str | None]:
    if run_num <= 1:
        return None, None
    prev = work_root / f"run{(run_num - 1):02d}"
    summary_md = prev / "summary.md"
    variant_py = prev / "variant.py"
    md_text = summary_md.read_text() if summary_md.is_file() else None
    py_text = variant_py.read_text() if variant_py.is_file() else None
    return md_text, py_text


def _write_variant_diff(work_root: Path, run_num: int, current_variant: Path) -> Path:
    """SPEC_AUDIT §6.2: emit work/<uuid>/run<N>/diff/variant.patch.

    Unified diff against the prior attempt's variant.py (empty for attempt 1). No git
    dependency: uses stdlib difflib so this works in the same environments the harness
    already runs in.
    """
    diff_dir = work_root / f"run{run_num:02d}" / "diff"
    diff_dir.mkdir(parents=True, exist_ok=True)
    prev = work_root / f"run{(run_num - 1):02d}" / "variant.py"
    prev_text = prev.read_text() if prev.is_file() else ""
    curr_text = current_variant.read_text() if current_variant.is_file() else ""
    prev_lines = prev_text.splitlines(keepends=True)
    curr_lines = curr_text.splitlines(keepends=True)
    patch = "".join(difflib.unified_diff(
        prev_lines, curr_lines,
        fromfile=f"run{(run_num - 1):02d}/variant.py",
        tofile=f"run{run_num:02d}/variant.py",
        n=3,
    ))
    out = diff_dir / "variant.patch"
    out.write_text(patch)
    return out


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


def _write_loop_terminated(work_root: Path, reason: str, message: str, *,
                            attempts_completed: int, last_run_num: int | None,
                            extras: dict | None = None) -> Path:
    payload = {
        "terminated_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "message": message,
        "attempts_completed": attempts_completed,
        "last_run_num": last_run_num,
    }
    if extras:
        payload.update(extras)
    path = work_root / "loop_terminated.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def run_loop(
    task,
    iterations: int,
    backend: str,
    llm_config: dict,
    *,
    agent: str = "claude-code",
    seeds_per_iter: int = 1,
    out_root: Path | None = None,
    ledger: Path | None = None,
    harness_root: Path = HARNESS_ROOT,
    llm_client: LLMClient | None = None,
    install_sigint: bool = True,
    concurrent: int = 1,
    jobs_dir: Path | None = None,
    verifier_enabled: bool = True,
    codex_bridge_url: str = "http://127.0.0.1:8788",
    codex_bridge_key: str | None = None,
    judge_model: str = "gpt5.6-sol",
    precomputed_judgements: Path | None = None,
    target_reward: float | None = None,
    max_wall_clock_sec: float | None = None,
    max_input_tokens: int | None = None,
    k_consecutive_incomplete: int | None = None,
    baseline_attempt_1: bool = False,
    retry_count: int = 0,
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

    work_root = resolve_work_root(task_dir, harness_root=harness_root)
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"[harness] task={task_id!r}  iterations={iterations}  backend={backend}  "
          f"seeds_per_iter={seeds_per_iter}  work_root={work_root}", file=sys.stderr, flush=True)

    all_rows: list[dict] = []
    attempts_completed = 0
    last_run_num: int | None = None
    cumulative_input_tokens = 0
    consecutive_incomplete = 0
    loop_start_wall = time.monotonic()
    terminate_reason: str | None = None
    terminate_message: str = ""
    terminate_extras: dict = {}

    for loop_idx in range(iterations):
        if halted["flag"]:
            terminate_reason = "sigint"
            terminate_message = "SIGINT received; loop halted between attempts."
            break
        if max_wall_clock_sec is not None and (time.monotonic() - loop_start_wall) >= max_wall_clock_sec:
            terminate_reason = "wall_clock_exceeded"
            terminate_message = (
                f"cumulative wall-clock {time.monotonic() - loop_start_wall:.1f}s "
                f">= --max-wall-clock-sec {max_wall_clock_sec}"
            )
            break
        if max_input_tokens is not None and cumulative_input_tokens >= max_input_tokens:
            terminate_reason = "token_budget_exceeded"
            terminate_message = (
                f"cumulative planner input_tokens {cumulative_input_tokens} "
                f">= --max-input-tokens {max_input_tokens}"
            )
            break
        run_num = loop_idx + 1
        print(f"[harness] === attempt {run_num}/{iterations} START ===",
              file=sys.stderr, flush=True)

        run_dir_policy = policy_dir / f"run{run_num:02d}"
        if not (run_dir_policy / "plan.md").exists():
            freeze_run_plan(policy_dir, run_num)

        run_dir_work = work_root / f"run{run_num:02d}"
        run_dir_work.mkdir(parents=True, exist_ok=True)
        (run_dir_work / "planner").mkdir(parents=True, exist_ok=True)
        for name in ("plan.md", "goal.md"):
            src = run_dir_policy / name
            dst = run_dir_work / name
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)

        prev_summary_md, prev_variant_py = _read_prior_attempt(work_root, run_num)

        if baseline_attempt_1 and run_num == 1:
            baseline_started = datetime.now(timezone.utc).isoformat()
            baseline_stub = run_dir_work / "variant.py"
            baseline_stub.write_text(
                "# ATTEMPT 1 BASELINE (SPEC_AUDIT R9)\n"
                "# Planner intentionally skipped; the raw trainer executes as-is.\n"
                "# Consumed by attempts 2+ as the reference point for improvement.\n"
            )
            _append_trajectory(run_dir_work / "planner" / "trajectory.jsonl", {
                "turn": 0,
                "role": "baseline",
                "note": "R9: attempt 1 baseline; no planner turn issued.",
                "timestamp_utc": baseline_started,
            })
            _write_variant_diff(work_root, run_num, baseline_stub)
            rows = harness_run(
                            task_dir, seeds_per_iter, backend, out_root,
                            variant=None, ledger=ledger, llm_config=llm_config, agent=agent,
                            concurrent=concurrent, attempt=run_num,
                            attempt_dir=run_dir_work, jobs_dir=jobs_dir,
                            verifier_enabled=verifier_enabled,
                            codex_bridge_url=codex_bridge_url,
                            codex_bridge_key=codex_bridge_key,
                            judge_model=judge_model,
                            precomputed_judgements=precomputed_judgements,
                            retry_count=retry_count,
                        )
            baseline_finished = datetime.now(timezone.utc).isoformat()
            (run_dir_work / "attempt_meta.json").write_text(json.dumps({
                "attempt": run_num,
                "baseline": True,
                "started_at": baseline_started,
                "finished_at": baseline_finished,
                "model": None,
                "proxy_base_url": llm_config.get("base_url"),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "stop_reason": "baseline_no_planner",
                "turns": 0,
            }, indent=2, sort_keys=True, default=str) + "\n")
            cumulative_rows = load_ledger(ledger)
            write_attempt_summary(run_dir_work, task_id, run_num, rows, cumulative_rows)
            summary_post = summarize(ledger)
            try:
                update_plan(policy_dir / "plan.md", summary_post)
            except (FileNotFoundError, ValueError):
                pass
            try:
                update_goal(policy_dir / "goal.md", summary_post)
            except (FileNotFoundError, ValueError):
                pass
            freeze_run_plan(policy_dir, run_num + 1)
            all_rows.extend(rows)
            attempts_completed += 1
            last_run_num = run_num
            continue

        summary_pre = summarize(ledger)
        system_prompt = build_system_prompt(task_dir, run_dir_policy, harness_root=harness_root)
        user_msg = build_user_message_from_run(
            iteration=loop_idx,
            prev_summary_md=prev_summary_md,
            prev_variant_py=prev_variant_py,
            stuck_signal=_stuck_signal(summary_pre),
        )

        trajectory_path = run_dir_work / "planner" / "trajectory.jsonl"
        started_at = datetime.now(timezone.utc).isoformat()
        _append_trajectory(trajectory_path, {
            "turn": 0,
            "role": "user",
            "system_prompt": system_prompt,
            "user_message": user_msg,
            "timestamp_utc": started_at,
        })

        conversation: list[dict] = [{"role": "user", "content": user_msg}]
        variant_path: Path | None = None
        iter_input_tokens = 0
        iter_output_tokens = 0
        turn_count = 0
        stop_reason = "max_turns"
        model_name = llm_config.get("model", "unknown")

        for _turn in range(MAX_TURNS_PER_ITER):
            turn_count += 1
            print(f"[harness] attempt {run_num} turn {turn_count}: calling planner LLM "
                  f"(system={len(system_prompt)}B, msgs={len(conversation)}) …",
                  file=sys.stderr, flush=True)
            resp = client.messages(
                system=system_prompt,
                messages=conversation,
                tools=TOOL_SCHEMAS,
            )
            usage = (resp.raw or {}).get("usage") or {}
            turn_in = int(usage.get("input_tokens") or 0)
            turn_out = int(usage.get("output_tokens") or 0)
            iter_input_tokens += turn_in
            iter_output_tokens += turn_out
            print(f"[harness] attempt {run_num} turn {turn_count}: LLM returned "
                  f"tokens_in={turn_in} tokens_out={turn_out} "
                  f"tool_uses={len(resp.tool_uses)} stop_reason={resp.stop_reason}",
                  file=sys.stderr, flush=True)

            raw_content = resp.raw.get("content") if isinstance(resp.raw, dict) else None
            _append_trajectory(trajectory_path, {
                "turn": turn_count,
                "role": "assistant",
                "assistant_content": raw_content,
                "tool_uses": resp.tool_uses,
                "input_tokens": turn_in,
                "output_tokens": turn_out,
                "model": model_name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

            if not resp.tool_uses:
                stop_reason = "no_tool_use"
                break

            if raw_content:
                conversation.append({"role": "assistant", "content": raw_content})

            tool_results: list[dict] = []
            has_subagent = False
            for tu in resp.tool_uses:
                try:
                    if tu["name"] == "spawn_subagent":
                        has_subagent = True
                        result_text = execute_spawn_subagent(tu, client)
                        subagent_dir = run_dir_work / "planner" / "subagents"
                        subagent_dir.mkdir(parents=True, exist_ok=True)
                        kind = tu.get("input", {}).get("kind", "subagent")
                        (subagent_dir / f"{turn_count}_{kind}.md").write_text(result_text)
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

            _append_trajectory(trajectory_path, {
                "turn": turn_count,
                "role": "tool_results",
                "tool_results": tool_results,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

            if not has_subagent:
                stop_reason = "variant_written" if variant_path else "no_subagent_no_variant"
                break

        if variant_path is None:
            print(f"[orchestrator] run {run_num}: no write_variant emitted; skipping dispatch/summary/plan.",
                  file=sys.stderr)
            continue

        variant_snapshot = run_dir_work / "variant.py"
        shutil.copy2(variant_path, variant_snapshot)
        variants_dir = run_dir_policy / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)
        if not (variants_dir / variant_path.name).exists():
            shutil.copy2(variant_path, variants_dir / variant_path.name)
        _write_variant_diff(work_root, run_num, variant_snapshot)

        print(f"[harness] attempt {run_num}: variant snapshot -> {variant_snapshot}; "
              f"dispatching backend={backend} seeds={seeds_per_iter} …",
              file=sys.stderr, flush=True)

        rows = harness_run(
            task_dir,
            seeds_per_iter,
            backend,
            out_root,
            variant=variant_path,
            ledger=ledger,
            llm_config=llm_config,
            agent=agent,
            concurrent=concurrent,
            attempt=run_num,
            input_tokens=iter_input_tokens or None,
            output_tokens=iter_output_tokens or None,
            total_tokens=(iter_input_tokens + iter_output_tokens) or None,
            attempt_dir=run_dir_work,
            jobs_dir=jobs_dir,
            verifier_enabled=verifier_enabled,
            codex_bridge_url=codex_bridge_url,
            codex_bridge_key=codex_bridge_key,
            judge_model=judge_model,
            precomputed_judgements=precomputed_judgements,
            retry_count=retry_count,
        )
        finished_at = datetime.now(timezone.utc).isoformat()

        meta = {
            "attempt": run_num,
            "started_at": started_at,
            "finished_at": finished_at,
            "model": model_name,
            "proxy_base_url": llm_config.get("base_url"),
            "input_tokens": iter_input_tokens,
            "output_tokens": iter_output_tokens,
            "total_tokens": iter_input_tokens + iter_output_tokens,
            "stop_reason": stop_reason,
            "turns": turn_count,
        }
        (run_dir_work / "attempt_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n"
        )

        cumulative_rows = load_ledger(ledger)
        write_attempt_summary(run_dir_work, task_id, run_num, rows, cumulative_rows)

        summary_post = summarize(ledger)
        try:
            update_plan(policy_dir / "plan.md", summary_post)
        except (FileNotFoundError, ValueError):
            pass
        try:
            update_goal(policy_dir / "goal.md", summary_post)
        except (FileNotFoundError, ValueError):
            pass
        freeze_run_plan(policy_dir, run_num + 1)

        all_rows.extend(rows)
        attempts_completed += 1
        last_run_num = run_num
        cumulative_input_tokens += iter_input_tokens

        seed_statuses = [r.get("status") for r in rows]
        all_incomplete = bool(seed_statuses) and all(
            s == "verification_incomplete" for s in seed_statuses
        )
        if all_incomplete:
            consecutive_incomplete += 1
        else:
            consecutive_incomplete = 0

        if target_reward is not None:
            def _meets(row: dict) -> bool:
                cr = row.get("consolidated_reward")
                return cr is not None and cr >= target_reward
            hit_seed = next((r for r in rows if _meets(r)), None)
            if hit_seed is not None:
                terminate_reason = "target_reached"
                terminate_message = (
                    f"seed_{hit_seed.get('seed')} of attempt {run_num} hit "
                    f"consolidated_reward={hit_seed.get('consolidated_reward')} "
                    f">= --target-reward {target_reward}"
                )
                terminate_extras = {
                    "hit_seed": hit_seed.get("seed"),
                    "hit_variant": hit_seed.get("variant"),
                    "hit_consolidated_reward": hit_seed.get("consolidated_reward"),
                }
                break

        if (k_consecutive_incomplete is not None
                and consecutive_incomplete >= k_consecutive_incomplete):
            terminate_reason = "k_consecutive_incomplete"
            terminate_message = (
                f"{consecutive_incomplete} consecutive attempts had every seed "
                f"status=verification_incomplete (K={k_consecutive_incomplete})"
            )
            break

    if terminate_reason is None and attempts_completed >= iterations:
        pass
    elif terminate_reason is not None:
        _write_loop_terminated(
            work_root, terminate_reason, terminate_message,
            attempts_completed=attempts_completed,
            last_run_num=last_run_num,
            extras=terminate_extras or None,
        )

    return all_rows
