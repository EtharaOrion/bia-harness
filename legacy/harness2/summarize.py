from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_ledger(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def family_of(name: str) -> str:
    if not name:
        return ""
    parts = name.split("_")
    if len(parts) < 3:
        return name
    return "_".join(parts[:3])


def best_per_variant(rows: list[dict]) -> dict[str, dict]:
    hits: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        v = row.get("variant")
        if v and row.get("hit_target") and row.get("step_to_3_28") is not None:
            hits[v].append(row)
    return {v: min(rs, key=lambda r: r["step_to_3_28"]) for v, rs in hits.items()}


def frontier(rows: list[dict], top_n: int = 5) -> list[dict]:
    best = best_per_variant(rows)
    return sorted(best.values(), key=lambda r: r["step_to_3_28"])[:top_n]


def _qualifying_step(variant_rows: list[dict]) -> int | None:
    hits = [r for r in variant_rows if r.get("hit_target") and r.get("step_to_3_28") is not None]
    seeds = {r.get("seed") for r in hits if r.get("seed") is not None}
    if len(seeds) < 2:
        return None
    return min(r["step_to_3_28"] for r in hits)


def stuck_count_by_family(rows: list[dict]) -> dict[str, int]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        v = row.get("variant")
        if v:
            by_variant[v].append(row)

    variant_meta: list[tuple[str, str, int | None]] = []
    for variant, vrows in by_variant.items():
        latest_ts = max((r.get("timestamp_utc", "") for r in vrows), default="")
        qstep = _qualifying_step(vrows)
        variant_meta.append((latest_ts, variant, qstep))
    variant_meta.sort(key=lambda t: t[0])

    family_state: dict[str, dict] = {}
    for _ts, variant, qstep in variant_meta:
        fam = family_of(variant)
        state = family_state.setdefault(fam, {"best": None, "since": 0})
        if qstep is not None and (state["best"] is None or qstep < state["best"]):
            state["best"] = qstep
            state["since"] = 0
        else:
            state["since"] += 1
    return {f: s["since"] for f, s in family_state.items()}


def noise_floor_stats(rows: list[dict], baseline_variants=("(none)", "baseline")) -> dict | None:
    steps: list[float] = []
    losses: list[float] = []
    for row in rows:
        v = row.get("variant")
        if v in baseline_variants and row.get("hit_target") and row.get("step_to_3_28") is not None:
            steps.append(float(row["step_to_3_28"]))
            if row.get("final_val_loss") is not None:
                losses.append(float(row["final_val_loss"]))
    if len(steps) < 2:
        return None
    import statistics
    return {
        "n": len(steps),
        "mean_step_to_3_28": round(statistics.mean(steps), 2),
        "std_step_to_3_28": round(statistics.pstdev(steps), 2),
        "mean_final_val_loss": round(statistics.mean(losses), 5) if losses else None,
        "std_final_val_loss": round(statistics.pstdev(losses), 5) if len(losses) >= 2 else None,
    }


def ruled_down_suggestions(rows: list[dict]) -> list[str]:
    fail_seeds: dict[str, set] = defaultdict(set)
    for row in rows:
        v = row.get("variant")
        if v and not row.get("hit_target") and row.get("reward") == 0.0:
            seed = row.get("seed")
            if seed is not None:
                fail_seeds[v].add(seed)
    return sorted(v for v, seeds in fail_seeds.items() if len(seeds) >= 2)


def summarize(path: Path | str) -> dict[str, Any]:
    rows = load_ledger(path)
    return {
        "n_rows": len(rows),
        "best_per_variant": best_per_variant(rows),
        "frontier": frontier(rows),
        "stuck_by_family": stuck_count_by_family(rows),
        "ruled_down_suggestions": ruled_down_suggestions(rows),
        "noise_floor": noise_floor_stats(rows),
    }


def _exclude_incomplete(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") != "verification_incomplete"]


def _attempt_aggregates(current_rows: list[dict]) -> dict:
    qualifying = _exclude_incomplete(current_rows)
    n_total = len(current_rows)
    n_qual = len(qualifying)
    n_incomplete = n_total - n_qual
    statuses: dict[str, int] = defaultdict(int)
    for r in current_rows:
        statuses[str(r.get("status") or "unknown")] += 1
    rewards = [float(r["reward"]) for r in qualifying if r.get("reward") is not None]
    hits = [1 for r in qualifying if r.get("hit_target")]
    steps = [float(r["step_to_3_28"]) for r in qualifying
             if r.get("hit_target") and r.get("step_to_3_28") is not None]
    import statistics
    agg: dict[str, Any] = {
        "n_seeds": n_total,
        "n_qualifying": n_qual,
        "n_verification_incomplete": n_incomplete,
        "statuses": dict(statuses),
        "hit_rate": (sum(hits) / n_qual) if n_qual else None,
    }
    if len(rewards) >= 1:
        agg["mean_reward"] = round(statistics.mean(rewards), 6)
        agg["std_reward"] = round(statistics.pstdev(rewards), 6) if len(rewards) >= 2 else 0.0
    else:
        agg["mean_reward"] = None
        agg["std_reward"] = None
    if steps:
        agg["mean_step_to_3_28"] = round(statistics.mean(steps), 2)
        agg["std_step_to_3_28"] = round(statistics.pstdev(steps), 2) if len(steps) >= 2 else 0.0
        agg["min_step_to_3_28"] = min(steps)
    else:
        agg["mean_step_to_3_28"] = None
        agg["std_step_to_3_28"] = None
        agg["min_step_to_3_28"] = None
    return agg


def render_attempt_summary_md(current_rows: list[dict], cumulative_rows: list[dict],
                              task_id: str, attempt_num: int) -> str:
    ts = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    agg = _attempt_aggregates(current_rows)
    cum_qual = _exclude_incomplete(cumulative_rows)
    frontier_rows = frontier(cum_qual)
    ruled = ruled_down_suggestions(cum_qual)
    stuck = stuck_count_by_family(cum_qual)
    nf = noise_floor_stats(cum_qual)

    lines: list[str] = []
    lines.append(f"# Attempt {attempt_num} summary — `{task_id}`")
    lines.append("")
    lines.append(f"_timestamp_utc: {ts}_")
    lines.append("")

    lines.append("## 1. Header")
    lines.append(f"- task_id: `{task_id}`")
    lines.append(f"- attempt: {attempt_num}")
    lines.append(f"- seeds dispatched this attempt: {agg['n_seeds']}")
    lines.append("")

    lines.append("## 2. Per-attempt aggregates")
    lines.append(f"- statuses: {agg['statuses']}")
    lines.append(f"- verification_incomplete (excluded from aggregates): {agg['n_verification_incomplete']}")
    lines.append(f"- qualifying seeds: {agg['n_qualifying']}")
    lines.append(f"- hit_rate: {agg['hit_rate']}")
    lines.append(f"- mean reward: {agg['mean_reward']} (std {agg['std_reward']})")
    lines.append(
        f"- step_to_3_28: mean {agg['mean_step_to_3_28']} std {agg['std_step_to_3_28']} "
        f"min {agg['min_step_to_3_28']}"
    )
    lines.append("")

    lines.append("## 3. Frontier (across all attempts, qualifying only)")
    if frontier_rows:
        lines.append("| Rank | Variant | attempt | step_to_3_28 | final_val_loss | seed |")
        lines.append("|------|---------|---------|--------------|----------------|------|")
        for i, r in enumerate(frontier_rows, 1):
            lines.append(
                f"| {i} | `{r.get('variant','?')}` | {r.get('attempt','—')} | "
                f"{r.get('step_to_3_28','—')} | {r.get('final_val_loss','—')} | {r.get('seed','—')} |"
            )
    else:
        lines.append("_(no qualifying result recorded yet)_")
    lines.append("")

    lines.append("## 4. Ruled-down suggestions")
    if ruled:
        lines.append("- " + ", ".join(f"`{v}`" for v in ruled))
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append("## 5. Stuck-by-family signals")
    if stuck:
        for fam, n in sorted(stuck.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{fam}`: {n} attempts since last improvement")
    else:
        lines.append("_(no family tracking yet)_")
    lines.append("")

    lines.append("## 6. Lessons")
    lines.append("_(planner authors under plan.md §3 next attempt)_")
    lines.append("")

    lines.append("## 7. Next-attempt hint")
    if agg["n_qualifying"] < 2:
        lines.append("- Rule 5 not satisfied this attempt (need >=2 verified seeds). Repeat or diversify seeds.")
    if stuck:
        hot = [f for f, n in stuck.items() if n >= 15]
        if hot:
            lines.append(f"- Stuck families {hot} — pivot to a new slug-family.")
    if not lines[-1].startswith("- "):
        lines.append("- Continue current family; no stuck signal.")
    lines.append("")
    return "\n".join(lines)


def write_attempt_summary(attempt_dir: Path, task_id: str, attempt_num: int,
                          current_rows: list[dict],
                          cumulative_rows: list[dict]) -> tuple[Path, Path]:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    md = render_attempt_summary_md(current_rows, cumulative_rows, task_id, attempt_num)
    md_path = attempt_dir / "summary.md"
    md_path.write_text(md)
    agg = _attempt_aggregates(current_rows)
    js = {
        "task_id": task_id,
        "attempt": attempt_num,
        "aggregates": agg,
        "rows": current_rows,
    }
    json_path = attempt_dir / "summary.json"
    json_path.write_text(json.dumps(js, indent=2, sort_keys=True, default=str) + "\n")
    return md_path, json_path
