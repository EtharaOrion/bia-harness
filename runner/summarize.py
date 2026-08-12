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
