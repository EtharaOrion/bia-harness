from __future__ import annotations

import json
from pathlib import Path

from summarize import (
    best_per_variant,
    family_of,
    frontier,
    load_ledger,
    ruled_down_suggestions,
    stuck_count_by_family,
    summarize,
)


FIXTURE = Path(__file__).parent / "fixtures" / "ledger_frontier.jsonl"


def _row(**overrides):
    base = {
        "variant": "v",
        "seed": 0,
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "hit_target": True,
        "step_to_3_28": 3000,
        "reward": 0.5,
        "final_val_loss": 3.27,
    }
    base.update(overrides)
    return base


def test_load_ledger_missing_file(tmp_path):
    assert load_ledger(tmp_path / "nope.jsonl") == []


def test_load_ledger_empty(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("")
    assert load_ledger(p) == []


def test_load_ledger_skips_malformed(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"variant":"a"}\nnot-json\n{"variant":"b"}\n\n')
    rows = load_ledger(p)
    assert [r["variant"] for r in rows] == ["a", "b"]


def test_family_of():
    assert family_of("baseline_muon") == "baseline_muon"
    assert family_of("muon_lr025_wd05") == "muon_lr025_wd05"
    assert family_of("muon_lr025_wd05_polar") == "muon_lr025_wd05"
    assert family_of("muon_lr025_wd05_polar_extra_soap") == "muon_lr025_wd05"
    assert family_of("") == ""


def test_best_per_variant_picks_min_step():
    rows = [
        _row(variant="a", seed=0, step_to_3_28=3000),
        _row(variant="a", seed=1, step_to_3_28=2800),
        _row(variant="a", seed=2, step_to_3_28=2900),
    ]
    best = best_per_variant(rows)
    assert best["a"]["step_to_3_28"] == 2800


def test_best_per_variant_ignores_non_hits():
    rows = [
        _row(variant="a", hit_target=False, step_to_3_28=None, reward=0.0),
        _row(variant="a", seed=1, step_to_3_28=2900),
    ]
    best = best_per_variant(rows)
    assert list(best.keys()) == ["a"]
    assert best["a"]["step_to_3_28"] == 2900


def test_frontier_dedupes_by_variant_and_sorts():
    rows = [
        _row(variant="a", seed=0, step_to_3_28=3000),
        _row(variant="a", seed=1, step_to_3_28=2800),
        _row(variant="b", seed=0, step_to_3_28=2900),
    ]
    f = frontier(rows)
    assert [r["variant"] for r in f] == ["a", "b"]
    assert [r["step_to_3_28"] for r in f] == [2800, 2900]


def test_stuck_by_family_zero_when_improving():
    rows = [
        _row(variant="mu_lr_wd", seed=0, timestamp_utc="t1", step_to_3_28=3000),
        _row(variant="mu_lr_wd", seed=1, timestamp_utc="t2", step_to_3_28=3000),
        _row(variant="mu_lr_wd_polar", seed=0, timestamp_utc="t3", step_to_3_28=2800),
        _row(variant="mu_lr_wd_polar", seed=1, timestamp_utc="t4", step_to_3_28=2800),
    ]
    stuck = stuck_count_by_family(rows)
    assert stuck["mu_lr_wd"] == 0


def test_stuck_by_family_counts_regressions_and_no_2seed_hits():
    rows = [
        _row(variant="a_b_c", seed=0, timestamp_utc="t1", step_to_3_28=2800),
        _row(variant="a_b_c", seed=1, timestamp_utc="t2", step_to_3_28=2800),
        _row(variant="a_b_c_regress", seed=0, timestamp_utc="t3", step_to_3_28=3100),
        _row(variant="a_b_c_regress", seed=1, timestamp_utc="t4", step_to_3_28=3100),
        _row(variant="a_b_c_only_one_seed", seed=0, timestamp_utc="t5", step_to_3_28=2700),
    ]
    stuck = stuck_count_by_family(rows)
    assert stuck["a_b_c"] == 2


def test_ruled_down_needs_two_different_seeds():
    rows = [
        _row(variant="dead", seed=0, hit_target=False, reward=0.0, step_to_3_28=None),
        _row(variant="dead", seed=1, hit_target=False, reward=0.0, step_to_3_28=None),
        _row(variant="alive", seed=0, hit_target=True, reward=0.5, step_to_3_28=2900),
        _row(variant="one_seed_zero", seed=0, hit_target=False, reward=0.0, step_to_3_28=None),
    ]
    ruled = ruled_down_suggestions(rows)
    assert ruled == ["dead"]


def test_summarize_composes_all_keys(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps(_row()) + "\n")
    s = summarize(p)
    assert set(s.keys()) == {
        "n_rows",
        "best_per_variant",
        "frontier",
        "stuck_by_family",
        "ruled_down_suggestions",
        "noise_floor",
    }
    assert s["n_rows"] == 1


def test_summarize_on_real_fixture():
    s = summarize(FIXTURE)
    assert s["n_rows"] == 10
    best = s["best_per_variant"]
    assert "muon_lr025_wd05_polar" in best
    assert best["muon_lr025_wd05_polar"]["step_to_3_28"] == 2820
    assert best["baseline_muon"]["step_to_3_28"] == 3500
    assert s["frontier"][0]["variant"] == "muon_lr025_wd05_polar"
    assert s["ruled_down_suggestions"] == ["muon_lr025_broken"]
    assert s["stuck_by_family"]["muon_lr025_wd05"] == 1


def test_noise_floor_stats_none_with_fewer_than_2_baseline_rows():
    from summarize import noise_floor_stats
    rows = [_row(variant="(none)", seed=0, step_to_3_28=3200)]
    assert noise_floor_stats(rows) is None


def test_noise_floor_stats_computes_mean_and_std():
    from summarize import noise_floor_stats
    rows = [
        _row(variant="(none)", seed=0, step_to_3_28=3200, final_val_loss=3.278),
        _row(variant="(none)", seed=1, step_to_3_28=3220, final_val_loss=3.279),
        _row(variant="(none)", seed=2, step_to_3_28=3180, final_val_loss=3.277),
    ]
    stats = noise_floor_stats(rows)
    assert stats["n"] == 3
    assert stats["mean_step_to_3_28"] == 3200.0
    assert stats["std_step_to_3_28"] > 0


def test_noise_floor_stats_ignores_non_baseline():
    from summarize import noise_floor_stats
    rows = [
        _row(variant="my_variant", seed=0, step_to_3_28=3200),
        _row(variant="my_variant", seed=1, step_to_3_28=3220),
    ]
    assert noise_floor_stats(rows) is None


def test_noise_floor_stats_accepts_baseline_alias():
    from summarize import noise_floor_stats
    rows = [
        _row(variant="baseline", seed=0, step_to_3_28=3500),
        _row(variant="baseline", seed=1, step_to_3_28=3480),
    ]
    stats = noise_floor_stats(rows)
    assert stats is not None
    assert stats["n"] == 2
