from __future__ import annotations

from pathlib import Path

import pytest

from plan_writer import (
    CURRENT_STATE,
    FRONTIER,
    render_current_state,
    render_frontier_table,
    update_goal,
    update_plan,
    write_between_markers,
)


PLAN_TEMPLATE = """\
# Plan

## 1. Current state

<!-- AUTO-GENERATED:CURRENT_STATE:START -->
placeholder
<!-- AUTO-GENERATED:CURRENT_STATE:END -->

## 2. Active picklist (LLM-owned)

- Try muon_lr025_wd05

## 3. Lessons (LLM-owned, append-only)
"""

GOAL_TEMPLATE = """\
# Goal

## Frontier

<!-- AUTO-GENERATED:FRONTIER:START -->
placeholder
<!-- AUTO-GENERATED:FRONTIER:END -->

## Combos worth trying (LLM-owned)
"""


def _summary(**overrides):
    base = {
        "n_rows": 3,
        "best_per_variant": {
            "a": {"variant": "a", "step_to_3_28": 2800, "final_val_loss": 3.27, "seed": 1},
        },
        "frontier": [
            {"variant": "a", "step_to_3_28": 2800, "final_val_loss": 3.27, "seed": 1},
        ],
        "stuck_by_family": {"a": 0},
        "ruled_down_suggestions": [],
    }
    base.update(overrides)
    return base


def test_write_between_markers_replaces_body(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text(PLAN_TEMPLATE)
    write_between_markers(p, CURRENT_STATE, "new body content")
    text = p.read_text()
    assert "new body content" in text
    assert "placeholder" not in text
    assert "## 2. Active picklist (LLM-owned)" in text
    assert "- Try muon_lr025_wd05" in text


def test_write_between_markers_idempotent(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text(PLAN_TEMPLATE)
    write_between_markers(p, CURRENT_STATE, "same body")
    once = p.read_text()
    write_between_markers(p, CURRENT_STATE, "same body")
    twice = p.read_text()
    assert once == twice


def test_write_between_markers_missing_markers_raises(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text("# Plan\n\nno markers here\n")
    with pytest.raises(ValueError, match="markers"):
        write_between_markers(p, CURRENT_STATE, "body")


def test_write_between_markers_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_between_markers(tmp_path / "nope.md", CURRENT_STATE, "body")


def test_prose_outside_markers_untouched(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text(PLAN_TEMPLATE)
    write_between_markers(p, CURRENT_STATE, "rendered stats")
    text = p.read_text()
    assert "## 2. Active picklist (LLM-owned)" in text
    assert "## 3. Lessons (LLM-owned, append-only)" in text
    assert "- Try muon_lr025_wd05" in text


def test_render_current_state_has_expected_fields():
    body = render_current_state(_summary())
    assert "Total attempts recorded**: 3" in body
    assert "Best step_to_3_28**: 2800" in body
    assert "Distinct variants that hit target**: 1" in body


def test_render_current_state_lists_ruled_down():
    body = render_current_state(_summary(ruled_down_suggestions=["dead_a", "dead_b"]))
    assert "Suggested ruled-down** (2)" in body
    assert "`dead_a`" in body
    assert "`dead_b`" in body


def test_render_frontier_table_empty():
    body = render_frontier_table(_summary(frontier=[]))
    assert "no target hits" in body


def test_render_frontier_table_has_rows():
    body = render_frontier_table(_summary())
    assert "| Rank |" in body
    assert "| 1 | `a` | 2800 | 3.27 | 1 |" in body


def test_update_plan_and_goal_end_to_end(tmp_path):
    plan = tmp_path / "plan.md"
    goal = tmp_path / "goal.md"
    plan.write_text(PLAN_TEMPLATE)
    goal.write_text(GOAL_TEMPLATE)
    s = _summary()
    update_plan(plan, s)
    update_goal(goal, s)
    plan_text = plan.read_text()
    goal_text = goal.read_text()
    assert "placeholder" not in plan_text
    assert "placeholder" not in goal_text
    assert "Best step_to_3_28**: 2800" in plan_text
    assert "| 1 | `a` | 2800" in goal_text


def test_render_current_state_shows_noise_floor():
    body = render_current_state(_summary(noise_floor={"n": 3, "mean_step_to_3_28": 3200.0, "std_step_to_3_28": 20.0, "mean_final_val_loss": 3.279, "std_final_val_loss": 0.001}))
    assert "Noise floor" in body
    assert "3200.0" in body
    assert "n=3" in body
