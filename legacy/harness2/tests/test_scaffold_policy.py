from __future__ import annotations

from pathlib import Path

from scaffold_policy import ensure_policy_scaffolded


def _make_task(tmp_path: Path, *, name: str, description: str | None = None, dialect: str = "bracketed") -> Path:
    task_dir = tmp_path / "tasks" / name
    task_dir.mkdir(parents=True)
    if dialect == "bracketed":
        lines = ['[task]', f'name = "{name}"']
        if description:
            lines.append(f'description = "{description}"')
    else:
        lines = [f'name = "{name}"']
        if description:
            lines.append(f'description = "{description}"')
    (task_dir / "task.toml").write_text("\n".join(lines) + "\n")
    return task_dir


def test_scaffold_creates_all_expected_files(tmp_path):
    task = _make_task(tmp_path, name="my-task", description="Do the thing.")
    policy_dir = ensure_policy_scaffolded(task, harness_root=tmp_path)
    assert (policy_dir / "plan.md").is_file()
    assert (policy_dir / "goal.md").is_file()
    assert (policy_dir / "scratchpad" / "THREAD.md").is_file()
    assert (policy_dir / "scratchpad" / "picklist.md").is_file()
    assert (policy_dir / "scratchpad" / "audits.md").is_file()
    assert (policy_dir / "scratchpad" / "variants" / ".gitkeep").is_file()


def test_scaffold_embeds_task_description_in_goal(tmp_path):
    task = _make_task(tmp_path, name="my-task", description="Optimize the widget.")
    policy_dir = ensure_policy_scaffolded(task, harness_root=tmp_path)
    assert "Optimize the widget." in (policy_dir / "goal.md").read_text()


def test_scaffold_uses_bia_dialect_top_level_name(tmp_path):
    task = _make_task(tmp_path, name="bia-shape", description="bia mission", dialect="top-level")
    policy_dir = ensure_policy_scaffolded(task, harness_root=tmp_path)
    assert policy_dir.name == "bia-shape"
    assert "bia mission" in (policy_dir / "goal.md").read_text()


def test_scaffold_plan_has_auto_generated_markers(tmp_path):
    task = _make_task(tmp_path, name="my-task")
    policy_dir = ensure_policy_scaffolded(task, harness_root=tmp_path)
    plan = (policy_dir / "plan.md").read_text()
    assert "<!-- AUTO-GENERATED:CURRENT_STATE:START -->" in plan
    assert "<!-- AUTO-GENERATED:CURRENT_STATE:END -->" in plan


def test_scaffold_goal_has_auto_generated_markers(tmp_path):
    task = _make_task(tmp_path, name="my-task")
    policy_dir = ensure_policy_scaffolded(task, harness_root=tmp_path)
    goal = (policy_dir / "goal.md").read_text()
    assert "<!-- AUTO-GENERATED:FRONTIER:START -->" in goal
    assert "<!-- AUTO-GENERATED:FRONTIER:END -->" in goal


def test_scaffold_is_idempotent_and_preserves_llm_edits(tmp_path):
    task = _make_task(tmp_path, name="my-task")
    policy_dir = ensure_policy_scaffolded(task, harness_root=tmp_path)
    llm_edit = "\n## 3. Lessons\n\n- 2026-08-12 my_iter — did something\n"
    (policy_dir / "plan.md").write_text((policy_dir / "plan.md").read_text() + llm_edit)
    after_edit = (policy_dir / "plan.md").read_text()
    ensure_policy_scaffolded(task, harness_root=tmp_path)
    assert (policy_dir / "plan.md").read_text() == after_edit


def test_scaffold_synthesizes_description_when_missing(tmp_path):
    task = _make_task(tmp_path, name="my-task")
    policy_dir = ensure_policy_scaffolded(task, harness_root=tmp_path)
    goal = (policy_dir / "goal.md").read_text()
    assert "my-task" in goal
