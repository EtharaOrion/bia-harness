from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_client import LLMResponse
from orchestrator import (
    build_system_prompt,
    build_user_message,
    execute_tool_use,
    next_iteration_number,
    run_loop,
)


# legacy/harness2/tests/<file> -> tests -> harness2 -> legacy -> harness root
HARNESS_ROOT = Path(__file__).resolve().parents[3]


def _make_task(tmp_path, *, name="my-task", description="Do the thing.", instruction="Follow the plan."):
    task_dir = tmp_path / "tasks" / name
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(f'[task]\nname = "{name}"\ndescription = "{description}"\n')
    (task_dir / "instruction.md").write_text(instruction)
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "grader.py").write_text("import json, sys; print(json.dumps({'reward':0.5,'hit_target':True,'step_to_3_28':3000}))")
    (tmp_path / "policy").mkdir()
    (tmp_path / "policy" / "AGENTS.md").write_text("# AGENTS constitution\n")
    return task_dir


def test_next_iteration_number_empty(tmp_path):
    (tmp_path / "scratchpad" / "variants").mkdir(parents=True)
    assert next_iteration_number(tmp_path) == 0


def test_next_iteration_number_counts_iter_files(tmp_path):
    d = tmp_path / "scratchpad" / "variants"
    d.mkdir(parents=True)
    (d / "iter0.py").write_text("")
    (d / "iter1.py").write_text("")
    (d / "other.py").write_text("")
    assert next_iteration_number(tmp_path) == 2


def test_execute_write_variant_creates_file(tmp_path):
    tu = {"name": "write_variant", "input": {"filename": "iter7.py", "contents": "print('hi')\n"}}
    path = execute_tool_use(tu, tmp_path)
    assert path == tmp_path / "scratchpad" / "variants" / "iter7.py"
    assert path.read_text() == "print('hi')\n"


def test_execute_write_variant_rejects_path_traversal(tmp_path):
    for bad in ["../evil.py", "sub/x.py", ".hidden.py", "no-extension"]:
        tu = {"name": "write_variant", "input": {"filename": bad, "contents": ""}}
        with pytest.raises(ValueError):
            execute_tool_use(tu, tmp_path)


def test_execute_append_thread_appends(tmp_path):
    tu = {"name": "append_thread", "input": {"entry": "## line"}}
    execute_tool_use(tu, tmp_path)
    execute_tool_use(tu, tmp_path)
    text = (tmp_path / "scratchpad" / "THREAD.md").read_text()
    assert text.count("## line") == 2


def test_execute_update_plan_section_labels(tmp_path):
    (tmp_path / "plan.md").write_text("# original\n")
    tu = {"name": "update_plan_section", "input": {"section": 3, "contents": "- did a thing"}}
    execute_tool_use(tu, tmp_path)
    text = (tmp_path / "plan.md").read_text()
    assert "# original" in text
    assert "<!-- LLM §3 @" in text
    assert "- did a thing" in text


def test_execute_update_plan_section_rejects_bad_section(tmp_path):
    (tmp_path / "plan.md").write_text("")
    tu = {"name": "update_plan_section", "input": {"section": 5, "contents": "x"}}
    with pytest.raises(ValueError):
        execute_tool_use(tu, tmp_path)


def test_execute_add_ruled_down_appends(tmp_path):
    (tmp_path / "goal.md").write_text("# goal\n")
    tu = {"name": "add_ruled_down", "input": {"entry": "- v42 — N=4 no improve"}}
    execute_tool_use(tu, tmp_path)
    assert "- v42 — N=4 no improve" in (tmp_path / "goal.md").read_text()


def test_execute_unknown_tool_raises(tmp_path):
    with pytest.raises(ValueError):
        execute_tool_use({"name": "delete_the_universe", "input": {}}, tmp_path)


def test_build_system_prompt_includes_all_sources(tmp_path):
    task_dir = _make_task(tmp_path)
    policy_dir = tmp_path / "policy" / "my-task"
    policy_dir.mkdir(parents=True)
    (policy_dir / "plan.md").write_text("# plan body\n")
    (policy_dir / "goal.md").write_text("# goal body\n")
    prompt = build_system_prompt(task_dir, policy_dir, harness_root=tmp_path)
    assert "# AGENTS constitution" in prompt
    assert "Follow the plan." in prompt
    assert "# plan body" in prompt
    assert "# goal body" in prompt


def test_build_user_message_first_iteration_no_prev_row():
    msg = build_user_message(0, prev_row=None)
    assert "Iteration 0" in msg
    assert "first attempt" in msg
    assert "iter0.py" in msg


def test_build_user_message_with_prev_row():
    prev = {"variant": "iter0.py", "reward": 0.5, "hit_target": True}
    msg = build_user_message(1, prev_row=prev)
    assert "Iteration 1" in msg
    assert "iter1.py" in msg
    assert '"reward": 0.5' in msg


def test_run_loop_calls_llm_and_writes_variant_and_calls_harness(tmp_path):
    task_dir = _make_task(tmp_path)
    ledger = tmp_path / "runs" / "my-task" / "runs.jsonl"

    fake_llm = MagicMock()
    fake_llm.messages.return_value = LLMResponse(
        text_blocks=[],
        tool_uses=[
            {"id": "t1", "name": "write_variant", "input": {"filename": "iter0.py", "contents": "# variant\n"}},
        ],
        stop_reason="tool_use",
        raw={},
    )

    with patch("orchestrator.harness_run") as mock_hrun:
        mock_hrun.return_value = [{"variant": "iter0.py", "reward": 0.5, "hit_target": True}]
        rows = run_loop(
            task_dir,
            iterations=1,
            backend="dry",
            llm_config={"base_url": "u", "api_key": "k", "model": "m"},
            out_root=tmp_path / "runs",
            ledger=ledger,
            harness_root=tmp_path,
            llm_client=fake_llm,
            install_sigint=False,
        )

    assert fake_llm.messages.call_count == 1
    variant_file = tmp_path / "policy" / "my-task" / "scratchpad" / "variants" / "iter0.py"
    assert variant_file.is_file()
    assert variant_file.read_text() == "# variant\n"
    mock_hrun.assert_called_once()
    _args, kwargs = mock_hrun.call_args
    assert kwargs["variant"] == variant_file
    assert kwargs["ledger"] == ledger
    assert rows and rows[0]["reward"] == 0.5


def test_run_loop_skips_harness_when_no_variant_emitted(tmp_path):
    task_dir = _make_task(tmp_path)
    ledger = tmp_path / "runs" / "my-task" / "runs.jsonl"

    fake_llm = MagicMock()
    fake_llm.messages.return_value = LLMResponse(
        text_blocks=["I have no ideas"], tool_uses=[], stop_reason="end_turn", raw={}
    )

    with patch("orchestrator.harness_run") as mock_hrun:
        rows = run_loop(
            task_dir,
            iterations=1,
            backend="dry",
            llm_config={"base_url": "u", "api_key": "k", "model": "m"},
            out_root=tmp_path / "runs",
            ledger=ledger,
            harness_root=tmp_path,
            llm_client=fake_llm,
            install_sigint=False,
        )

    mock_hrun.assert_not_called()
    assert rows == []


def test_run_loop_iterations_argument_validated(tmp_path):
    task_dir = _make_task(tmp_path)
    with pytest.raises(ValueError):
        run_loop(
            task_dir, iterations=0, backend="dry",
            llm_config={"base_url": "u", "api_key": "k", "model": "m"},
            harness_root=tmp_path,
            llm_client=MagicMock(),
            install_sigint=False,
        )


def test_parse_ruled_down_finds_backticked_slugs(tmp_path):
    from orchestrator import _parse_ruled_down
    (tmp_path / "goal.md").write_text(
        "# Goal\n\n## Ruled down\n\n- `muon_lr025_broken` — N=2 no improve\n- also `foo_bar_baz`\n"
    )
    slugs = _parse_ruled_down(tmp_path / "goal.md")
    assert "muon_lr025_broken" in slugs
    assert "foo_bar_baz" in slugs


def test_parse_ruled_down_finds_bulleted_slugs(tmp_path):
    from orchestrator import _parse_ruled_down
    (tmp_path / "goal.md").write_text("## Ruled down\n\n- muon_lr025 — N=2 no improve\n")
    slugs = _parse_ruled_down(tmp_path / "goal.md")
    assert "muon_lr025" in slugs


def test_parse_ruled_down_stops_at_next_section(tmp_path):
    from orchestrator import _parse_ruled_down
    (tmp_path / "goal.md").write_text(
        "## Ruled down\n- `bad_slug`\n\n## Combos\n- `good_slug`\n"
    )
    slugs = _parse_ruled_down(tmp_path / "goal.md")
    assert "bad_slug" in slugs
    assert "good_slug" not in slugs


def test_parse_ruled_down_returns_empty_when_no_section(tmp_path):
    from orchestrator import _parse_ruled_down
    (tmp_path / "goal.md").write_text("# Goal\n\n## Frontier\n")
    assert _parse_ruled_down(tmp_path / "goal.md") == set()


def test_write_variant_rejects_ruled_down_slug(tmp_path):
    from orchestrator import execute_tool_use
    (tmp_path / "goal.md").write_text("## Ruled down\n\n- `bad_variant`\n")
    tu = {"name": "write_variant", "input": {"filename": "bad_variant.py", "contents": ""}}
    with pytest.raises(ValueError, match="ruled down"):
        execute_tool_use(tu, tmp_path)


def test_write_variant_allows_non_ruled_down_slug(tmp_path):
    from orchestrator import execute_tool_use
    (tmp_path / "goal.md").write_text("## Ruled down\n\n- `bad_variant`\n")
    tu = {"name": "write_variant", "input": {"filename": "good_variant.py", "contents": "x"}}
    path = execute_tool_use(tu, tmp_path)
    assert path.read_text() == "x"


def test_slug_stack_rejects_over_4_segments():
    from orchestrator import _validate_variant_filename
    with pytest.raises(ValueError, match="slug-stack"):
        _validate_variant_filename("muon_lr025_wd05_polar_extra_soap.py")


def test_slug_stack_allows_iter_pattern():
    from orchestrator import _validate_variant_filename
    _validate_variant_filename("iter0.py")
    _validate_variant_filename("iter123.py")


def test_slug_stack_allows_4_segments():
    from orchestrator import _validate_variant_filename
    _validate_variant_filename("muon_lr025_wd05_polar.py")


def test_stuck_signal_none_below_threshold():
    from orchestrator import _stuck_signal
    assert _stuck_signal({"stuck_by_family": {"f": 14}}) is None
    assert _stuck_signal({"stuck_by_family": {}}) is None
    assert _stuck_signal({}) is None


def test_stuck_signal_present_at_threshold():
    from orchestrator import _stuck_signal, STUCK_THRESHOLD
    sig = _stuck_signal({"stuck_by_family": {"muon_lr": STUCK_THRESHOLD}})
    assert sig is not None
    assert "STUCK-DETECTOR" in sig
    assert "muon_lr" in sig


def test_build_user_message_includes_stuck_signal():
    msg = build_user_message(3, prev_row={"reward": 0.5}, stuck_signal="STUCK-DETECTOR: pivot now")
    assert "STUCK-DETECTOR" in msg


def test_execute_spawn_subagent_calls_client_with_kind_prompt():
    from orchestrator import SUBAGENT_PROMPTS, execute_spawn_subagent
    fake = MagicMock()
    fake.messages.return_value = LLMResponse(text_blocks=["subagent output"], tool_uses=[], stop_reason="end_turn", raw={})
    tu = {"name": "spawn_subagent", "input": {"topic": "Muon paper", "kind": "paper", "notes": ""}}
    result = execute_spawn_subagent(tu, fake)
    assert result == "subagent output"
    kwargs = fake.messages.call_args.kwargs
    assert kwargs["system"] == SUBAGENT_PROMPTS["paper"]
    assert "Muon paper" in kwargs["messages"][0]["content"]


def test_spawn_subagent_rejects_unknown_kind():
    from orchestrator import execute_spawn_subagent
    fake = MagicMock()
    tu = {"name": "spawn_subagent", "input": {"topic": "x", "kind": "unknown", "notes": ""}}
    with pytest.raises(ValueError, match="unknown subagent kind"):
        execute_spawn_subagent(tu, fake)
