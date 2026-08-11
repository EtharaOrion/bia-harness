from __future__ import annotations

from pathlib import Path

from harness import _slugify, resolve_ledger_path, resolve_policy_root


HARNESS_ROOT = Path(__file__).resolve().parent.parent


def test_slugify_normalizes_slash():
    assert _slugify("nanogpt/track3-speedrun") == "nanogpt-track3-speedrun"
    assert _slugify("bia-track3-optimizer-novelty") == "bia-track3-optimizer-novelty"
    assert _slugify("a/b/c") == "a-b-c"


def test_resolve_ledger_path_shape():
    p = resolve_ledger_path("bia-track3-optimizer-novelty", harness_root=Path("/root"))
    assert p == Path("/root/runs/bia-track3-optimizer-novelty/runs.jsonl")


def test_resolve_ledger_path_slugifies_task_id():
    p = resolve_ledger_path("nanogpt/track3-speedrun", harness_root=Path("/root"))
    assert p == Path("/root/runs/nanogpt-track3-speedrun/runs.jsonl")


def test_resolve_policy_root_from_bracketed_name(tmp_path):
    task_dir = tmp_path / "tasks" / "sometask"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[task]\nname = "foo/bar"\n')
    p = resolve_policy_root(task_dir, harness_root=tmp_path)
    assert p == tmp_path / "policy" / "foo-bar"


def test_resolve_policy_root_from_top_level_name_bia_shape(tmp_path):
    task_dir = tmp_path / "tasks" / "b92e1502"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('name = "bia-track3-optimizer-novelty"\n')
    p = resolve_policy_root(task_dir, harness_root=tmp_path)
    assert p == tmp_path / "policy" / "bia-track3-optimizer-novelty"


def test_real_bia_symlink_resolves_to_shared_policy_dir():
    bia_task = HARNESS_ROOT / "tasks" / "b92e1502-7efe-5004-af4a-a7715da77b41"
    if not bia_task.exists():
        return
    p = resolve_policy_root(bia_task, harness_root=HARNESS_ROOT)
    assert p == HARNESS_ROOT / "policy" / "bia-track3-optimizer-novelty"
