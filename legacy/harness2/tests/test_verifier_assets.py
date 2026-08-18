from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from bia_verifier.pipeline import DatasetSpec
from bia_verifier.registry import Registry
from bia_verifier.rubric import Rubric
from bia_verifier.schemas import Owner

HARNESS_ROOT = Path(__file__).resolve().parents[3]
TASKS_ROOT = HARNESS_ROOT / "tasks"
# `python -m bia_verifier.cli` resolves from here, not HARNESS_ROOT, since the move.
BIA_VERIFIER_PARENT = HARNESS_ROOT / "legacy" / "harness2"


def _verifier_dirs() -> list[Path]:
    return sorted(p for p in TASKS_ROOT.glob("*/verifier") if p.is_dir())


def _load_predicates(verifier_dir: Path, suffix: str) -> dict:
    predicates_path = verifier_dir / "predicates.py"
    assert predicates_path.is_file(), f"missing predicates.py in {verifier_dir}"
    module_name = f"bv_predicates_{verifier_dir.parent.name}_{suffix}"
    spec_loader = importlib.util.spec_from_file_location(module_name, predicates_path)
    assert spec_loader is not None and spec_loader.loader is not None
    module = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(module)
    predicates = getattr(module, "PREDICATES", None)
    assert isinstance(predicates, dict), f"{predicates_path}: PREDICATES must be a dict"
    return predicates


@pytest.mark.parametrize("verifier_dir", _verifier_dirs(),
                         ids=lambda p: p.parent.name)
def test_dataset_spec_parses(verifier_dir):
    dataset = verifier_dir / "dataset.json"
    assert dataset.is_file(), f"missing dataset.json in {verifier_dir}"
    spec = DatasetSpec.load(str(dataset))
    assert spec.name, f"{verifier_dir}: dataset.name must be non-empty"
    assert spec.truth.steps, f"{verifier_dir}: truth spec has no steps"
    assert any(s.final_outcome for s in spec.truth.steps), (
        f"{verifier_dir}: no truth step is marked final_outcome")


@pytest.mark.parametrize("verifier_dir", _verifier_dirs(),
                         ids=lambda p: p.parent.name)
def test_cli_prove_reports_mece_sound(verifier_dir):
    dataset = verifier_dir / "dataset.json"
    result = subprocess.run(
        [sys.executable, "-m", "bia_verifier.cli", "prove", "--dataset", str(dataset)],
        cwd=str(BIA_VERIFIER_PARENT), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"prove failed for {verifier_dir}:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}")
    assert "MECE: sound" in result.stdout


@pytest.mark.parametrize("verifier_dir", _verifier_dirs(),
                         ids=lambda p: p.parent.name)
def test_predicates_module_covers_deterministic_ids(verifier_dir):
    spec = DatasetSpec.load(str(verifier_dir / "dataset.json"))
    predicates = _load_predicates(verifier_dir, "covers")
    det_ids = {c.id for c in spec.concerns if c.owner is Owner.DETERMINISTIC}
    assert set(predicates) == det_ids, (
        f"{verifier_dir}: PREDICATES ids != deterministic concern ids;\n"
        f"missing={sorted(det_ids - set(predicates))}\n"
        f"extra={sorted(set(predicates) - det_ids)}")
    for cid, fn in predicates.items():
        assert callable(fn), f"{verifier_dir}: PREDICATES[{cid!r}] must be callable"


@pytest.mark.parametrize("verifier_dir", _verifier_dirs(),
                         ids=lambda p: p.parent.name)
def test_rubric_ids_match_rubric_owned_concerns(verifier_dir):
    spec = DatasetSpec.load(str(verifier_dir / "dataset.json"))
    rubric_ids = sorted({c.id for c in spec.concerns if c.owner is Owner.RUBRIC})
    rubric_path = verifier_dir / "rubric.json"
    if not rubric_ids:
        assert not rubric_path.exists(), (
            f"{verifier_dir}: rubric.json exists but spec declares no rubric concerns")
        return
    assert rubric_path.is_file(), (
        f"{verifier_dir}: spec declares rubric concerns but rubric.json is missing")
    Rubric.load(str(rubric_path)).validate(rubric_ids)


@pytest.mark.parametrize("verifier_dir", _verifier_dirs(),
                         ids=lambda p: p.parent.name)
def test_registry_reports_no_violations_with_bound_predicates(verifier_dir):
    spec = DatasetSpec.load(str(verifier_dir / "dataset.json"))
    predicates = _load_predicates(verifier_dir, "reg")
    reg = Registry(spec.concerns, spec.truth, set(predicates))
    v = reg.violations()
    assert v == [], (
        f"{verifier_dir}: registry violations:\n  - " + "\n  - ".join(v))
