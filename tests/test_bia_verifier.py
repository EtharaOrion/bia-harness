from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from bia_verifier.cli import main
from bia_verifier.judge import LLMJudge, _chat_completions_url, parse_reply
from bia_verifier.pipeline import DatasetSpec, Evidence, PredicateOutcome, grade
from bia_verifier.pytest_gen import generate, run as run_generated_pytest
from bia_verifier.rubric import Rubric, RubricError
from bia_verifier.schemas import CheckStatus, Concern, Dimension, Layer, Owner
from bia_verifier.adapters import RunLayout


def _concern(concern_id: str, *, always_measurable: bool = True) -> Concern:
    return Concern(
        id=concern_id,
        title="",
        layer=Layer.STRUCTURE,
        owner=Owner.DETERMINISTIC,
        dimension=Dimension.PROCESS,
        truth_step="step",
        always_measurable=always_measurable,
    )


def test_rubric_rejects_spec_mismatch():
    rubric = Rubric.from_dict({"items": [
        {"concern_id": "rubric.a", "criterion": "Show evidence."},
    ]})
    with pytest.raises(RubricError, match="missing=.*rubric.b"):
        rubric.validate(["rubric.a", "rubric.b"])


def test_rubric_rejects_duplicate_ids():
    with pytest.raises(RubricError, match="duplicate"):
        Rubric.from_dict({"items": [
            {"concern_id": "rubric.a", "criterion": "First."},
            {"concern_id": "rubric.a", "criterion": "Second."},
        ]})


def test_codex_judge_uses_authenticated_chat_endpoint():
    judge = LLMJudge("http://127.0.0.1:8788", api_key="secret")
    response = type("Response", (), {
        "read": lambda self: json.dumps({
            "choices": [{"message": {"content": '{"verdict":"pass","score":1,"rationale":"ok"}'}}]
        }).encode(),
        "__enter__": lambda self: self,
        "__exit__": lambda self, *args: None,
    })()
    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        text = judge._post("prompt")
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:8788/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    assert text is not None
    parsed = parse_reply(text)
    assert parsed is not None
    assert parsed["verdict"] == "pass"


def test_chat_endpoint_preserves_full_path():
    assert _chat_completions_url("http://x/v1/chat/completions") == \
        "http://x/v1/chat/completions"


def test_generated_pytest_passes_and_writes_junit(tmp_path):
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(json.dumps({
        "det.ok": {"status": "pass", "summary": "", "evidence": {}},
    }))
    report = run_generated_pytest([_concern("det.ok")], str(outcomes), str(tmp_path))
    assert report.passed, report.stderr or report.stdout
    assert (tmp_path / "pytest.xml").is_file()


def test_generated_pytest_fails_missing_outcome(tmp_path):
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text("{}")
    report = run_generated_pytest([_concern("det.missing")], str(outcomes), str(tmp_path))
    assert not report.passed
    assert "det.missing" in report.stdout


def test_generated_pytest_treats_concern_id_as_data(tmp_path):
    concern_id = "det.x():\n    raise RuntimeError('injected')\n#"
    source = generate([_concern(concern_id)])
    compile(source, "generated.py", "exec")
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(json.dumps({
        concern_id: {"status": "pass", "summary": "", "evidence": {}},
    }))
    report = run_generated_pytest([_concern(concern_id)], str(outcomes), str(tmp_path))
    assert report.passed, report.stderr or report.stdout


def test_generated_pytest_timeout_is_failure(tmp_path):
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text("{}")
    with patch("bia_verifier.pytest_gen.subprocess.run",
               side_effect=__import__("subprocess").TimeoutExpired("pytest", 1)):
        report = run_generated_pytest([_concern("det.slow")], str(outcomes), str(tmp_path),
                                      timeout=1)
    assert not report.passed
    assert report.returncode == 124
    assert "timed out" in report.stderr


def test_cli_grade_writes_rubric_and_pytest_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "optimizer.py").write_text("x = 1\n")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({
        "name": "sample",
        "submission_marker": "optimizer.py",
        "truth": {
            "task": "sample",
            "steps": [{
                "id": "done", "action": "finish", "establishes": "result",
                "weight": 1, "final_outcome": True,
            }],
        },
        "concerns": [
            {"id": "det.present", "layer": "structure", "owner": "deterministic",
             "dimension": "process", "truth_step": "done"},
            {"id": "rub.account", "layer": "judgment", "owner": "rubric",
             "dimension": "process", "truth_step": "done"},
        ],
    }))
    predicates = tmp_path / "predicates.py"
    predicates.write_text(
        "from bia_verifier.pipeline import PredicateOutcome\n"
        "from bia_verifier.schemas import CheckStatus\n"
        "PREDICATES = {'det.present': lambda evidence: PredicateOutcome(CheckStatus.PASS)}\n"
    )
    judgements = tmp_path / "judgements.json"
    judgements.write_text(json.dumps({
        "rub.account": {"verdict": "pass", "score": 0.9, "rationale": "clear account"},
    }))
    out = tmp_path / "out"
    rc = main([
        "grade", "--dataset", str(dataset), "--run", str(run_dir),
        "--predicates", str(predicates), "--judgements", str(judgements),
        "--out", str(out),
    ])
    assert rc == 0
    for name in ("report.json", "reward.json", "outcomes.json", "rubric.json",
                 "pytest_report.json", "pytest.xml", "test_verifier_generated.py"):
        assert (out / name).is_file(), name
    report = json.loads((out / "report.json").read_text())
    rubric_result = next(r for r in report["results"] if r["concern_id"] == "rub.account")
    assert rubric_result["summary"] == "clear account"
    assert rubric_result["evidence"]["score"] == 0.9


def test_cli_rejects_incomplete_precomputed_judgements(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({
        "name": "sample",
        "truth": {"task": "sample", "steps": [{
            "id": "done", "action": "finish", "establishes": "result",
            "weight": 1, "final_outcome": True,
        }]},
        "concerns": [
            {"id": "det.present", "layer": "structure", "owner": "deterministic",
             "dimension": "process", "truth_step": "done"},
            {"id": "rub.account", "layer": "judgment", "owner": "rubric",
             "dimension": "process", "truth_step": "done"},
        ],
    }))
    judgements = tmp_path / "judgements.json"
    judgements.write_text("{}")
    predicates = tmp_path / "predicates.py"
    predicates.write_text(
        "from bia_verifier.pipeline import PredicateOutcome\n"
        "from bia_verifier.schemas import CheckStatus\n"
        "PREDICATES = {'det.present': lambda evidence: PredicateOutcome(CheckStatus.PASS)}\n"
    )
    rc = main(["grade", "--dataset", str(dataset), "--run", str(run_dir),
               "--predicates", str(predicates), "--judgements", str(judgements),
               "--no-pytest"])
    assert rc == 2
    assert "missing=['rub.account']" in capsys.readouterr().err


def test_cli_rejects_invalid_precomputed_verdict(tmp_path, capsys):
    run_dir, dataset, predicates = _verification_fixture(tmp_path)
    judgements = tmp_path / "judgements.json"
    judgements.write_text(json.dumps({"rub.account": {"verdict": "banana"}}))
    rc = main(["grade", "--dataset", str(dataset), "--run", str(run_dir),
               "--predicates", str(predicates), "--judgements", str(judgements),
               "--no-pytest"])
    assert rc == 2
    assert "invalid judgement verdict" in capsys.readouterr().err


def test_cli_requires_complete_predicate_binding(tmp_path, capsys):
    run_dir, dataset, _ = _verification_fixture(tmp_path)
    judgements = tmp_path / "judgements.json"
    judgements.write_text(json.dumps({"rub.account": "pass"}))
    rc = main(["grade", "--dataset", str(dataset), "--run", str(run_dir),
               "--judgements", str(judgements), "--no-pytest"])
    assert rc == 2
    assert "predicate/spec id mismatch" in capsys.readouterr().err


def test_cli_requires_rubric_verification(tmp_path, capsys):
    run_dir, dataset, predicates = _verification_fixture(tmp_path)
    rc = main(["grade", "--dataset", str(dataset), "--run", str(run_dir),
               "--predicates", str(predicates), "--no-pytest"])
    assert rc == 2
    assert "refusing an unjudged grade" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--judge-attempts", "--judge-timeout", "--judge-budget"])
def test_cli_rejects_nonpositive_judge_limits(tmp_path, capsys, flag):
    run_dir, dataset, predicates = _verification_fixture(tmp_path)
    rc = main(["grade", "--dataset", str(dataset), "--run", str(run_dir),
               "--predicates", str(predicates), flag, "0", "--no-pytest"])
    assert rc == 2
    assert "must be at least 1" in capsys.readouterr().err


def test_cli_inconclusive_rubric_exits_failure(tmp_path):
    run_dir, dataset, predicates = _verification_fixture(tmp_path)
    judgements = tmp_path / "judgements.json"
    judgements.write_text(json.dumps({"rub.account": {"verdict": "inconclusive"}}))
    rc = main(["grade", "--dataset", str(dataset), "--run", str(run_dir),
               "--predicates", str(predicates), "--judgements", str(judgements),
               "--no-pytest", "--out", str(tmp_path / "out")])
    assert rc == 1
    reward = json.loads((tmp_path / "out" / "reward.json").read_text())
    assert reward["verification_complete"] is False
    assert reward["consolidated_reward"] is None


def test_core_grade_rejects_missing_rubric_judgement(tmp_path):
    spec = DatasetSpec.from_dict(json.loads(_verification_fixture(tmp_path)[1].read_text()))
    evidence = Evidence("run", RunLayout(str(tmp_path)), None)
    predicates = {"det.present": lambda _: PredicateOutcome(CheckStatus.PASS)}
    with pytest.raises(ValueError, match="missing=.*rub.account"):
        grade(spec, evidence, predicates, {})


def _verification_fixture(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({
        "name": "sample",
        "truth": {"task": "sample", "steps": [{
            "id": "done", "action": "finish", "establishes": "result",
            "weight": 1, "final_outcome": True,
        }]},
        "concerns": [
            {"id": "det.present", "layer": "structure", "owner": "deterministic",
             "dimension": "process", "truth_step": "done"},
            {"id": "rub.account", "layer": "judgment", "owner": "rubric",
             "dimension": "process", "truth_step": "done"},
        ],
    }))
    predicates = tmp_path / "predicates.py"
    predicates.write_text(
        "from bia_verifier.pipeline import PredicateOutcome\n"
        "from bia_verifier.schemas import CheckStatus\n"
        "PREDICATES = {'det.present': lambda evidence: PredicateOutcome(CheckStatus.PASS)}\n"
    )
    return run_dir, dataset, predicates
