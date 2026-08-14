"""bia-verifier CLI.

    python -m bia_verifier.cli selfcheck
    python -m bia_verifier.cli prove   --dataset <spec.json>
    python -m bia_verifier.cli grade   --dataset <spec.json> --run <dir> [--out <dir>]

`selfcheck` runs the acceptance suite in CHECKLIST.md and is the only thing anyone needs to run
to decide whether this pipeline can be trusted. `prove` checks a dataset spec is MECE before any
run is graded, so an unsound instrument fails at authoring time rather than silently mis-scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _selfcheck(argv) -> int:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", os.path.join(ROOT, "tests")],
                       cwd=ROOT, text=True)
    print("\nCHECKLIST:", "ALL ITEMS ENFORCED AND PASSING" if r.returncode == 0
          else "FAILURES ABOVE -- see CHECKLIST.md for what each test enforces")
    return r.returncode


def _prove(args) -> int:
    from .pipeline import DatasetSpec
    from .registry import Registry
    spec = DatasetSpec.load(args.dataset)
    reg = Registry(spec.concerns, spec.truth)
    proof = reg.proof()
    print(json.dumps(proof, indent=1, sort_keys=True))
    if proof["violations"]:
        print("\nDATASET IS NOT MECE -- refusing. Fix the violations above.", file=sys.stderr)
        return 1
    print("\nMECE: sound.")
    return 0


def _grade(args) -> int:
    from .pipeline import DatasetSpec, gather_evidence, grade, write_artifacts
    from .schemas import CheckStatus, Owner
    spec = DatasetSpec.load(args.dataset)
    evidence = gather_evidence(args.run, spec, run_id=args.run_id)

    if args.rubric and args.judgements:
        print("--rubric and --judgements are mutually exclusive", file=sys.stderr)
        return 2
    if args.judge != "none" and not args.rubric:
        print("--judge requires --rubric", file=sys.stderr)
        return 2
    if args.judge_attempts < 1:
        print("--judge-attempts must be at least 1", file=sys.stderr)
        return 2
    if args.judge_timeout < 1:
        print("--judge-timeout must be at least 1 second", file=sys.stderr)
        return 2
    if args.judge_budget < 1:
        print("--judge-budget must be at least 1 character", file=sys.stderr)
        return 2

    predicates: dict = {}
    if args.predicates:
        import importlib.util
        s = importlib.util.spec_from_file_location("bv_predicates", args.predicates)
        if s is None or s.loader is None:
            raise ImportError(f"could not load predicates module: {args.predicates}")
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        predicates = dict(getattr(mod, "PREDICATES", {}))

    judgements: dict[str, CheckStatus] = {}
    judgement_evidence: dict[str, dict] = {}
    judgement_artifact = None
    if args.judgements:
        if not os.path.exists(args.judgements):
            print(f"judgements file not found: {args.judgements}", file=sys.stderr)
            return 2
        with open(args.judgements, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            print("judgements must be a JSON object keyed by concern ID", file=sys.stderr)
            return 2
        for concern_id, value in raw.items():
            if isinstance(value, dict):
                verdict = value.get("verdict", value.get("status"))
                judgement_evidence[concern_id] = dict(value)
            elif isinstance(value, str):
                verdict = value
            else:
                print(f"invalid judgement for {concern_id!r}: expected string or object",
                      file=sys.stderr)
                return 2
            if verdict not in ("pass", "fail", "inconclusive", "not_measured"):
                print(f"invalid judgement verdict for {concern_id!r}: {verdict!r}",
                      file=sys.stderr)
                return 2
            judgements[concern_id] = (CheckStatus(verdict)
                                      if verdict in ("pass", "fail")
                                      else CheckStatus.NOT_MEASURED)
        judgement_artifact = raw

    rubric_ids = [c.id for c in spec.concerns if c.owner is Owner.RUBRIC]
    deterministic_ids = {c.id for c in spec.concerns if c.owner is Owner.DETERMINISTIC}
    if set(predicates) != deterministic_ids:
        print(f"predicate/spec id mismatch: missing={sorted(deterministic_ids - set(predicates))}, "
              f"extra={sorted(set(predicates) - deterministic_ids)}", file=sys.stderr)
        return 2
    if args.judgements:
        supplied = set(judgements)
        expected = set(rubric_ids)
        if supplied != expected:
            print(f"judgement/spec id mismatch: missing={sorted(expected - supplied)}, "
                  f"extra={sorted(supplied - expected)}", file=sys.stderr)
            return 2
    if args.rubric:
        from .judge import LLMJudge, judge_rubric
        from .rubric import Rubric

        rubric = Rubric.load(args.rubric).validate(rubric_ids)
        if args.judge == "none":
            print("--rubric requires --judge codex (or supply --judgements)", file=sys.stderr)
            return 2
        api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY", "") \
            or os.environ.get("KAIJU_CODEX_BRIDGE_SECRET", "")
        if not api_key and not args.allow_unauthenticated_judge:
            print("judge API key is required; set OPENAI_API_KEY or pass --judge-api-key. "
                  "Use --allow-unauthenticated-judge only for a deliberately open local bridge.",
                  file=sys.stderr)
            return 2
        backend = LLMJudge(
            args.judge_url, model=args.judge_model, api_key=api_key,
            attempts=args.judge_attempts, timeout=args.judge_timeout,
            budget=args.judge_budget,
        )
        verdicts = judge_rubric(rubric, evidence.trajectory, backend,
                                budget=args.judge_budget)
        judgement_artifact = {k: v.to_dict() for k, v in verdicts.items()}
        judgements = {k: v.to_status() for k, v in verdicts.items()}
        judgement_evidence = judgement_artifact

    if rubric_ids and not args.rubric and not args.judgements:
        print("dataset declares rubric concerns but neither --rubric nor --judgements was supplied; "
              "refusing an unjudged grade", file=sys.stderr)
        return 2

    report, breakdown = grade(spec, evidence, predicates, judgements, judgement_evidence,
                              task_reward=args.task_reward)
    out = args.out or os.path.join(ROOT, "output", evidence.run_id)
    paths = write_artifacts(out, report, breakdown, judgement_artifact)

    pytest_report = None
    if not args.no_pytest:
        from .pytest_gen import run as run_pytest
        pytest_report = run_pytest(spec.concerns, paths["outcomes.json"], out)
        pytest_path = os.path.join(out, "pytest_report.json")
        with open(pytest_path, "w") as f:
            json.dump(pytest_report.to_dict(), f, indent=1, sort_keys=True)
        paths["pytest_report.json"] = pytest_path
        paths["pytest.xml"] = pytest_report.junit_xml

    print(f"consolidated_reward={breakdown.consolidated_reward}  gate={breakdown.gate}  "
          f"process_score={breakdown.process_score}  task_reward={breakdown.task_reward}")
    for s in report.steps:
        print(f"  {s.truth_step} w={s.weight:.3f} det={s.deterministic_fraction:.3f} "
              f"mult={s.judged_multiplier:.3f} disp={s.disposition:.3f} cum={s.cumulative:.3f}")
    for k, v in paths.items():
        print(f"wrote {v}")
    required_deterministic_ok = all(
        r.status is CheckStatus.PASS or (
            r.status is CheckStatus.NOT_MEASURED
            and not next(c for c in spec.concerns if c.id == r.concern_id).always_measurable
        )
        for r in report.results if r.owner == Owner.DETERMINISTIC.value
    )
    rubric_ok = all(
        r.status is CheckStatus.PASS
        for r in report.results if r.owner == Owner.RUBRIC.value
    )
    pytest_ok = pytest_report is None or pytest_report.passed
    return 0 if required_deterministic_ok and rubric_ok and pytest_ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bia-verifier")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selfcheck", help="run the CHECKLIST.md acceptance suite")

    p = sub.add_parser("prove", help="prove a dataset spec is MECE before grading anything")
    p.add_argument("--dataset", required=True)

    g = sub.add_parser("grade", help="grade one run")
    g.add_argument("--dataset", required=True)
    g.add_argument("--run", required=True)
    g.add_argument("--run-id")
    g.add_argument("--predicates", help="python file exposing PREDICATES: dict[str, callable]")
    g.add_argument("--judgements", help="json of already-computed rubric verdicts")
    g.add_argument("--rubric", help="JSON/YAML rubric bound exactly to rubric-owned concerns")
    g.add_argument("--judge", choices=["none", "codex"], default="none",
                   help="rubric judge backend; codex uses the local OpenAI-compatible bridge")
    g.add_argument("--judge-url", default=os.environ.get("OPENAI_BASE_URL",
                                                          "http://127.0.0.1:8788"))
    g.add_argument("--judge-api-key", default=None,
                   help="bridge secret; defaults to OPENAI_API_KEY or KAIJU_CODEX_BRIDGE_SECRET")
    g.add_argument("--judge-model", default="gpt5.6-sol")
    g.add_argument("--judge-attempts", type=int, default=4)
    g.add_argument("--judge-timeout", type=int, default=180)
    g.add_argument("--judge-budget", type=int, default=60_000,
                   help="maximum transcript characters presented to each rubric criterion")
    g.add_argument("--allow-unauthenticated-judge", action="store_true")
    g.add_argument("--no-pytest", action="store_true",
                   help="skip generated deterministic pytest execution")
    g.add_argument("--task-reward", type=float, default=None)
    g.add_argument("--out")

    args = ap.parse_args(argv)
    if args.cmd == "selfcheck":
        return _selfcheck(args)
    if args.cmd == "prove":
        return _prove(args)
    return _grade(args)


if __name__ == "__main__":
    raise SystemExit(main())
