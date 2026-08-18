"""Convert an agentloop campaign run root into a client delivery bundle.

Source (read only):

    runs/agentloop/<task-uuid>/
        ledger.jsonl  .cfg_iterNN.json  history/
        jobs/agentic_iterNN/<trial-name>/

Destination:

    <out>/<task-uuid>/
        task.toml  instruction.md  environment/  solution/  tests/
        manifest.json
        trajectories/<model-slug>/run_N/
            agent/{trajectory.json, history.md}
            artifacts/optimizer.py
            config.json  result.json
            verifier/{score.json, score.md, grade-stdout.md, test-stdout.md}

Four transforms carry the weight, and each one is recorded in `manifest.json`
rather than applied in silence:

* `verifier_result.rewards` becomes `verifier_result.scores` (and the inner
  `reward` key becomes `score`). Harbor names that key after the reward FILE the
  task's test.sh wrote, so a bundle graded before the task was rewired speaks
  `rewards` while the delivery format speaks `scores`. A trial that already says
  `scores` is left untouched and nothing is recorded for it.
* The verifier filenames have two vocabularies for the same four artifacts. The
  new names (`score.json`, `score.md`, `grade-stdout.md`, `test-stdout.md`) win
  when present; the old ones (`reward.json`, `grade-stdout.txt`,
  `test-stdout.txt`) are the fallback, and `score.md` is synthesised from the
  score when no file carries it. A fallback that would produce an empty file is
  an error, not an empty file.
* Every env var whose name looks like a credential is replaced by `${NAME}`.
  `audit_output` re-walks the finished tree and raises rather than let a literal
  ship; `convert` calls it before it returns.
* `agentic_iterNN` becomes `run_N`, `agent/` is pruned to the trajectory, and the
  optimizer is flattened out of `artifacts/workspace/submission/`.

`rubric_verdicts.json` is deliberately absent: the LLM judge (runner/agentloop/
judge.py) is unwired in this harness, so no verdicts exist to convert. It is
listed in `manifest.json` under `not_produced`.

Stdlib only. The source tree is never written to.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

CONVERTER_VERSION = "1.0.0"

JOB_RE = re.compile(r"^agentic_iter(\d+)$")

# The five paths a delivery root carries from the task bundle.
TASK_BUNDLE_ENTRIES = ("task.toml", "instruction.md", "environment", "solution", "tests")

# Everything this converter owns under `--out`, and therefore everything it may
# remove on a reconvert. A file the operator put there by hand survives.
MANAGED_ENTRIES = ("manifest.json", "trajectories") + TASK_BUNDLE_ENTRIES

DELIVERY_CONFIG_KEYS = ("task", "trial_name", "trials_dir",
                        "agent_setup_timeout_multiplier", "agent",
                        "extra_instruction_paths", "job_id")
DELIVERY_AGENT_KEYS = ("name", "model_name", "env")
DELIVERY_TASK_KEYS = ("path",)

DROPPED_SOURCES = [
    "agent/claude-code.txt",
    "agent/sessions/",
    "agent/setup/",
    "artifacts/manifest.json",
    "artifacts/logs/",
    "artifacts/workspace/ (the optimizer is flattened out of this nesting)",
    "artifacts/workspace/submission/logs/",
    "lock.json",
    "trial.log",
    "<job>/config.json",
    "<job>/lock.json",
    "<job>/result.json",
    "<job>/job.log",
    "<run_root>/ledger.jsonl",
    "<run_root>/.cfg_iterNN.json",
    "<run_root>/.lock",
    "<run_root>/history/iterNN_facts.json",
]

NOT_PRODUCED = [
    "run_N/rubric_verdicts.json: the LLM judge (runner/agentloop/judge.py) is "
    "unwired in this harness, so the campaign produced no rubric verdicts to convert",
]

REASON_REWARDS = ("harbor names the verifier result key after the reward file the "
                  "task's test.sh wrote; the delivery format names it after the score")
REASON_REWARD_KEY = "the delivery format calls the graded value `score`, not `reward`"
REASON_SCORE_JSON = ("synthesised from the pre-rewiring verifier/reward.json, whose "
                     "graded value is keyed `reward`")

PLACEHOLDER_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


class DeliveryError(Exception):
    """The conversion cannot proceed or would produce an incomplete bundle."""


class SecretLeakError(DeliveryError):
    """A credential survived redaction and reached the output tree."""


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------

def is_secret_name(name):
    """True for env var names that carry a credential.

    `SECRET` and `PASSWORD` match anywhere because they appear mid-name
    (`AWS_SECRET_ACCESS_KEY`, `DB_PASSWORD`); `API_KEY` and `TOKEN` match only as
    a trailing component, so `KEYBOARD_LAYOUT` and `MODEL_NAME` stay in the clear.
    """
    upper = str(name).upper()
    if "SECRET" in upper or "PASSWORD" in upper:
        return True
    for tail in ("API_KEY", "TOKEN"):
        if upper == tail or upper.endswith("_" + tail):
            return True
    return False


def redact_secrets(obj, out):
    """Replace every secret-named string value with `${NAME}`.

    Returns `(copy, records)`. The input is not mutated: the source trial is
    read-only and the same parsed object is written twice, once pruned into
    `config.json` and once whole into `result.json`.

    `out` is the destination-relative path of the file being written; it is
    carried into each record so the manifest names the file a redaction applies
    to rather than only the variable.
    """
    records = []

    def walk(node, path):
        if isinstance(node, dict):
            new = {}
            for key, value in node.items():
                where = f"{path}.{key}" if path else str(key)
                if isinstance(value, str) and is_secret_name(key):
                    placeholder = "${%s}" % key
                    new[key] = placeholder
                    records.append({"env_var": str(key), "file": out,
                                    "json_path": where, "placeholder": placeholder})
                else:
                    new[key] = walk(value, where)
            return new
        if isinstance(node, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(node)]
        return node

    return walk(obj, ""), records


def _secret_fields(obj, path=""):
    """Every `(json_path, name, value)` whose key names a credential."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            where = f"{path}.{key}" if path else str(key)
            if isinstance(value, str) and is_secret_name(key):
                yield where, str(key), value
            else:
                yield from _secret_fields(value, where)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _secret_fields(value, f"{path}[{i}]")


def _scan(name, data, forbidden):
    """Raise if one delivered file carries a credential.

    The structural check runs before the literal check so the error names the
    variable that leaked. A literal can appear anywhere -- prose in a trajectory,
    a shell transcript -- where no key names it, and that is what the second
    check is for.
    """
    text = data.decode("utf-8", "replace")
    if name.endswith(".json"):
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if obj is not None:
            for where, var, value in _secret_fields(obj):
                if value and not PLACEHOLDER_RE.fullmatch(value):
                    raise SecretLeakError(
                        f"{name}: {var} at {where} is not templatized; "
                        f"expected ${{{var}}}")
    for value in sorted(forbidden):
        if value and value in text:
            raise SecretLeakError(
                f"{name}: carries a literal collected from a secret-named source "
                f"field ({len(value)} chars, starts {value[:4]!r})")


def audit_output(root, forbidden_values=()):
    """Walk a finished delivery tree and raise on any surviving credential.

    Called by `convert` before it reports success, so a redaction gap aborts the
    conversion instead of shipping the tree.
    """
    root = Path(root)
    forbidden = {v for v in forbidden_values if v}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        _scan(str(path), path.read_bytes(), forbidden)


def _collect_secret_values(run_root, trials):
    """The literal credential values the source campaign holds.

    Collected from the source, never from the redacted copy, so the audit stays
    honest when the redaction itself is broken. Values already written as
    `${NAME}` are skipped: they are the placeholder the output is supposed to
    contain.
    """
    paths = sorted(run_root.glob(".cfg_iter*.json"))
    for trial in trials:
        paths += [trial / "config.json", trial / "result.json", trial / "lock.json",
                  trial.parent / "config.json", trial.parent / "lock.json"]
    values = set()
    for path in paths:
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for _, _, value in _secret_fields(obj):
            if value and not PLACEHOLDER_RE.fullmatch(value):
                values.add(value)
    return values


# --------------------------------------------------------------------------
# reading the source campaign
# --------------------------------------------------------------------------

def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except OSError as exc:
        raise DeliveryError(f"cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise DeliveryError(f"{path} is not valid JSON: {exc}") from exc


def _json_bytes(obj):
    """The delivery format's JSON: four-space indent, no trailing newline."""
    return json.dumps(obj, indent=4).encode()


def _iteration_jobs(jobs_dir):
    """Every `agentic_iterNN` job dir as `(N, dir)`, ordered by N."""
    jobs = []
    for child in jobs_dir.iterdir():
        match = JOB_RE.match(child.name)
        if match and child.is_dir():
            jobs.append((int(match.group(1)), child))
    return sorted(jobs)


def _newest_trial(job_dir):
    """The trial this job produced, or None.

    A job dir is reused across relaunches, so more than one trial can survive in
    it; the newest by mtime is the one the iteration recorded, and the name
    breaks a tie so a reconvert picks the same one.
    """
    cands = []
    for child in sorted(job_dir.iterdir()):
        if child.is_dir() and (child / "result.json").is_file():
            cands.append((child.stat().st_mtime, child.name, child))
    if not cands:
        return None
    return max(cands)[2]


def _model_slug(trials):
    """The delivery's model directory name, read from the trials, not assumed.

    Two trials of one campaign disagreeing about the model means the run root
    holds two campaigns, and one `trajectories/<slug>/` would misattribute half
    of them.
    """
    names = sorted({_model_name(t) for t in trials})
    if len(names) > 1:
        raise DeliveryError(f"trials disagree about the model: {names}")
    slug = re.sub(r"[^a-z0-9]+", "-", names[0].lower()).strip("-")
    if not slug:
        raise DeliveryError(f"model name {names[0]!r} slugifies to nothing")
    return slug


def _model_name(trial):
    result = _read_json(trial / "result.json")
    info = (result.get("agent_info") or {}).get("model_info") or {}
    name = info.get("name")
    if not name:
        raise DeliveryError(
            f"{trial}/result.json has no agent_info.model_info.name; the model "
            f"directory cannot be derived")
    return str(name)


# --------------------------------------------------------------------------
# per-run planning
# --------------------------------------------------------------------------

def _prune_config(cfg):
    """The delivery key set, in the delivery's order, and nothing else."""
    out = {}
    for key in DELIVERY_CONFIG_KEYS:
        if key not in cfg:
            continue
        value = cfg[key]
        if key == "task" and isinstance(value, dict):
            value = {k: value[k] for k in DELIVERY_TASK_KEYS if k in value}
        elif key == "agent" and isinstance(value, dict):
            value = {k: value[k] for k in DELIVERY_AGENT_KEYS if k in value}
        out[key] = value
    return out


def _normalise_result(result, run):
    """Rename `verifier_result.rewards` to `.scores`, and its inner key with it."""
    verifier = result.get("verifier_result")
    if not isinstance(verifier, dict) or "rewards" not in verifier or "scores" in verifier:
        return result, []

    renamed = [{"run": run, "from": "verifier_result.rewards",
                "to": "verifier_result.scores", "reason": REASON_REWARDS}]
    new = {}
    for key, value in verifier.items():
        if key != "rewards":
            new[key] = value
            continue
        if isinstance(value, dict) and "reward" in value and "score" not in value:
            value = {("score" if k == "reward" else k): v for k, v in value.items()}
            renamed.append({"run": run, "from": "verifier_result.rewards.reward",
                            "to": "verifier_result.scores.score",
                            "reason": REASON_REWARD_KEY})
        new["scores"] = value

    out = dict(result)
    out["verifier_result"] = new
    return out, renamed


def _plan_verifier(trial, run, dest, files, renamed, fallbacks, headers, finished_at):
    """The four delivered verifier files, preferring the new names.

    Every fallback is recorded; a source that exists but is empty is refused,
    because an empty `score.md` in a delivery reads as a score of nothing rather
    than as a missing file.
    """
    vdir = trial / "verifier"

    score_src = vdir / "score.json"
    if score_src.is_file():
        score_bytes = score_src.read_bytes()
    else:
        score_src = vdir / "reward.json"
        if not score_src.is_file():
            raise DeliveryError(
                f"{run}: {vdir} has neither score.json nor reward.json; the "
                f"trial was never graded")
        obj = _read_json(score_src)
        if "reward" in obj and "score" not in obj:
            obj = {("score" if k == "reward" else k): v for k, v in obj.items()}
            renamed.append({"run": run, "from": "verifier/score.json:reward",
                            "to": "verifier/score.json:score",
                            "reason": REASON_SCORE_JSON})
        score_bytes = (json.dumps(obj, indent=1) + "\n").encode()
        fallbacks.append({"run": run, "destination": "verifier/score.json",
                          "source": str(score_src), "synthesised": False})
    _require_content(score_bytes, run, "verifier/score.json", score_src)
    files[f"{dest}/verifier/score.json"] = score_bytes

    md_src = vdir / "score.md"
    if md_src.is_file() and md_src.read_bytes().strip():
        md_bytes = md_src.read_bytes()
    else:
        score = json.loads(score_bytes.decode()).get("score")
        if score is None:
            raise DeliveryError(
                f"{run}: no verifier/score.md and score.json carries no `score` "
                f"to synthesise one from")
        md_bytes = f"{score}\n".encode()
        fallbacks.append({"run": run, "destination": "verifier/score.md",
                          "source": str(score_src), "synthesised": True})
    _require_content(md_bytes, run, "verifier/score.md", md_src)
    files[f"{dest}/verifier/score.md"] = md_bytes

    for stem in ("grade-stdout", "test-stdout"):
        src = vdir / f"{stem}.md"
        if not src.is_file():
            src = vdir / f"{stem}.txt"
            if not src.is_file():
                raise DeliveryError(
                    f"{run}: {vdir} has neither {stem}.md nor {stem}.txt")
            fallbacks.append({"run": run, "destination": f"verifier/{stem}.md",
                              "source": str(src), "synthesised": False})
        data = src.read_bytes()
        _require_content(data, run, f"verifier/{stem}.md", src)
        if stem == "test-stdout":
            header = _test_stdout_header(dest, data, finished_at)
            if header is not None:
                data = header + data
                headers.append({"run": run,
                                "destination": f"verifier/{stem}.md",
                                "timestamp_source": "result.json:finished_at",
                                "timestamp": finished_at})
        files[f"{dest}/verifier/{stem}.md"] = data


STDOUT_HEADER_PREFIX = "# pytest "


def _iso8601z(value):
    """Normalise a harbor timestamp to second-precision UTC (`...THH:MM:SSZ`)."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pytest_ran(data):
    """True when the transcript carries a pytest summary line, not a skip notice."""
    tail = data.decode("utf-8", "replace")
    return bool(re.search(r"^=+ .*\d+ (passed|failed|error)", tail, re.M))


def _test_stdout_header(dest, data, finished_at):
    """Two provenance lines for `verifier/test-stdout.md`, or None if already present.

    The timestamp comes from the trial's own `finished_at` rather than the wall
    clock, so reconverting the same campaign reproduces the same bytes.

    The second line states what actually produced the transcript. A bundle whose
    image has no pytest gets a notice saying so; claiming a gate that never ran
    would be the false-provenance defect this format exists to surface.
    """
    first = data.lstrip().split(b"\n", 1)[0]
    if first.startswith(STDOUT_HEADER_PREFIX.encode()) and b"  target: " in first:
        return None
    stamp = finished_at or "unknown-time"
    if _pytest_ran(data):
        provenance = ("outcomes emitted by grade.py, asserted by the bundle's "
                      "own tests/test_output.py")
    else:
        provenance = ("pytest unavailable in the task image; grade.py is the "
                      "sole verifier for this run")
    return (f"{STDOUT_HEADER_PREFIX}{stamp}  target: {dest}\n"
            f"# {provenance}\n\n").encode()


def _require_content(data, run, destination, source):
    if not data.strip():
        raise DeliveryError(
            f"{run}: {destination} would be empty; its source {source} holds no "
            f"content and an empty verifier file must never be delivered")


def _plan_run(trial, run, dest, files, renamed, redacted, fallbacks, headers):
    cfg = _read_json(trial / "config.json")
    result = _read_json(trial / "result.json")

    trajectory = trial / "agent" / "trajectory.json"
    if not trajectory.is_file():
        raise DeliveryError(
            f"{run}: {trajectory} is missing; harbor writes it at trial end, so "
            f"the trial did not complete")
    files[f"{dest}/agent/trajectory.json"] = trajectory.read_bytes()

    # The injected history is the agent's own prior-attempt briefing and is the
    # one extra_instruction the delivery carries. Iteration 1 has none.
    for injected in (cfg.get("extra_instruction_paths") or [])[:1]:
        path = Path(injected)
        if not path.is_file():
            raise DeliveryError(
                f"{run}: config.json injects {path}, which no longer exists")
        files[f"{dest}/agent/history.md"] = path.read_bytes()

    optimizer = _find_optimizer(trial)
    files[f"{dest}/artifacts/optimizer.py"] = optimizer.read_bytes()

    cfg_out, records = redact_secrets(_prune_config(cfg), f"{dest}/config.json")
    redacted.extend(records)
    files[f"{dest}/config.json"] = _json_bytes(cfg_out)

    result_out, renames = _normalise_result(result, run)
    renamed.extend(renames)
    result_out, records = redact_secrets(result_out, f"{dest}/result.json")
    redacted.extend(records)
    files[f"{dest}/result.json"] = _json_bytes(result_out)

    _plan_verifier(trial, run, dest, files, renamed, fallbacks, headers,
                   _iso8601z(result.get("finished_at")))


def _find_optimizer(trial):
    """The submitted optimizer, shallowest first.

    Working copies and checkpoints accumulate deeper under the same tree, so the
    shallowest `submission/optimizer.py` is the submission.
    """
    for pattern in ("artifacts/**/submission/optimizer.py", "artifacts/**/optimizer.py"):
        found = sorted(trial.glob(pattern), key=lambda p: (len(p.parts), str(p)))
        if found:
            return found[0]
    raise DeliveryError(f"{trial}: no artifacts/**/optimizer.py; nothing was submitted")


def _has_submission(trial):
    return any(
        True
        for pattern in ("artifacts/**/submission/optimizer.py", "artifacts/**/optimizer.py")
        for _ in trial.glob(pattern))


def _plan_task_bundle(task_dir, files):
    """The five task-bundle paths the delivery root carries."""
    if not task_dir.is_dir():
        raise DeliveryError(
            f"the task bundle {task_dir} named by the trial config no longer "
            f"exists; the delivery root cannot be populated")
    for entry in TASK_BUNDLE_ENTRIES:
        src = task_dir / entry
        if src.is_file():
            files[entry] = src.read_bytes()
        elif src.is_dir():
            for path in sorted(p for p in src.rglob("*") if p.is_file()):
                files[f"{entry}/{path.relative_to(src).as_posix()}"] = path.read_bytes()


def _task_dir(trial):
    cfg = _read_json(trial / "config.json")
    path = (cfg.get("task") or {}).get("path")
    if not path:
        raise DeliveryError(f"{trial}/config.json has no task.path")
    return Path(path)


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def _prepare_out(out_dir, force):
    """Clear what this converter owns, refusing to clobber anything else.

    `out_dir` is one task's bundle, not the delivery root, so a sibling task's
    bundle is never in scope. Within it only `MANAGED_ENTRIES` are removed, so a
    reconvert drops stale runs without touching a file an operator added. A
    non-empty bundle still needs `--force`, because the caller may have meant a
    different directory.
    """
    if out_dir.exists():
        if any(out_dir.iterdir()) and not force:
            raise DeliveryError(
                f"{out_dir} is not empty; pass --force to overwrite the delivery "
                f"it holds")
        for name in MANAGED_ENTRIES:
            target = out_dir / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _assert_campaign_finished(run_root):
    """Refuse to package a campaign that is still running.

    The loop holds an exclusive flock on `<run_root>/.lock` for the whole of a
    refine(). Converting underneath it would package a trial mid-write, and the
    resulting failure would name a missing artifact rather than the live run
    that is about to produce it.
    """
    lock = run_root / ".lock"
    if not lock.exists():
        return
    with open(lock, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise DeliveryError(
                f"{run_root} is locked by a running campaign; package it once "
                f"the loop exits (check with: fuser {lock})") from None
        fcntl.flock(handle, fcntl.LOCK_UN)


def convert(run_root, out, force=False, dry_run=False):
    """Convert one campaign run root into a delivery bundle, returning the manifest.

    The whole tree is planned in memory before a byte is written, so a dry run
    reports exactly what a real run would produce and a failure part-way through
    planning leaves no half-written delivery.
    """
    run_root = Path(run_root)
    # Keyed by task uuid so one delivery root holds many tasks, and --force
    # clears only this task's bundle rather than a sibling's.
    out_dir = Path(out) / run_root.name

    if not run_root.is_dir():
        raise DeliveryError(f"run root does not exist: {run_root}")
    jobs_dir = run_root / "jobs"
    if not jobs_dir.is_dir():
        raise DeliveryError(f"run root has no jobs/ directory: {run_root}")
    _assert_campaign_finished(run_root)

    sources, skipped = [], []
    for number, job_dir in _iteration_jobs(jobs_dir):
        trial = _newest_trial(job_dir)
        if trial is None:
            print(f"WARNING: {job_dir} holds no trial with a result.json; "
                  f"iteration {number} is not delivered", file=sys.stderr)
            continue
        if not _has_submission(trial):
            print(f"WARNING: {trial} submitted no optimizer; iteration {number} is "
                  f"not delivered", file=sys.stderr)
            skipped.append({"iteration": number, "source_job": job_dir.name,
                            "source_trial_dir": str(trial),
                            "reason": "no optimizer submitted"})
            continue
        sources.append((number, job_dir, trial))
    if not sources:
        raise DeliveryError(f"no convertible iteration under {jobs_dir}")

    trials = [t for _, _, t in sources]
    forbidden = _collect_secret_values(run_root, trials)
    model_slug = _model_slug(trials)

    files = {}
    runs, renamed, redacted, fallbacks, headers = [], [], [], [], []
    for number, job_dir, trial in sources:
        run = f"run_{number}"
        dest = f"trajectories/{model_slug}/{run}"
        _plan_run(trial, run, dest, files, renamed, redacted, fallbacks, headers)
        runs.append({
            "run": run,
            "source_job": job_dir.name,
            "source_trial_dir": str(trial),
            "source_result_sha256": hashlib.sha256(
                (trial / "result.json").read_bytes()).hexdigest(),
            "destination": dest,
        })

    _plan_task_bundle(_task_dir(trials[0]), files)

    manifest = {
        "converter_version": CONVERTER_VERSION,
        "generated_at": _utc_now(),
        "source_run_root": str(run_root),
        "task_uuid": run_root.name,
        "model_slug": model_slug,
        "runs": runs,
        "renamed": renamed,
        "redacted": redacted,
        "fallbacks": fallbacks,
        "headers_added": headers,
        "dropped": list(DROPPED_SOURCES),
        "not_produced": list(NOT_PRODUCED),
        "skipped": skipped,
    }
    files["manifest.json"] = _json_bytes(manifest)

    if dry_run:
        for name, data in sorted(files.items()):
            _scan(name, data, forbidden)
        return manifest

    _prepare_out(out_dir, force)
    for rel, data in sorted(files.items()):
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    audit_output(out_dir, forbidden_values=forbidden)
    return manifest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _report(manifest, dry_run):
    print(f"{manifest['task_uuid']}/trajectories/{manifest['model_slug']}/"
          f" ({len(manifest['runs'])} runs)"
          + ("  [dry run, nothing written]" if dry_run else ""))
    by_run = {}
    for row in manifest["fallbacks"]:
        by_run.setdefault(row["run"], []).append(row)
    for row in manifest["runs"]:
        print(f"  {row['run']}  {row['source_job']}  "
              f"{Path(row['source_trial_dir']).name} -> {row['destination']}")
        for fb in by_run.get(row["run"], []):
            how = "synthesised from" if fb["synthesised"] else "fell back to"
            print(f"    {fb['destination']}  {how}  {Path(fb['source']).name}")
    print(f"  redacted {len(manifest['redacted'])} secret fields, "
          f"renamed {len(manifest['renamed'])} fields, "
          f"used {len(manifest['fallbacks'])} verifier fallbacks")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert an agentloop campaign run root into a client delivery bundle.")
    parser.add_argument("--run-root", required=True,
                        help="the campaign run root, runs/agentloop/<task-uuid>")
    parser.add_argument("--out", required=True, help="the delivery directory to write")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan and report the conversion without writing anything")
    parser.add_argument("--force", action="store_true",
                        help="overwrite a delivery that already exists in --out")
    args = parser.parse_args(argv)
    try:
        manifest = convert(args.run_root, args.out,
                           force=args.force, dry_run=args.dry_run)
    except DeliveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _report(manifest, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
