#!/usr/bin/env python3
"""Outer orchestrator: resolve task -> mount -> dispatch -> grade -> ingest.

Backends:
    harbor  - shells out to `harbor run` with the mounted task dir
    local   - shells out to `torchrun` directly (no Harbor, no container)
    dry     - fabricates a synthetic Harbor-shape result for wiring tests

Task selection (in order):
    --task <path>    absolute or ./relative path to a task dir
    --task <name>    resolved under <harness_root>/tasks/<name>
    --task <uuid>    matched against dir names AND [task].name / [task].uuid in tasks/*/task.toml

LLM config (harbor backend):
    --llm-config proxy/claude-code-oauth.json
        JSON with: model, base_url, api_key, timeout, num_retries
        Everything else (allowlist host, env vars, model flag) is derived internally.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

HARNESS_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = HARNESS_ROOT / "tasks"
DEFAULT_LEDGER = HARNESS_ROOT / "runs.jsonl"
LEGACY_HARNESS2 = HARNESS_ROOT / "legacy" / "harness2"

sys.path.insert(0, str(HARNESS_ROOT / "runner"))
# legacy/harness2 modules are imported flat (`import orchestrator`), not as a package.
sys.path.insert(0, str(LEGACY_HARNESS2))
from mount_variant import mount_task  # noqa: E402
from ingest_result import normalize, append  # noqa: E402
from agentloop.marking import mark_reward_payload  # noqa: E402


def resolve_task(task_arg: str) -> Path:
    p = Path(task_arg)
    if p.exists() and p.is_dir():
        return p.resolve()
    candidate = TASKS_ROOT / task_arg
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()
    matches: list[Path] = []
    if TASKS_ROOT.is_dir():
        for sub in sorted(TASKS_ROOT.iterdir()):
            if not sub.is_dir():
                continue
            toml_path = sub / "task.toml"
            if not toml_path.is_file():
                continue
            try:
                with toml_path.open("rb") as f:
                    data = tomllib.load(f)
            except Exception:
                continue
            meta = data.get("task", {})
            candidates = {meta.get("name"), meta.get("uuid"), data.get("name"), data.get("uuid")}
            if task_arg in candidates:
                matches.append(sub.resolve())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # SPEC_AUDIT R3/D3: ambiguous qualified-name must raise, not silently pick first.
        listing = "\n  ".join(str(m) for m in matches)
        raise ValueError(
            f"task selector '{task_arg}' is ambiguous; matches {len(matches)} tasks:\n  {listing}\n"
            "Disambiguate with a full path or a unique uuid."
        )
    raise FileNotFoundError(
        f"task not found: '{task_arg}' (tried as path, as dir name under {TASKS_ROOT}, "
        f"and against name/uuid in [task] or top level of each tasks/*/task.toml)"
    )


def read_task_id(task_dir: Path) -> str:
    with (task_dir / "task.toml").open("rb") as f:
        data = tomllib.load(f)
    name = data.get("task", {}).get("name") or data.get("name")
    if not name:
        raise ValueError(f"task.toml in {task_dir} missing task name (looked in [task].name and top-level name)")
    return name


def _slugify(task_id: str) -> str:
    return task_id.replace("/", "-")


def resolve_policy_root(task_dir: Path, harness_root: Path = HARNESS_ROOT) -> Path:
    task_id = read_task_id(task_dir)
    return harness_root / "policy" / _slugify(task_id)


def resolve_ledger_path(task_id: str, harness_root: Path = HARNESS_ROOT) -> Path:
    return harness_root / "runs" / _slugify(task_id) / "runs.jsonl"


def resolve_task_uuid(task_dir: Path) -> str:
    with (task_dir / "task.toml").open("rb") as f:
        data = tomllib.load(f)
    meta = data.get("task", {})
    uuid_val = meta.get("uuid") or data.get("uuid")
    if uuid_val:
        return str(uuid_val)
    name = meta.get("name") or data.get("name")
    if not name:
        raise ValueError(f"task.toml in {task_dir} missing both [task].uuid and [task].name")
    return _slugify(name)


def resolve_work_root(task_dir: Path, harness_root: Path = HARNESS_ROOT) -> Path:
    return harness_root / "work" / resolve_task_uuid(task_dir)


class VariantDeliveryImpossible(RuntimeError):
    """Raised when a --variant provably cannot reach the trial container.

    Harbor uploads a fixed set of paths into a trial: the workdir, the
    writable `_mount_targets()`, `/tests`, `/solution`, and the task's
    `environment/` dir (harbor/environments/islo.py, ~L918-922). The task
    root is NEVER mapped onto `/workspace`.

    For tasks that ship `/workspace` inside a prebuilt docker image (e.g.
    bia/track3nov:v2), `mount_task(..., variant=...)` copies the file into the
    mounted task tree and harbor then leaves it behind: the container trains
    whatever was baked into the image. The run reports success while grading
    something unrelated to the variant. A silent wrong result is worse than a
    crash, so the harness refuses the run instead.
    """


WORKSPACE_ARTIFACT_PREFIX = "/workspace/"


def _workspace_ships_in_image(task_dir: Path) -> bool:
    """True when the task's /workspace comes from its docker image, not the mount.

    The signal is a task.toml that BOTH pins `[environment].docker_image` and
    declares at least one `artifacts` entry under `/workspace/`: the task grades
    a path the mounted tree does not own. Never raises -- a missing or malformed
    task.toml simply yields False, leaving existing behaviour untouched.
    """
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return False
    try:
        with task_toml.open("rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False

    environment = data.get("environment")
    if not isinstance(environment, dict) or not environment.get("docker_image"):
        return False

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    return any(
        isinstance(a, str) and a.startswith(WORKSPACE_ARTIFACT_PREFIX)
        for a in artifacts
    )


def load_llm_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    for k in ("model", "base_url", "api_key"):
        if k not in data:
            raise ValueError(f"llm-config {path} missing required field: {k}")
    return data


def resolve_verifier_dir(task_dir: Path) -> Path | None:
    """Return tasks/<task>/verifier/ if it holds a complete BIA asset set, else None."""
    vd = task_dir / "verifier"
    if not vd.is_dir():
        return None
    if not (vd / "dataset.json").is_file():
        return None
    if not (vd / "predicates.py").is_file():
        return None
    return vd


def bia_grade_seed(seed_work_dir: Path, verifier_dir: Path, *,
                   task_reward: float | None, run_id: str,
                   codex_bridge_url: str,
                   codex_bridge_key: str | None,
                   judge_model: str = "gpt5.6-sol",
                   precomputed_judgements: Path | None = None) -> dict:
    """Invoke `python -m bia_verifier.cli grade` for one seed. Returns parsed reward.json.

    SPEC_AUDIT R1: reward = consolidated_reward on success.
    SPEC_AUDIT D2 + \u00a77.2: verification_complete=false surfaces via reward.json;
    caller MUST demote status to 'verification_incomplete' and reward=null.

    When precomputed_judgements is set, the BIA CLI is invoked with --judgements
    instead of --rubric + --judge codex; used for deterministic E2E harness runs
    when a live Codex bridge is unavailable. Production runs leave it None so
    the codex rubric judge is used (SPEC_AUDIT R14).
    """
    out_dir = seed_work_dir / "verifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "bia_verifier.cli", "grade",
        "--dataset", str(verifier_dir / "dataset.json"),
        "--predicates", str(verifier_dir / "predicates.py"),
        "--run", str(seed_work_dir),
        "--run-id", run_id,
        "--out", str(out_dir),
    ]
    rubric_path = verifier_dir / "rubric.json"
    if precomputed_judgements is not None:
        cmd += ["--judgements", str(precomputed_judgements)]
    elif rubric_path.is_file():
        cmd += [
            "--rubric", str(rubric_path),
            "--judge", "codex",
            "--judge-url", codex_bridge_url,
            "--judge-model", judge_model,
        ]
        if codex_bridge_key:
            cmd += ["--judge-api-key", codex_bridge_key]
    if task_reward is not None:
        cmd += ["--task-reward", str(task_reward)]

    env = os.environ.copy()
    # bia_verifier now lives under legacy/harness2/, so cwd no longer makes it
    # importable: this entry is load-bearing and must not be skipped by setdefault.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(LEGACY_HARNESS2), str(HARNESS_ROOT), env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(cmd, cwd=str(HARNESS_ROOT), capture_output=True, text=True,
                          env=env, check=False)
    reward_json_path = out_dir / "reward.json"
    if not reward_json_path.is_file():
        raise RuntimeError(
            f"bia_verifier.cli grade did not write reward.json to {out_dir} "
            f"(rc={proc.returncode}); stderr={proc.stderr[:500]}"
        )
    reward_payload = json.loads(reward_json_path.read_text())
    reward_payload["_bia_grade_rc"] = proc.returncode
    reward_payload["_bia_grade_stderr"] = proc.stderr[-2000:] if proc.stderr else ""
    reward_payload["_verifier_dir"] = str(out_dir)
    return reward_payload


def _grade(mounted_task: Path, log_path: Path) -> dict:
    grader = mounted_task / "tests" / "grader.py"
    if not grader.is_file():
        raise FileNotFoundError(f"grader.py missing in mounted task: {grader}")
    proc = subprocess.run(
        [sys.executable, str(grader), str(log_path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def dispatch_dry(mounted_task: Path, seed: int, work_dir: Path,
                 run_name: str, **_kwargs) -> tuple[Path, dict]:
    log = work_dir / "runs" / f"{run_name}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"logs/{run_name}_seed{seed}.txt",
        f"Running task {mounted_task.name} with dry backend (synthetic, seed={seed}).",
    ]
    jitter = ((seed * 2654435761) & 0xFFFF) / 0xFFFF * 0.002 - 0.001
    for step in range(250, 3001, 250):
        val = round(4.0 - 0.00025 * step + jitter, 5)
        lines.append(f"step:{step}/3000 val_loss:{val} train_time:{step * 0.35:.3f}s step_avg:{350:.2f}ms")
    log.write_text("\n".join(lines) + "\n")

    trajectory = {
        "schema": "dry-synthetic-v1",
        "run_name": run_name,
        "seed": seed,
        "steps": [
            {"role": "user", "content": (
                f"Task: {mounted_task.name}. Train a nanoGPT variant on FineWeb-10B "
                f"and cross val_loss=3.28 by step 2500 (stretch) or 3500 (baseline). "
                f"Only modify Optimization / Init & Optim Hyperparams sections; keep the "
                f"Model + distributed_data_generator sections byte-identical; one fwd/bwd "
                f"per step; third-party optimizers must be copied in whole (no new pip "
                f"install); hardcode hyperparameters (no CLI args)."
            )},
            {"role": "assistant", "content": (
                f"Executed the mounted variant on the dry backend (seed={seed}). "
                f"Trainer completed all 3000 steps without NaN or CUDA errors. "
                f"Final val_loss={round(4.0 - 0.00025 * 3000 + jitter, 5)}, "
                f"step_to_3_28=3000. No changes to restricted sections; single fwd/bwd "
                f"per step preserved; no CLI args introduced; no new pip installs. "
                f"For statistical significance under 2-seed reproduction "
                f"(Rule 5, (3.28 - mean_val_loss)*sqrt(N) >= 0.004), the reward emitter "
                f"reports stat_sig_at_n1 alongside the headline; multi-seed aggregation "
                f"is deferred to the run_loop's summary stage as designed."
            )},
        ],
    }
    (work_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")

    return log, {"status": "success", "returncode": 0, "is_synthetic": True}


def dispatch_local(mounted_task: Path, seed: int, work_dir: Path,
                   run_name: str, **_kwargs) -> tuple[Path, dict]:
    env_dir = mounted_task / "environment"
    trainer = env_dir / "train_gpt_simple.py"
    if not trainer.is_file():
        raise FileNotFoundError(
            f"trainer missing at {trainer} - mount.toml [shared]/[variant] did not populate it"
        )

    log = work_dir / "runs" / f"{run_name}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    nproc = _detect_gpu_count()
    if nproc == 0:
        raise RuntimeError("--backend local requires at least one CUDA GPU; none detected")

    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", str(trainer)]
    env = os.environ.copy()
    env["BIA_HARNESS_SEED"] = str(seed)
    env["SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=env_dir, stdout=f, stderr=subprocess.STDOUT,
                              env=env, check=False)
    return log, {"status": "success" if proc.returncode == 0 else "error",
                 "returncode": proc.returncode}


def resolve_harbor_bin() -> str:
    """Pick the harbor executable, preferring an explicit one over bare PATH.

    A bare "harbor" resolves through PATH, which on this host hits a stale uv-tool
    install (0.13.2) whose docker provider declares neither `gpus` nor
    `network_allowlist` -- so any GPU/allowlist task dies at capability validation
    even when a patched harbor is present in a venv. Order: $HARBOR_BIN, then the
    sibling .venv-harbor, then PATH.
    """
    explicit = os.environ.get("HARBOR_BIN")
    if explicit:
        return explicit
    for candidate in (
        Path.home() / "oer" / ".venv-harbor" / "bin" / "harbor",
        HARNESS_ROOT.parent / ".venv-harbor" / "bin" / "harbor",
    ):
        if candidate.is_file():
            return str(candidate)
    return "harbor"


def build_harbor_cmd(mounted_task: Path, seed: int, jobs_dir: Path, *,
                     llm_config: dict | None, agent: str, attempts: int,
                     concurrent: int) -> list[str]:
    # --export-traces is load-bearing: Harbor defaults to --no-export-traces, and without
    # trial_dir/trajectory.json the BIA verifier flags every seed verification_incomplete.
    cmd = [
        resolve_harbor_bin(), "run",
        "-p", str(mounted_task),
        "-a", agent,
        "-k", str(attempts),
        "-n", str(concurrent),
        "--env", "docker",
        "--jobs-dir", str(jobs_dir),
        "--export-traces",
        "--ae", f"BIA_HARNESS_SEED={seed}",
        "--ae", f"SEED={seed}",
        "--ae", f"PYTHONHASHSEED={seed}",
    ]
    if llm_config:
        from llm_client import _strip_provider_prefix

        container_url = llm_config["base_url"]
        cmd += ["-m", _strip_provider_prefix(llm_config["model"])]
        cmd += ["--ae", f"ANTHROPIC_BASE_URL={container_url}"]
        cmd += ["--ae", f"ANTHROPIC_API_KEY={llm_config['api_key']}"]
        if "timeout" in llm_config:
            cmd += ["--ae", f"LITELLM_TIMEOUT={llm_config['timeout']}"]
        if "num_retries" in llm_config:
            cmd += ["--ae", f"LITELLM_NUM_RETRIES={llm_config['num_retries']}"]
        host = urlparse(container_url).hostname
        if host:
            cmd += ["--allow-agent-host", host]
    return cmd


def _pick_harbor_log(trial_dir: Path, agent: str) -> Path | None:
    agent_txt = trial_dir / "agent" / f"{agent}.txt"
    if agent_txt.is_file():
        return agent_txt
    agent_dir = trial_dir / "agent"
    if agent_dir.is_dir():
        txts = sorted(agent_dir.glob("*.txt"))
        if txts:
            return txts[-1]
    runs_dir = trial_dir / "artifacts" / "runs"
    if runs_dir.exists():
        logs = sorted(runs_dir.glob("*.log"))
        if logs:
            return logs[-1]
    all_logs = sorted(trial_dir.rglob("*.log"))
    return all_logs[-1] if all_logs else None


def dispatch_harbor(mounted_task: Path, seed: int, work_dir: Path,
                    run_name: str, *, llm_config: dict | None = None,
                    agent: str = "claude-code", attempts: int = 1,
                    concurrent: int = 1,
                    jobs_dir: Path | None = None) -> tuple[Path | None, dict]:
    if jobs_dir is None:
        jobs_dir = work_dir / "jobs"
    else:
        jobs_dir = jobs_dir / f"{run_name}_seed{seed}"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    if llm_config:
        if agent == "nop":
            print("WARN: --llm-config set but --agent=nop (the no-op agent never calls the LLM). "
                  "Use --agent claude-code (or another litellm-based Harbor agent).",
                  file=sys.stderr)

    cmd = build_harbor_cmd(mounted_task, seed, jobs_dir,
                           llm_config=llm_config, agent=agent,
                           attempts=attempts, concurrent=concurrent)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    result_files = sorted(jobs_dir.rglob("result.json"))
    if not result_files:
        raise RuntimeError(f"harbor run produced no result.json; stderr={proc.stderr[:500]}")

    # The job-level result.json sits beside the per-trial ones and carries no reward.
    trial_results = [p for p in result_files if _is_trial_result(p)]
    chosen = trial_results[-1] if trial_results else result_files[-1]
    trial_dir = chosen.parent
    log_path = _pick_harbor_log(trial_dir, agent)

    harbor_trial_dst = work_dir / "harbor_trial"
    if harbor_trial_dst.exists():
        shutil.rmtree(harbor_trial_dst)
    shutil.copytree(trial_dir, harbor_trial_dst)

    trajectory_path = harbor_trial_dst / "trajectory.json"
    if not trajectory_path.is_file():
        raise RuntimeError(
            f"harbor run at {trial_dir} produced no trajectory.json; "
            "--export-traces must be enabled and Harbor >= trace-supporting version. "
            f"stderr={proc.stderr[:500]}"
        )

    return log_path, {
        "status": "success" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "harbor_result": json.loads(chosen.read_text()),
        "harbor_trial_dir": str(harbor_trial_dst),
    }


def _detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        return sum(1 for line in out.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


DISPATCHERS = {"dry": dispatch_dry, "local": dispatch_local, "harbor": dispatch_harbor}


def _reward_from_harbor(harbor_result: dict) -> dict:
    """Adapt a Harbor trial result to our grader shape.

    Harbor keeps the parsed verifier output under verifier_result.rewards; there is
    no top-level reward key, so reading one silently yields 0.0 for every trial.
    """
    rewards = (harbor_result.get("verifier_result") or {}).get("rewards") or {}
    if not rewards:
        rewards = harbor_result.get("metrics") or {}
    reward = rewards.get("reward", harbor_result.get("reward", 0.0))
    reward = float(reward) if isinstance(reward, (int, float)) else 0.0
    return {
        "reward": reward,
        "hit_target": reward > 0,
        **{k: v for k, v in rewards.items() if k != "reward"},
        "_harbor": harbor_result,
    }


def _is_trial_result(path: Path) -> bool:
    try:
        return "trial_name" in json.loads(path.read_text())
    except (OSError, ValueError):
        return False


DEFAULT_CODEX_BRIDGE_URL = "http://127.0.0.1:8788"
DEFAULT_JUDGE_MODEL = "gpt5.6-sol"


def run(task, seeds: int, backend: str, out_root: Path, *,
        variant: Path | None = None, ledger: Path | None = None,
        llm_config: dict | None = None, agent: str = "claude-code",
        concurrent: int = 1,
        attempt: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        attempt_dir: Path | None = None,
        jobs_dir: Path | None = None,
        verifier_enabled: bool = True,
        codex_bridge_url: str = DEFAULT_CODEX_BRIDGE_URL,
        codex_bridge_key: str | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        precomputed_judgements: Path | None = None,
        model_name: str | None = None,
        retry_count: int = 0) -> list[dict]:
    if seeds < 1:
        raise ValueError("--seeds must be >= 1")
    task_dir = resolve_task(str(task)) if not isinstance(task, Path) else task.resolve()
    task_id = read_task_id(task_dir)
    # Fail before any scaffolding/mounting/dispatch: harbor never uploads the task
    # root into /workspace, so a variant for an image-shipped workspace would be
    # silently dropped and the trial would grade the image's baked-in optimizer.
    if backend == "harbor" and variant is not None and _workspace_ships_in_image(task_dir):
        raise VariantDeliveryImpossible(
            f"task '{task_id}' ({task_dir.name}) ships /workspace inside its docker image, "
            f"so --variant {variant} cannot reach /workspace/submission/optimizer.py: "
            "harbor uploads only the workdir, its mount targets, /tests, /solution and "
            "environment/ into the trial -- never the task root. Running anyway would grade "
            "the optimizer baked into the image and silently ignore the variant. "
            "Drop --variant and have the agent author submission/optimizer.py in-container "
            "instead."
        )
    if ledger is None:
        ledger = resolve_ledger_path(task_id)
    from scaffold_policy import ensure_policy_scaffolded
    ensure_policy_scaffolded(task_dir)
    dispatch = DISPATCHERS[backend]

    out_root.mkdir(parents=True, exist_ok=True)
    if attempt_dir is not None:
        attempt_dir.mkdir(parents=True, exist_ok=True)
    run_name = variant.stem if variant else task_dir.name
    rows = []
    proxy_base_url = llm_config["base_url"] if llm_config else None
    if model_name is None and llm_config is not None:
        model_name = llm_config.get("model")

    verifier_dir = resolve_verifier_dir(task_dir) if verifier_enabled else None
    bia_active = verifier_dir is not None and attempt_dir is not None

    for seed in range(seeds):
        if attempt_dir is not None:
            work = attempt_dir / f"seed_{seed}"
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            work = out_root / f"{task_dir.name}_{run_name}_seed{seed}_{stamp}"
        mounted = work / "task"
        mount_task(task_dir, mounted, variant=variant)

        error = None
        log = None
        harbor_trial_dir = None
        retries_used = 0
        dispatch_synthetic = False
        dispatch_kwargs = {
            "llm_config": llm_config, "agent": agent,
            "attempts": 1, "concurrent": concurrent,
        }
        if backend == "harbor" and jobs_dir is not None:
            dispatch_kwargs["jobs_dir"] = jobs_dir
        for attempt_idx in range(retry_count + 1):
            try:
                if attempt_idx > 0:
                    mount_task(task_dir, mounted, variant=variant)
                log, meta = dispatch(mounted, seed, work, run_name, **dispatch_kwargs)
                harbor_trial_dir = meta.get("harbor_trial_dir")
                dispatch_synthetic = dispatch_synthetic or bool(meta.get("is_synthetic"))
                if backend == "harbor" and "harbor_result" in meta:
                    reward_payload = _reward_from_harbor(meta["harbor_result"])
                else:
                    if log is None:
                        raise RuntimeError(f"{backend} dispatch returned no log path")
                    reward_payload = _grade(mounted, log)
                status = meta["status"]
                error = None
                if status != "error":
                    break
                retries_used = attempt_idx + 1 if attempt_idx < retry_count else attempt_idx
            except Exception as e:
                reward_payload = {"reward": 0.0, "hit_target": False, "reason": str(e)}
                status = "error"
                error = f"{type(e).__name__}: {e}"
                retries_used = attempt_idx + 1 if attempt_idx < retry_count else attempt_idx

        if attempt_dir is not None:
            work.mkdir(parents=True, exist_ok=True)
            reward_json_path = work / "reward.json"
            marked_payload = mark_reward_payload(reward_payload, backend)
            if dispatch_synthetic:
                marked_payload["is_synthetic"] = True
            reward_json_payload = {
                **marked_payload,
                "_status": status,
                "_error": error,
                "_backend": backend,
                "_seed": seed,
                "_variant": variant.name if variant else "(none)",
                "_task_id": task_id,
            }
            reward_json_path.write_text(
                json.dumps(reward_json_payload, indent=2, sort_keys=True, default=str) + "\n"
            )
            if log is not None and log.is_file():
                canonical_log = work / "log"
                if canonical_log.resolve() != log.resolve():
                    try:
                        shutil.copy2(log, canonical_log)
                    except (OSError, shutil.SameFileError):
                        pass

        verifier_dir_str: str | None = None
        consolidated_reward: float | None = None
        verification_complete: bool | None = None
        process_score: float | None = None
        if bia_active and status != "error":
            task_grader_reward = reward_payload.get("reward")
            reward_payload["_task_grader_reward"] = task_grader_reward
            run_id = (f"run{attempt:02d}_seed_{seed}"
                      if attempt is not None else f"seed_{seed}")
            try:
                bia_reward = bia_grade_seed(
                    work, verifier_dir,
                    task_reward=task_grader_reward,
                    run_id=run_id,
                    codex_bridge_url=codex_bridge_url,
                    codex_bridge_key=codex_bridge_key,
                    judge_model=judge_model,
                    precomputed_judgements=precomputed_judgements,
                )
                verifier_dir_str = bia_reward.get("_verifier_dir")
                verification_complete = bool(bia_reward.get("verification_complete"))
                process_score = bia_reward.get("process_score")
                consolidated_reward = bia_reward.get("consolidated_reward")
                if not verification_complete:
                    status = "verification_incomplete"
                    reward_payload["reward"] = None
                    reward_payload["hit_target"] = None
                else:
                    reward_payload["reward"] = consolidated_reward
                    reward_payload["hit_target"] = (
                        consolidated_reward is not None and consolidated_reward > 0
                    )
                reward_payload["_bia"] = {
                    "consolidated_reward": consolidated_reward,
                    "verification_complete": verification_complete,
                    "process_score": process_score,
                    "gate": bia_reward.get("gate"),
                    "task_reward": bia_reward.get("task_reward"),
                    "verifier_dir": verifier_dir_str,
                    "grade_rc": bia_reward.get("_bia_grade_rc"),
                }
            except Exception as bia_err:
                status = "verification_incomplete"
                reward_payload["reward"] = None
                reward_payload["hit_target"] = None
                reward_payload["_bia_error"] = f"{type(bia_err).__name__}: {bia_err}"
                error = error or f"BIA grade failed: {type(bia_err).__name__}: {bia_err}"

        row = normalize(
            variant=variant.name if variant else "(none)",
            attempt=attempt,
            seed=seed, backend=backend,
            task_id=task_id, trial=seed,
            status=status, reward_payload=reward_payload,
            error=error, log_path=str(log) if log else None,
            proxy_base_url=proxy_base_url,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            work_dir=str(work),
            harbor_trial_dir=harbor_trial_dir,
            verifier_dir=verifier_dir_str,
            consolidated_reward=consolidated_reward,
            verification_complete=verification_complete,
            process_score=process_score,
            model_name=model_name,
            retry_count=retries_used,
        )
        append(ledger, row)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    return rows


def parse_args(argv):
    p = argparse.ArgumentParser(description="Outer runner for bia-harness.")
    p.add_argument("--task", required=True,
                   help="Task name, UUID, or path. Resolved under tasks/ or against [task].name/[task].uuid.")
    p.add_argument("--variant", type=Path, default=None,
                   help="Optional candidate variant file (task's mount.toml [variant].dst).")
    p.add_argument("--seeds", type=int, default=2,
                   help="Number of RNG seeds per attempt. Default 2 = Rule 5 (2-seed reproduction).")
    p.add_argument("--backend", choices=list(DISPATCHERS), required=True)
    p.add_argument("--out-root", type=Path, default=HARNESS_ROOT / "runs")
    p.add_argument("--ledger", type=Path, default=None,
                   help="Per-task ledger path. Default: runs/<task-slug>/runs.jsonl (auto-resolved from task_id).")
    p.add_argument("--llm-config", type=Path, default=None,
                   help="Path to proxy JSON (model, base_url, api_key, timeout, num_retries). Only used with --backend harbor.")
    p.add_argument("--agent", default="claude-code",
                   help="Harbor agent name (claude-code/codex/aider/opencode/nop/...). "
                        "Default: claude-code. Use nop for a no-agent verifier-only smoke.")
    p.add_argument("--attempts", "--iterations", dest="attempts", type=int, default=1,
                   help="Number of LLM-authored candidates (RFP 'attempts', PI 'variants'). "
                        ">1 requires --llm-config and delegates to orchestrator.run_loop. "
                        "--iterations is a deprecated alias.")
    p.add_argument("--llm-retries", type=int, default=3,
                   help="LLM API retries on 429/timeout (LLMClient.num_retries). "
                        "Default 3. Overrides llm-config's num_retries field if set.")
    p.add_argument("--retry", dest="retry_count", type=int, default=0,
                   help="Infra-level dispatch retries per seed on non-zero exit. "
                        "Default 0 (single try). SPEC_AUDIT R12.")
    p.add_argument("--baseline-attempt-1", action="store_true",
                   help="SPEC_AUDIT R9: on attempt 1, skip planner and run raw trainer as baseline. "
                        "Only meaningful with --attempts > 1.")
    p.add_argument("--concurrent", type=int, default=1,
                   help="Harbor -n: parallel trials.")
    p.add_argument("--jobs-dir", type=Path, default=None,
                   help="Root directory Harbor writes trial artifacts under (passed as harbor --jobs-dir). "
                        "Default: per-seed work_dir/jobs. Only used with --backend harbor.")
    p.add_argument("--no-verifier", action="store_true",
                   help="Disable BIA verifier grading per seed (default: enabled when "
                        "tasks/<task>/verifier/ exists and --attempts > 1).")
    p.add_argument("--codex-bridge-url", default=DEFAULT_CODEX_BRIDGE_URL,
                   help=f"Codex bridge base URL for rubric judge (default: {DEFAULT_CODEX_BRIDGE_URL}).")
    p.add_argument("--codex-bridge-key", default=None,
                   help="Codex bridge auth secret. Default: env KAIJU_CODEX_BRIDGE_SECRET or OPENAI_API_KEY.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help=f"Rubric judge model name (default: {DEFAULT_JUDGE_MODEL}).")
    p.add_argument("--target-reward", type=float, default=None,
                   help="R11 stop condition: halt loop when any seed's consolidated_reward >= target.")
    p.add_argument("--max-wall-clock-sec", type=float, default=None,
                   help="R11 stop condition: halt loop when cumulative wall-clock >= this many seconds.")
    p.add_argument("--max-input-tokens", type=int, default=None,
                   help="R11 stop condition: halt loop when cumulative planner input tokens >= this budget.")
    p.add_argument("--k-consecutive-incomplete", type=int, default=None,
                   help="R11 stop condition: halt after K consecutive attempts where every seed "
                        "returned status=verification_incomplete.")
    p.add_argument("--final", choices=["none", "last", "best"], default="none",
                   help="After run completes, write runs/<task>/final_selection.json "
                        "picking either the last row or the best-reward row. "
                        "Default 'none' skips selection.")
    return p.parse_args(argv)


def write_final_selection(ledger: Path, policy: str, task_id: str) -> Path | None:
    if policy == "none" or not ledger.is_file():
        return None
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("task_id") == task_id and r.get("status") == "success"]
    if not rows:
        return None
    if policy == "last":
        chosen = rows[-1]
    else:
        chosen = max(rows, key=lambda r: (r.get("reward") or 0.0, -(r.get("step_to_3_28") or 1e18)))
    out = ledger.parent / "final_selection.json"
    payload = {
        "task_id": task_id,
        "final_selection_policy": policy,
        "selected_attempt": chosen.get("attempt"),
        "selected_variant": chosen.get("variant"),
        "selected_seed": chosen.get("seed"),
        "selected_reward": chosen.get("reward"),
        "selected_step_to_3_28": chosen.get("step_to_3_28"),
        "selected_log_path": chosen.get("log_path"),
        "selected_timestamp_utc": chosen.get("timestamp_utc"),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def main(argv):
    args = parse_args(argv[1:])
    llm_config = load_llm_config(args.llm_config) if args.llm_config else None
    if llm_config is not None:
        llm_config["num_retries"] = args.llm_retries
    codex_bridge_key = args.codex_bridge_key or os.environ.get("KAIJU_CODEX_BRIDGE_SECRET") \
        or os.environ.get("OPENAI_API_KEY")
    verifier_enabled = not args.no_verifier
    if args.attempts > 1:
        if llm_config is None:
            print("--attempts > 1 requires --llm-config", file=sys.stderr)
            return 2
        from orchestrator import run_loop
        rows = run_loop(
            args.task, iterations=args.attempts, backend=args.backend,
            llm_config=llm_config, agent=args.agent,
            seeds_per_iter=args.seeds, out_root=args.out_root, ledger=args.ledger,
            concurrent=args.concurrent, jobs_dir=args.jobs_dir,
            verifier_enabled=verifier_enabled,
            codex_bridge_url=args.codex_bridge_url,
            codex_bridge_key=codex_bridge_key,
            judge_model=args.judge_model,
            target_reward=args.target_reward,
            max_wall_clock_sec=args.max_wall_clock_sec,
            max_input_tokens=args.max_input_tokens,
            k_consecutive_incomplete=args.k_consecutive_incomplete,
            baseline_attempt_1=args.baseline_attempt_1,
            retry_count=args.retry_count,
        )
    else:
        rows = run(args.task, args.seeds, args.backend, args.out_root,
                   variant=args.variant, ledger=args.ledger,
                   llm_config=llm_config, agent=args.agent,
                   concurrent=args.concurrent, jobs_dir=args.jobs_dir,
                   verifier_enabled=verifier_enabled,
                   codex_bridge_url=args.codex_bridge_url,
                   codex_bridge_key=codex_bridge_key,
                   judge_model=args.judge_model,
                   retry_count=args.retry_count)
    ok = all(r.get("status") == "success" for r in rows) if rows else True
    if args.final != "none" and rows:
        task_id = rows[0].get("task_id")
        ledger = args.ledger or resolve_ledger_path(task_id)
        out = write_final_selection(ledger, args.final, task_id)
        if out:
            print(f"[final:{args.final}] wrote {out}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
