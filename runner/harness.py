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
import platform
import socket
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

HARNESS_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = HARNESS_ROOT / "tasks"
DEFAULT_LEDGER = HARNESS_ROOT / "runs.jsonl"

sys.path.insert(0, str(HARNESS_ROOT / "runner"))
from mount_variant import mount_task  # noqa: E402
from ingest_result import normalize, append  # noqa: E402


def resolve_task(task_arg: str) -> Path:
    p = Path(task_arg)
    if p.exists() and p.is_dir():
        return p.resolve()
    candidate = TASKS_ROOT / task_arg
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()
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
                return sub.resolve()
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


def load_llm_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    for k in ("model", "base_url", "api_key"):
        if k not in data:
            raise ValueError(f"llm-config {path} missing required field: {k}")
    return data


def _container_base_url(base_url: str) -> str:
    """On macOS Docker Desktop, 172.17.0.1 is not the host loopback; use host.docker.internal."""
    if platform.system() != "Darwin":
        return base_url
    parsed = urlparse(base_url)
    if parsed.hostname == "172.17.0.1":
        netloc = f"host.docker.internal:{parsed.port}" if parsed.port else "host.docker.internal"
        return parsed._replace(netloc=netloc).geturl()
    return base_url


def _host_side_base_url(base_url: str) -> str:
    """Preflight from host must reach the bridge on the host, not the container-visible alias."""
    parsed = urlparse(base_url)
    if platform.system() == "Darwin" and parsed.hostname in ("172.17.0.1", "host.docker.internal"):
        netloc = f"localhost:{parsed.port}" if parsed.port else "localhost"
        return parsed._replace(netloc=netloc).geturl()
    return base_url


def preflight_bridge(base_url: str, timeout_sec: float = 3.0) -> None:
    check_url = _host_side_base_url(base_url)
    parsed = urlparse(check_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            pass
    except OSError as e:
        raise RuntimeError(
            f"claude_code_bridge unreachable at {host}:{port} ({e}). "
            f"Start with: bash proxy/claude_code_bridge.sh start"
        )


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
        f"Running task {mounted_task.name} with dry backend (synthetic).",
    ]
    for step in range(250, 3001, 250):
        val = round(4.0 - 0.00025 * step, 5)
        lines.append(f"step:{step}/3000 val_loss:{val} train_time:{step * 0.35:.3f}s step_avg:{350:.2f}ms")
    log.write_text("\n".join(lines) + "\n")
    return log, {"status": "success", "returncode": 0}


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
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=env_dir, stdout=f, stderr=subprocess.STDOUT, check=False)
    return log, {"status": "success" if proc.returncode == 0 else "error",
                 "returncode": proc.returncode}


def build_harbor_cmd(mounted_task: Path, seed: int, jobs_dir: Path, *,
                     llm_config: dict | None, agent: str, attempts: int,
                     concurrent: int) -> list[str]:
    cmd = [
        "harbor", "run",
        "-p", str(mounted_task),
        "-a", agent,
        "-k", str(attempts),
        "-n", str(concurrent),
        "--env", "docker",
        "--jobs-dir", str(jobs_dir),
    ]
    _ = seed
    if llm_config:
        container_url = _container_base_url(llm_config["base_url"])
        cmd += ["-m", llm_config["model"]]
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


def dispatch_harbor(mounted_task: Path, seed: int, work_dir: Path,
                    run_name: str, *, llm_config: dict | None = None,
                    agent: str = "bash", attempts: int = 1,
                    concurrent: int = 1) -> tuple[Path | None, dict]:
    jobs_dir = work_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    if llm_config:
        preflight_bridge(llm_config["base_url"])
        if agent == "bash":
            print("WARN: --llm-config set but --agent=bash (bash agent will not call the LLM). "
                  "Use --agent claude_code (or another litellm-based Harbor agent).",
                  file=sys.stderr)

    cmd = build_harbor_cmd(mounted_task, seed, jobs_dir,
                           llm_config=llm_config, agent=agent,
                           attempts=attempts, concurrent=concurrent)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    result_files = list(jobs_dir.rglob("result.json"))
    if not result_files:
        raise RuntimeError(f"harbor run produced no result.json; stderr={proc.stderr[:500]}")

    trial_dir = result_files[-1].parent
    artifact_logs = list((trial_dir / "artifacts" / "runs").glob("*.log")) \
        if (trial_dir / "artifacts" / "runs").exists() else []
    log_candidates = artifact_logs or list(trial_dir.rglob("*.log"))
    log_path = log_candidates[-1] if log_candidates else None

    return log_path, {
        "status": "success" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "harbor_result": json.loads(result_files[-1].read_text()),
    }


def _detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        return sum(1 for line in out.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


DISPATCHERS = {"dry": dispatch_dry, "local": dispatch_local, "harbor": dispatch_harbor}


def _reward_from_harbor(harbor_result: dict) -> dict:
    """Harbor's result.json already contains reward+metrics; adapt to our grader-shape."""
    reward = harbor_result.get("reward", 0.0)
    metrics = harbor_result.get("metrics") or {}
    return {
        "reward": reward,
        "hit_target": reward > 0,
        **{k: v for k, v in metrics.items()},
    }


def run(task, seeds: int, backend: str, out_root: Path, *,
        variant: Path | None = None, ledger: Path | None = None,
        llm_config: dict | None = None, agent: str = "bash",
        attempts: int = 1, concurrent: int = 1) -> list[dict]:
    if seeds < 1:
        raise ValueError("--seeds must be >= 1")
    task_dir = resolve_task(str(task)) if not isinstance(task, Path) else task.resolve()
    task_id = read_task_id(task_dir)
    if ledger is None:
        ledger = resolve_ledger_path(task_id)
    from scaffold_policy import ensure_policy_scaffolded
    ensure_policy_scaffolded(task_dir)
    dispatch = DISPATCHERS[backend]

    out_root.mkdir(parents=True, exist_ok=True)
    run_name = variant.stem if variant else task_dir.name
    rows = []
    proxy_base_url = llm_config["base_url"] if llm_config else None

    for seed in range(seeds):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        work = out_root / f"{task_dir.name}_{run_name}_seed{seed}_{stamp}"
        mounted = work / "task"
        mount_task(task_dir, mounted, variant=variant)

        error = None
        log = None
        try:
            log, meta = dispatch(mounted, seed, work, run_name,
                                 llm_config=llm_config, agent=agent,
                                 attempts=attempts, concurrent=concurrent)
            if backend == "harbor" and "harbor_result" in meta:
                reward_payload = _reward_from_harbor(meta["harbor_result"])
            else:
                if log is None:
                    raise RuntimeError(f"{backend} dispatch returned no log path")
                reward_payload = _grade(mounted, log)
            status = meta["status"]
        except Exception as e:
            reward_payload = {"reward": 0.0, "hit_target": False, "reason": str(e)}
            status = "error"
            error = f"{type(e).__name__}: {e}"

        row = normalize(
            variant=variant.name if variant else "(none)",
            seed=seed, backend=backend,
            task_id=task_id, trial=seed,
            status=status, reward_payload=reward_payload,
            error=error, log_path=str(log) if log else None,
            proxy_base_url=proxy_base_url,
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
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--backend", choices=list(DISPATCHERS), required=True)
    p.add_argument("--out-root", type=Path, default=HARNESS_ROOT / "runs")
    p.add_argument("--ledger", type=Path, default=None,
                   help="Per-task ledger path. Default: runs/<task-slug>/runs.jsonl (auto-resolved from task_id).")
    p.add_argument("--llm-config", type=Path, default=None,
                   help="Path to proxy JSON (model, base_url, api_key, timeout, num_retries). Only used with --backend harbor.")
    p.add_argument("--agent", default="bash",
                   help="Harbor agent name (bash/claude_code/codex/aider/opencode/...). Default: bash (deterministic).")
    p.add_argument("--attempts", type=int, default=1,
                   help="Harbor -k: attempts per trial.")
    p.add_argument("--concurrent", type=int, default=1,
                   help="Harbor -n: parallel attempts per trial.")
    p.add_argument("--iterations", type=int, default=1,
                   help="Number of LLM-driven refinement iterations. >1 requires --llm-config; delegates to orchestrator.run_loop.")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv[1:])
    llm_config = load_llm_config(args.llm_config) if args.llm_config else None
    if args.iterations > 1:
        if llm_config is None:
            print("--iterations > 1 requires --llm-config", file=sys.stderr)
            return 2
        from orchestrator import run_loop
        rows = run_loop(
            args.task, iterations=args.iterations, backend=args.backend,
            llm_config=llm_config, agent=args.agent,
            seeds_per_iter=args.seeds, out_root=args.out_root, ledger=args.ledger,
        )
    else:
        rows = run(args.task, args.seeds, args.backend, args.out_root,
                   variant=args.variant, ledger=args.ledger,
                   llm_config=llm_config, agent=args.agent,
                   attempts=args.attempts, concurrent=args.concurrent)
    ok = all(r.get("status") == "success" for r in rows) if rows else True
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
