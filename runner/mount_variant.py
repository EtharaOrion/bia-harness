#!/usr/bin/env python3
"""Mount a Harbor task template into a working directory.

Config lookup order (first hit wins):
  1. <task_dir>/mount.toml               (in-tree; task authored by us)
  2. <harness_root>/mount-configs/<task_dir.name>.toml  (out-of-tree; digest-pinned tasks like bia)
  3. <task_dir>/task.toml [artifacts]    (inferred; tasks that declare their own submission path)
  4. no config (task copied verbatim, no injection)

Level 3 exists because Harbor tasks already declare where the graded artifact
lives, e.g. bia/track3nov's task.toml:

    artifacts = [..., "/workspace/submission/optimizer.py", ...]

and its instruction.md tells the agent "Write `submission/optimizer.py`". Making
the harness read that directly means such a task needs no hand-written
mount-config to accept a --variant, and the variant can never drift from the
path the task actually grades.
"""
import shutil
import sys
import tomllib
from pathlib import Path


DEFAULT_HARNESS_ROOT = Path(__file__).resolve().parent.parent
MOUNT_TOML = "mount.toml"
MOUNT_CONFIGS_DIR = "mount-configs"
TASK_TOML = "task.toml"
# The mounted task tree IS the container workspace, so this prefix is stripped
# off artifact paths to yield a dst relative to the mount root.
WORKSPACE_PREFIX = "/workspace/"


def _resolve_mount_config_path(task_dir: Path, harness_root: Path) -> Path | None:
    in_tree = task_dir / MOUNT_TOML
    if in_tree.is_file():
        return in_tree
    fallback = harness_root / MOUNT_CONFIGS_DIR / f"{task_dir.name}.toml"
    if fallback.is_file():
        return fallback
    return None


def _variant_dst_from_task_toml(task_dir: Path) -> str | None:
    """Returns a dst relative to the mount root, or None if nothing qualifies."""
    task_toml = task_dir / TASK_TOML
    if not task_toml.is_file():
        return None
    try:
        with task_toml.open("rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError):
        return None

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        return None

    candidates = [
        a[len(WORKSPACE_PREFIX):]
        for a in artifacts
        if isinstance(a, str) and a.startswith(WORKSPACE_PREFIX) and a.endswith(".py")
    ]
    if not candidates:
        return None
    for c in candidates:
        if "submission/" in c:
            return c
    return candidates[0]


def _load_mount_config(task_dir: Path, harness_root: Path) -> tuple[dict, Path | None]:
    cfg_path = _resolve_mount_config_path(task_dir, harness_root)
    if cfg_path is not None:
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
        return {
            "shared": data.get("shared", []),
            "variant": data.get("variant"),
        }, cfg_path

    inferred = _variant_dst_from_task_toml(task_dir)
    if inferred is not None:
        return {
            "shared": [],
            "variant": {"dst": inferred, "required": False, "inferred": True},
        }, task_dir / TASK_TOML

    return {"shared": [], "variant": None}, None


def mount_task(task_dir: Path, out_dir: Path, *,
               variant: Path | None = None,
               harness_root: Path = DEFAULT_HARNESS_ROOT) -> Path:
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task template not found: {task_dir}")
    if variant is not None and not variant.is_file():
        raise FileNotFoundError(f"variant not found: {variant}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(task_dir, out_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "policy"))

    cfg, cfg_path = _load_mount_config(task_dir, harness_root)

    for entry in cfg["shared"]:
        src = harness_root / entry["src"]
        dst = out_dir / entry["dst"]
        if not src.is_file():
            raise FileNotFoundError(
                f"shared asset missing: {src} (declared in {cfg_path})"
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    if variant is not None:
        variant_cfg = cfg["variant"]
        if variant_cfg is None:
            raise ValueError(
                f"task {task_dir.name} does not accept a --variant "
                f"(no [variant] block in {cfg_path or 'mount.toml'})"
            )
        dst = out_dir / variant_cfg["dst"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(variant, dst)
    elif cfg["variant"] is not None and cfg["variant"].get("required"):
        raise ValueError(
            f"task {task_dir.name} requires a --variant but none was provided"
        )

    return out_dir


def main(argv):
    if len(argv) not in (3, 4):
        print("usage: mount_variant.py <task_dir> <out_dir> [<variant.py>]", file=sys.stderr)
        return 2
    task_dir = Path(argv[1])
    out_dir = Path(argv[2])
    variant = Path(argv[3]) if len(argv) == 4 else None
    result = mount_task(task_dir, out_dir, variant=variant)
    print(str(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
