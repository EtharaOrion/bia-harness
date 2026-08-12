#!/usr/bin/env python3
"""Env-smoke grader. Reward 1.0 iff python + torch lines both present."""
import json
import re
import sys
from pathlib import Path


PY_RE = re.compile(r"^Python\s+3\.(\d+)")
TORCH_RE = re.compile(r"torch:([\d.]+)")


def grade(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    py_ok = False
    py_minor = None
    torch_ok = False
    torch_version = None

    for line in text.splitlines():
        m = PY_RE.match(line)
        if m:
            py_minor = int(m.group(1))
            py_ok = py_minor >= 12
        m = TORCH_RE.search(line)
        if m:
            torch_ok = True
            torch_version = m.group(1)

    reward = 1.0 if (py_ok and torch_ok) else 0.0
    return {
        "reward": reward,
        "hit_target": reward == 1.0,
        "python_minor": py_minor,
        "torch_version": torch_version,
        "checks": {"python>=3.12": py_ok, "torch_importable": torch_ok},
    }


def main(argv):
    if len(argv) == 2:
        print(json.dumps(grade(Path(argv[1])), indent=2))
        return 0
    if len(argv) == 4 and argv[1] == "--write":
        result = grade(Path(argv[2]))
        Path(argv[3]).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
