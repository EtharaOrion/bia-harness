#!/usr/bin/env python3
"""Frozen trainer for bia/minicalc. The solver never edits this file.

This is the miniature of `runner/run_track3.py` in the bia/track3nov bundle: it owns
the objective, it owns the frozen contract, and it is the ONLY writer of the graded
telemetry. A solver that hand-writes a log is writing fiction; the graded artifact is
whatever this file emitted.

THE OBJECTIVE (frozen, stdlib only, no numpy/torch)

    f(x) = 0.5 * (x - c)^T Q^T diag(lam) Q (x - c)

A rotated, ill-conditioned quadratic bowl in DIM dimensions. `Q` is an orthonormal
basis, `lam` is log-spaced over COND, and `c` is a shift of fixed norm RADIUS. All
three are drawn from `random.Random(RNG_BASE + seed)`, so a seed determines the
problem exactly and two different seeds are two different (equally hard) problems.
The solver starts at the origin and is never told where `c` is.

The rotation is the point. A diagonal (axis-aligned) bowl is fixed by any
per-coordinate step-size rule; a rotated one is not, so beating the gradient
direction requires estimating curvature, which is the actual skill under test.

THE FROZEN CONTRACT

  ONE GRADIENT PER STEP. The solver's `step(x, grad)` sees exactly one gradient per
  update, and cannot evaluate `f` itself. That is what makes a line search
  impossible and makes a step-size rule a real decision rather than a search.

  MAX_UPDATE_NORM. Every proposed update is clipped to a euclidean norm of
  MAX_UPDATE_NORM before it is applied. This is a trust region, and it is the
  reason the task is cheap: it decouples "how many steps this takes" from "how
  expensive a step is", so a 24-dimensional bowl still needs thousands of steps and
  still runs in seconds. It also decides what a good optimizer is here -- the step
  budget buys a fixed amount of PATH LENGTH, so the winner is whoever walks the
  shortest path to the bowl's floor, not whoever takes the biggest steps.

  NO EARLY STOP. Every seed runs the full MAX_STEPS regardless of what the loss
  does, so the curve cannot be truncated at a lucky point.

Losses are formatted fixed-point on purpose: the grader and
`runner/agentloop/trial_io.py` both parse `step:(\\d+)/\\d+ val_loss:([\\d.]+)`, which
scientific notation and a minus sign would both break. `f` is a sum of squares so it
is never negative, and it is clamped to DIVERGED_LOSS on the way out.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import time

# ---------------------------------------------------------------- frozen problem
DIM = 24
COND = 2000.0
RADIUS = 6.0
MAX_UPDATE_NORM = 0.0022
MAX_STEPS = 4200
SEEDS = (0, 1)
RNG_BASE = 9781

# A loss this large is reported instead of inf/nan so the line still parses. The seed
# is abandoned at that point; a diverged run has nothing further to measure.
DIVERGED_LOSS = 1.0e9

DEFAULT_SUBMISSION = "/workspace/submission"


# ------------------------------------------------------------------- the problem
def make_problem(seed: int):
    """`(rows, lam, c)` for `seed`. Deterministic: Mersenne Twister, stdlib only.

    `rows` are the orthonormal rows of Q, built by modified Gram-Schmidt on a
    gaussian matrix. `lam` is log-spaced over `COND` and then shuffled, so the
    stiff directions are not correlated with the row order.
    """
    rng = random.Random(RNG_BASE + seed)
    rows: list[list[float]] = []
    for _ in range(DIM):
        v = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
        for r in rows:
            d = sum(a * b for a, b in zip(v, r))
            v = [a - d * b for a, b in zip(v, r)]
        n = math.sqrt(sum(a * a for a in v))
        rows.append([a / n for a in v])
    lam = [math.exp(math.log(COND) * i / (DIM - 1)) for i in range(DIM)]
    rng.shuffle(lam)
    d = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
    n = math.sqrt(sum(a * a for a in d))
    c = [RADIUS * a / n for a in d]
    return rows, lam, c


def hessian(rows, lam):
    """Q^T diag(lam) Q, materialised once so a step costs ONE matvec, not two."""
    return [[sum(lam[k] * rows[k][i] * rows[k][j] for k in range(DIM))
             for j in range(DIM)] for i in range(DIM)]


def value_and_grad(x, h, c):
    y = [x[i] - c[i] for i in range(DIM)]
    g = [sum(hi[j] * y[j] for j in range(DIM)) for hi in h]
    return 0.5 * sum(y[i] * g[i] for i in range(DIM)), g


# -------------------------------------------------------------- the submission
class SubmissionError(RuntimeError):
    """The submitted optimizer does not honour the documented contract."""


def load_build_optimizer(path: str):
    if not os.path.isfile(path):
        raise SubmissionError(f"no optimizer at {path}")
    spec = importlib.util.spec_from_file_location("submission_optimizer", path)
    if spec is None or spec.loader is None:
        raise SubmissionError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "build_optimizer", None)
    if not callable(fn):
        raise SubmissionError(f"{path} defines no callable build_optimizer(dim)")
    return fn


def _checked(new_x, x):
    """The optimizer's return value, or a SubmissionError naming what was wrong.

    Checked every step rather than once: an optimizer that starts returning the
    wrong shape after its warmup branch fires would otherwise poison the curve
    silently, and a silently wrong curve is worse than a crash.
    """
    try:
        out = [float(v) for v in new_x]
    except (TypeError, ValueError) as exc:
        raise SubmissionError(
            f"step() must return a sequence of {DIM} floats: {exc}") from exc
    if len(out) != DIM:
        raise SubmissionError(f"step() returned {len(out)} values, expected {DIM}")
    if any(not math.isfinite(v) for v in out):
        raise SubmissionError("step() returned a non-finite value")
    d = [out[i] - x[i] for i in range(DIM)]
    n = math.sqrt(sum(v * v for v in d))
    if n > MAX_UPDATE_NORM:
        # Clipped, not rejected: the trust region is a property of the harness, and
        # an optimizer is entitled to propose a longer step and be cut short.
        s = MAX_UPDATE_NORM / n
        out = [x[i] + d[i] * s for i in range(DIM)]
    return out


# ------------------------------------------------------------------- the run
def run_seed(seed: int, build_optimizer, log_path: str, max_steps: int,
             target: float | None, deadline: float | None) -> None:
    rows, lam, c = make_problem(seed)
    h = hessian(rows, lam)
    opt = build_optimizer(DIM)
    if not hasattr(opt, "step"):
        raise SubmissionError("build_optimizer(dim) returned an object with no .step()")

    x = [0.0] * DIM
    first_cross = None
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as fh:
        fh.write(f"# bia/minicalc seed={seed} dim={DIM} cond={COND:g} "
                 f"radius={RADIUS:g} max_update_norm={MAX_UPDATE_NORM:g} "
                 f"steps={max_steps}\n")
        # s counts COMPLETED optimizer steps, so `step:s` is the loss after exactly s
        # updates and step:1 is never the untouched initial point.
        for s in range(0, max_steps + 1):
            f, g = value_and_grad(x, h, c)
            if not math.isfinite(f):
                f = DIVERGED_LOSS
            if s >= 1:
                fh.write(f"step:{s}/{max_steps} val_loss:{min(f, DIVERGED_LOSS):.6f}\n")
                if target is not None and first_cross is None and f <= target:
                    first_cross = s
            if f >= DIVERGED_LOSS:
                fh.write(f"# seed {seed} diverged at step {s}; abandoning this seed\n")
                break
            if s == max_steps:
                break
            if deadline is not None and time.monotonic() > deadline:
                raise SubmissionError(
                    f"wall-clock budget exhausted at seed {seed} step {s}; the "
                    f"submitted optimizer is too slow per step")
            x = _checked(opt.step(list(x), list(g)), x)
        fh.flush()
        os.fsync(fh.fileno())

    tail = "" if target is None else (
        f"  first step at or below {target:g}: "
        f"{first_cross if first_cross is not None else 'NEVER'}")
    print(f"seed {seed}: wrote {log_path}  final_loss={f:.6f}{tail}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="frozen bia/minicalc trainer")
    p.add_argument("--submission", default=DEFAULT_SUBMISSION,
                   help="dir holding optimizer.py; logs land in <dir>/logs")
    p.add_argument("--optimizer", default=None,
                   help="explicit path to optimizer.py (default <submission>/optimizer.py)")
    p.add_argument("--logs", default=None,
                   help="explicit log dir (default <submission>/logs)")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                   help="comma-separated seeds; a GRADED run needs 0,1")
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--target", type=float, default=None,
                   help="report the first step at or below this loss (display only; "
                        "tests/grade.py owns the graded threshold)")
    p.add_argument("--max-seconds", type=float, default=600.0,
                   help="abort if the whole run exceeds this many seconds")
    a = p.parse_args(argv)

    opt_path = a.optimizer or os.path.join(a.submission, "optimizer.py")
    log_dir = a.logs or os.path.join(a.submission, "logs")
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    build = load_build_optimizer(opt_path)
    t0 = time.monotonic()
    deadline = t0 + a.max_seconds if a.max_seconds > 0 else None
    for seed in seeds:
        run_seed(seed, build, os.path.join(log_dir, f"full_seed{seed}.log"),
                 a.steps, a.target, deadline)
    print(f"done in {time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SubmissionError as exc:
        print(f"SUBMISSION ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
