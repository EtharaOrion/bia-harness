# Attempt 2 — your previous attempts at this task

You have attempted this task once. This is attempt 2; 1 further attempt will follow.

`reward`, `graded_step` and `seeds` below are recorded by the verifier and are not
open to dispute. `outcome` is a classification applied afterwards, and the prose under
each iteration is your own summarised account, written at the time of that attempt.

CURRENT budget is 0.5 hours of wall clock. Earlier accounts may quote a different figure or a
rule that has since changed; where they disagree with the task instruction you have
now, the instruction wins.

| iter | reward | graded_step | outcome (classified) | seeds |
|---|---|---|---|---|
| 1 | 1.0000 | 2665 | graded_pass | 2 |

## Iteration 1 — what you tried

**Done — graded_step 2665, score 1.00** (worst seed; ≤2900 is full marks).

| seed | first step ≤ 3.28 | final loss |
|---|---|---|
| 0 | 2644 | 0.000000 |
| 1 | 2665 | 0.000000 |

Both logs are full-length (4201 lines), written by the frozen runner.

**Why the baselines all stall at 0.30:** the update cap means the step budget buys a fixed *path length*, not a fixed step count. The steepest-descent direction is `H(x−x*)`, which on a bowl with condition number 2000 points well away from `x−x*`. Every gradient-following rule therefore walks the same long curved arc, and no step-size dressing shortens it.

**What the submission does instead** (`submission/optimizer.py`):

1. **Identify (24 steps).** For a quadratic, gradient differences are exact curvature samples — `g(x+s) − g(x) = H s`, no truncation error. Twenty-four probes along linearly independent directions give secant pairs with `Y = H S`, so `H⁻¹ = S Y⁻¹` is fully determined. Recovered to `max|H⁻¹H − I| ≈ 2.6e-10`.
2. **Pursue (the rest).** Return the Newton point `x − H⁻¹g` and let the harness clip it. Proposing a far-away target and being cut short *is* a unit-speed walk down the straight line to it — the shortest path av

## Now attempt the task again

Your best attempt is iteration 1 at reward 1.0000 (graded_step 2665). Its source is reproduced below.
EXPLOIT it -- start from that submission and improve on it -- or, if you judge the
approach exhausted, replace it with a materially different update rule.
Beat 1.0000. Submit a graded 2-seed run either way.

## Your previous submission (iteration 1)

Validation curve it produced:

```
seed0  1:2641.57126  351:1988.84999  701:1448.92448  1051:994.32736  1401:625.05864  1751:341.11833  2101:142.50641  2451:29.22290  2801:0.00000  3151:0.00000  3501:0.00000  3851:0.00000  4200:0.00000
seed1  1:4114.22562  351:3077.97424  701:2242.85422  1051:1539.62612  1401:968.28995  1751:528.84572  2101:221.29340  2451:45.63302  2801:0.00000  3151:0.00000  3501:0.00000  3851:0.00000  4200:0.00000
```

This is the exact `submission/optimizer.py` you submitted. Modify it,
or replace it, but do not reimplement it blind.

```python
"""A curvature-estimating optimizer for the bia/minicalc trust-region loop.

The harness clips every update to a fixed euclidean norm, so the step budget buys a
fixed amount of PATH LENGTH. Any rule that follows the current gradient walks a bent
arc: the steepest-descent direction points along `H(x - x*)`, which for an
ill-conditioned bowl is badly misaligned with the direction that actually matters,
`x - x*`. Correcting that misalignment needs curvature, and curvature is not in a
single gradient -- it is in the RELATION between gradients already seen.

So this optimizer spends its first `dim` steps probing, then walks a straight line.

  PHASE 1 -- identification. For a quadratic, gradient differences are exact
  curvature samples: `g(x + s) - g(x) = H s`, with no truncation error at all. Taking
  `dim` steps along linearly independent directions therefore yields `dim` secant
  pairs `(s_i, y_i)` with `Y = H S`, and `S` invertible means `H = Y S^-1` is
  determined. Nothing about the objective is assumed beyond local quadraticity --
  the same construction is the limit that BFGS and SR1 approximate incrementally,
  just gathered in one batch because the trust region makes probe steps cheap.

  The probe directions are not the coordinate axes. `dim` orthogonal probes cancel
  out to a net displacement of only `sqrt(dim) * h`, which throws away most of the
  path length they cost. Instead each probe is a unit vector tilted off a common
  heading `u`: `d_i ~ u + eps * e_i`. Those stay linearly independent (the matrix is
  `u 1^T + eps I`, conditioned like `dim / eps`), while their sum stays close to
  `dim * u`, so the probing phase also travels. `u` is the steepest-descent direction
  at the start -- the only heading available before any curvature is known, and one
  that is at least positively correlated with the way out.

  PHASE 2 -- straight-line pursuit. With `H^-1 = S Y^-1` in hand the Newton point
  `p = x - H^-1 g` is the model's minimiser, recomputed from the live gradient at
  every step rather than frozen once, so arithmetic drift and the clipping of each
  update correct themselves instead of accumulating. `p` is returned as-is and the
  harness clips it: proposing the far-away target and being cut short is exactly a
  unit-speed walk along the straight line to it, which is the shortest path any
  optimizer could take and hence the fewest steps this loop can be made to spend.

Degeneracy is handled by falling back to normalized steepest descent: if the secant
system is singular, or the Newton direction comes out non-finite or uphill, the
gradient still points somewhere useful and the run keeps moving.
"""
from __future__ import annotations

import math

# Tilt of each probe off the common heading. Small keeps the probes productive;
# large keeps the secant matrix well conditioned. 0.6 is comfortably inside both.
PROBE_TILT = 0.6


def _solve(a, b):
    """Solve `a x = b` for square `a` by Gaussian elimination with partial pivoting.

    `b` is a list of right-hand-side columns; returns the solution columns, or None
    if `a` is numerically singular.
    """
    n = len(a)
    m = [row[:] + [col[i] for col in b] for i, row in enumerate(a)]
    w = n + len(b)
    for k in range(n):
        piv = max(range(k, n), key=lambda r: abs(m[r][k]))
        if abs(m[piv][k]) < 1e-300:
            return None
        if piv != k:
            m[k], m[piv] = m[piv], m[k]
        d = m[k][k]
        mk = m[k]
        for r in range(k + 1, n):
            f = m[r][k] / d
            if f:
                row = m[r]
                for j in range(k, w):
                    row[j] -= f * mk[j]
    for k in range(n - 1, -1, -1):
        mk = m[k]
        d = mk[k]
        for j in range(n, w):
            s = mk[j]
            for t in range(k + 1, n):
                s -= mk[t] * m[t][j]
            mk[j] = s / d
    return [[m[i][n + j] for i in range(n)] for j in range(len(b))]


class CurvatureOptimizer:
    def __init__(self, dim: int):
        self.dim = dim
        self.dirs = None          # probe headings, built once the first gradient lands
        self.xs = []              # visited points, phase 1 only
        self.gs = []              # gradients at those points
        self.hinv = None          # recovered inverse Hessian, rows

    # -- probe headings -------------------------------------------------------
    def _make_dirs(self, g):
        n = self.dim
        gn = math.sqrt(sum(v * v for v in g))
        u = [-v / gn for v in g] if gn > 0.0 else [0.0] * n
        dirs = []
        for i in range(n):
            d = u[:]
            d[i] += PROBE_TILT
            dn = math.sqrt(sum(v * v for v in d))
            if dn < 1e-12:                      # u ~ -eps e_i; any independent axis does
                d = [0.0] * n
                d[i] = 1.0
                dn = 1.0
            dirs.append([v / dn for v in d])
        return dirs

    # -- curvature recovery ---------------------------------------------------
    def _identify(self):
        """Build H^-1 = S Y^-1 from the stored secant pairs. None if degenerate."""
        n = self.dim
        s_cols, y_cols = [], []
        for k in range(n):
            s_cols.append([self.xs[k + 1][i] - self.xs[k][i] for i in range(n)])
            y_cols.append([self.gs[k + 1][i] - self.gs[k][i] for i in range(n)])
        # Want M = S Y^-1, i.e. M Y = S. Transposed that is Y^T M^T = S^T, a plain
        # square solve whose j-th solution column is the j-th ROW of M.
        rows = _solve(y_cols, [[s_cols[k][j] for k in range(n)] for j in range(n)])
        if rows is None:
            return None
        hinv = rows
        # Symmetrise: H is symmetric, so averaging halves the rounding error.
        hinv = [[0.5 * (hinv[i][j] + hinv[j][i]) for j in range(n)] for i in range(n)]
        if any(not math.isfinite(v) for row in hinv for v in row):
            return None
        return hinv

    # -- fallback -------------------------------------------------------------
    def _descent(self, x, g):
        gn = math.sqrt(sum(v * v for v in g))
        if gn <= 0.0 or not math.isfinite(gn):
            return x
        return [x[i] - g[i] / gn for i in range(self.dim)]

    # -- the contract ---------------------------------------------------------
    def step(self, x, grad):
        n = self.dim

        if self.hinv is None:
            if self.dirs is None:
                self.dirs = self._make_dirs(grad)
            self.xs.append(list(x))
            self.gs.append(list(grad))
            k = len(self.xs) - 1
            if k < n:                            # still probing
                d = self.dirs[k]
                return [x[i] + d[i] for i in range(n)]
            self.hinv = self._identify()
            self.xs = self.gs = None
            if self.hinv is None:
                return self._descent(x, grad)

        # Newton point of the recovered model, from the live gradient.
        d = [-sum(row[j] * grad[j] for j in range(n)) for row in self.hinv]
        if any(not math.isfinite(v) for v in d):
            return self._descent(x, grad)
        if sum(d[i] * grad[i] for i in range(n)) >= 0.0:   # not a descent direction
            return self._descent(x, grad)
        return [x[i] + d[i] for i in range(n)]


def build_optimizer(dim: int):
    """Return a fresh optimizer object for a `dim`-dimensional problem."""
    return CurvatureOptimizer(dim)
```
