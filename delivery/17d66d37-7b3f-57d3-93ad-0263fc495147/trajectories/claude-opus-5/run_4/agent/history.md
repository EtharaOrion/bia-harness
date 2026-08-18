# Attempt 4 — your previous attempts at this task

You have attempted this task 3 times. This is attempt 4; this is your last attempt.

`reward`, `graded_step` and `seeds` below are recorded by the verifier and are not
open to dispute. `outcome` is a classification applied afterwards, and the prose under
each iteration is your own summarised account, written at the time of that attempt.

CURRENT budget is 0.5 hours of wall clock. Earlier accounts may quote a different figure or a
rule that has since changed; where they disagree with the task instruction you have
now, the instruction wins.

| iter | reward | graded_step | outcome (classified) | seeds |
|---|---|---|---|---|
| 1 | 1.0000 | 2665 | graded_pass | 2 |
| 2 | 1.0000 | 2663 | graded_pass | 2 |
| 3 | 1.0000 | 2653 | graded_pass | 2 |

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

## Iteration 2 — what you tried

Graded run submitted and verified from the frozen runner's own logs.

| seed | first step ≤ 3.28 | final loss |
|---|---|---|
| 0 | 2642 | 0.000000 |
| 1 | 2663 | 0.000000 |

**graded_step 2663 → score 1.0000** (was 2665 last attempt). Both logs are full-length 4201 lines.

**What I did.** The prior submission was already at the score ceiling, so I reproduced it, confirmed 2665, then measured where its remaining slack actually was:

- Built an offline lab that replays the runner's loop in-process; it reproduces the runner's crossing steps exactly, so I could sweep parameters at ~0.3 s per candidate.
- Computed the true floors. A straight walk from the origin to the bowl's centre bottoms out at 2632/2651 steps, so the old run was carrying ~13 steps of probe-phase overhead.
- Swept the probe tilt: the loss is dominated by the probe *heading*, not conditioning — the step count improves by 2 and then saturates below 0.15. Checked recovery accuracy across the sweep (`max|H⁻¹H − I|` = 4e-9 at 0.15 vs 2.6e-10 at 0.6), so 0.15 is the knee: the two steps back, three orders of margin still in hand.
- Tried steering the probe heading with a min-residual (GMRES-like) estimate from the pairs co

## Iteration 3 — what you tried

Graded run complete and verified from the frozen runner's own logs.

| seed | first step ≤ 3.28 | final loss | log |
|---|---|---|---|
| 0 | 2635 | 0.000000 | 4201 lines |
| 1 | 2653 | 0.000000 | 4201 lines |

**graded_step 2653** (previous best 2663). Score is 1.0000 — that is the ceiling, since anything ≤2900 saturates the formula; 2653 buys margin, not points.

**What changed.** The prior submission's structure was already right (24 secant probes → exact `H⁻¹ = S Y⁻¹` → straight-line Newton pursuit), and its remaining slack was entirely in the probe phase: the 24 probes travelled along a *fixed* heading, `-g` at the origin, which makes an angle of ~58°/64° with the direction to the minimiser on these two problems. That misheading cost ~11 steps of path length.

The fix steers each probe with the information collected so far. At probe `k` the heading is `-M g`, where `M` is an SR1 inverse-curvature estimate built from the `k` pairs already in hand on top of a Barzilai–Borwein-scaled identity: the pairs pin `M yᵢ = sᵢ` on the explored subspace, the identity term carries the unexplored rest as soft. Early probes are barely better than `-g` (nothing else is known); the last ones are

## Now attempt the task again

Your best attempt is iteration 1 at reward 1.0000 (graded_step 2665). Its source is reproduced below.
EXPLOIT it -- start from that submission and improve on it -- or, if you judge the
approach exhausted, replace it with a materially different update rule.
Beat 1.0000. Submit a graded 2-seed run either way.

## Your previous submission (iteration 3)

Validation curve it produced:

```
seed0  1:2641.04415  351:2005.02066  701:1458.65923  1051:999.02382  1401:626.11442  1751:339.93104  2101:140.47368  2451:27.74234  2801:0.00000  3151:0.00000  3501:0.00000  3851:0.00000  4200:0.00000
seed1  1:4113.06391  351:3122.04975  701:2270.98685  1051:1555.07499  1401:974.31417  1751:528.70440  2101:218.24566  2451:42.93797  2801:0.00000  3151:0.00000  3501:0.00000  3851:0.00000  4200:0.00000
```

This is the exact `submission/optimizer.py` you submitted. Modify it,
or replace it, but do not reimplement it blind.

```python
"""A curvature-estimating optimizer for the bia/minicalc trust-region loop.

The harness clips every update to a fixed euclidean norm, so the step budget buys a
fixed amount of PATH LENGTH, not a fixed number of "big" steps. Any rule that follows
the current gradient walks a bent arc: the steepest-descent direction points along
`H(x - x*)`, which on an ill-conditioned bowl is badly misaligned with the direction
that actually matters, `x - x*`. Correcting that misalignment needs curvature, and
curvature is not in a single gradient -- it lives in the RELATION between gradients
already seen. So the shortest path is bought with curvature, and this optimizer
spends its first `dim` steps buying it while still travelling.

  PHASE 1 -- identification, `dim` steps. For a quadratic, gradient differences are
  exact curvature samples: `g(x + s) - g(x) = H s`, with no truncation error at all.
  `dim` steps along linearly independent directions give `dim` secant pairs
  `(s_i, y_i)` with `Y = H S`, and `S` invertible means `H^-1 = S Y^-1` is fully
  determined. This is the batch limit of what BFGS/SR1 approximate incrementally;
  the trust region makes probe steps cheap enough to gather it in one go.

  The probe directions are not the coordinate axes: `dim` orthogonal probes cancel
  to a net displacement of only `sqrt(dim) * h`, throwing away nearly all of the
  path length they cost. Each probe is instead a unit vector tilted slightly off a
  travelling HEADING, so the probe phase both informs and advances.

  The heading is re-estimated every probe step from the pairs collected so far, by
  an SR1 quasi-Newton inverse built on a small multiple of the identity: the pairs
  pin `M y_i = s_i` on the subspace already explored, and the identity term carries
  the rest with a Barzilai-Borwein scale, so `-M g` is the best available guess at
  the direction to the minimiser. Early on it is little better than `-g` (nothing
  else is known yet); by the last probes it is nearly the Newton direction. Steering
  the probes this way recovers most of the path length the old fixed heading wasted:
  measured across ten seeds it crosses ~8 steps earlier, and lands within 2-3 steps
  of the unreachable floor set by a straight line from the origin to the minimiser.

  Only the HEADING is taken from the quasi-Newton model. The pairs themselves are
  still kept for the exact solve, because the SR1 estimate on a partial subspace is
  a guess while `S Y^-1` on the full one is not.

  PHASE 2 -- straight-line pursuit. With `H^-1` in hand the Newton point
  `p = x - H^-1 g` is the model's minimiser, recomputed from the LIVE gradient every
  step rather than frozen once, so arithmetic drift and the clipping of each update
  correct themselves instead of accumulating. `p` is returned as-is and the harness
  clips it: proposing the far target and being cut short is exactly a unit-speed walk
  along the straight line to it, which is the shortest path any optimizer could take
  and hence the fewest steps this loop can be made to spend.

Degeneracy is handled by falling back to normalized steepest descent: if the secant
system is singular, or the Newton direction comes out non-finite or uphill, the
gradient still points somewhere useful and the run keeps moving.

On what is deliberately NOT done: a shorter path exists. The fewest steps to reach
some loss level L is a straight line to the nearest point of the sublevel set
`{f <= L}`, not to the minimiser, and for this bowl that is worth a few hundred
steps. But L is the grader's threshold, not something the loop tells the optimizer,
so an optimizer that aimed there would be tuned to this benchmark rather than
solving the problem it was handed. Aiming at the minimiser is the horizon-free
choice: it is the same trajectory for every L at once.
"""
from __future__ import annotations

import math

# Tilt of each probe off the current heading. Small keeps the probe displacement
# productive; large keeps the secant matrix well conditioned. The secant residuals
# carry no truncation error, so conditioning only has to beat float64 rounding, and
# the trade is lopsided: at 0.1 the recovered inverse still satisfies
# max|H^-1 H - I| ~ 1e-7, six orders of margin from anything that could bend the
# pursuit, while the crossing step stops improving below it.
PROBE_TILT = 0.1

# Scale of the identity seed for the SR1 heading model, in units of the
# Barzilai-Borwein curvature estimate `s.y / y.y` of the latest pair. Large means
# the unexplored subspace is treated as soft (long steps there), which is the right
# prior on an ill-conditioned bowl. The crossing step is flat within one step over
# beta in [8, 32], so this is a plateau, not a tuned point.
SEED_SCALE = 16.0


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


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
        self.prev_x = None        # previous point/gradient, to form the next pair
        self.prev_g = None
        self.ss = []              # secant displacements, phase 1 only
        self.ys = []              # matching gradient differences
        self.hinv = None          # recovered inverse Hessian, rows
        self.u0 = None            # heading of first resort: steepest descent

    # -- heading model --------------------------------------------------------
    def _sr1_inverse(self):
        """SR1 inverse-curvature estimate from the pairs so far, seeded `gam * I`."""
        n = self.dim
        sy = _dot(self.ss[-1], self.ys[-1])
        yy = _dot(self.ys[-1], self.ys[-1])
        gam = SEED_SCALE * (sy / yy) if (yy > 1e-300 and sy > 0.0) else SEED_SCALE
        m = [[gam if i == j else 0.0 for j in range(n)] for i in range(n)]
        for s, y in zip(self.ss, self.ys):
            my = [_dot(m[r], y) for r in range(n)]
            w = [s[r] - my[r] for r in range(n)]
            den = _dot(w, y)
            # Skip the update when the pair adds nothing new: SR1's denominator
            # vanishes exactly then, and dividing by it manufactures noise.
            if abs(den) < 1e-11 * math.sqrt(_dot(w, w) * _dot(y, y) + 1e-300):
                continue
            for r in range(n):
                wr = w[r] / den
                if wr == 0.0:
                    continue
                mr = m[r]
                for c in range(n):
                    mr[c] += wr * w[c]
        return m

    def _heading(self, g):
        if not self.ss:
            return self.u0
        m = self._sr1_inverse()
        z = [-_dot(m[r], g) for r in range(self.dim)]
        zn = math.sqrt(_dot(z, z))
        if not math.isfinite(zn) or zn <= 0.0:
            return self.u0
        return [v / zn for v in z]

    # -- curvature recovery ---------------------------------------------------
    def _identify(self):
        """Build H^-1 = S Y^-1 from the stored secant pairs. None if degenerate."""
        n = self.dim
        # Want M = S Y^-1, i.e. M Y = S. Transposed that is Y^T M^T = S^T, a plain
        # square solve whose j-th solution column is the j-th ROW of M.
        rows = _solve([self.ys[k] for k in range(n)],
                      [[self.ss[k][j] for k in range(n)] for j in range(n)])
        if rows is None:
            return None
        # Symmetrise: H is symmetric, so averaging halves the rounding error.
        hinv = [[0.5 * (rows[i][j] + rows[j][i]) for j in range(n)] for i in range(n)]
        if any(not math.isfinite(v) for row in hinv for v in row):
            return None
        return hinv

    # -- fallback -------------------------------------------------------------
    def _descent(self, x, g):
        gn = math.sqrt(_dot(g, g))
        if gn <= 0.0 or not math.isfinite(gn):
            return x
        return [x[i] - g[i] / gn for i in range(self.dim)]

    # -- the contract ---------------------------------------------------------
    def step(self, x, grad):
        n = self.dim

        if self.hinv is None:
            if self.prev_x is None:
                gn = math.sqrt(_dot(grad, grad))
                self.u0 = ([-v / gn for v in grad] if gn > 0.0 else [0.0] * n)
            else:
                self.ss.append([x[i] - self.prev_x[i] for i in range(n)])
                self.ys.append([grad[i] - self.prev_g[i] for i in range(n)])
            self.prev_x = list(x)
            self.prev_g = list(grad)

            k = len(self.ss)
            if k < n:                                   # still probing
                u = self._heading(grad)
                d = u[:]
                d[k] += PROBE_TILT                      # keep the pairs independent
                dn = math.sqrt(_dot(d, d))
                if dn < 1e-12:                          # u ~ -tilt * e_k; any axis does
                    d = [0.0] * n
                    d[k] = 1.0
                    dn = 1.0
                return [x[i] + d[i] / dn for i in range(n)]

            self.hinv = self._identify()
            self.ss = self.ys = self.prev_x = self.prev_g = None
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
