# bia/minicalc — what is measured, what is asserted, and what is not

`bia/minicalc` is a CPU-only miniature of `bia/track3nov`. It exists for exactly one
purpose: to run `runner/agentloop` end to end against a real LLM agent in minutes
instead of the ~2.2 GPU-hours (~$9) that one graded track3nov attempt costs. It is a
harness test that happens to be a real optimization problem, not a research task.

Everything below was measured on this host, not estimated.

## The problem

`f(x) = 0.5 (x - c)^T Q^T diag(lam) Q (x - c)` over 24 variables. `Q` is an
orthonormal basis, `lam` is log-spaced over a condition number of 2000, and `c` has
norm 6. All three come from `random.Random(9781 + seed)`, so a seed fixes the problem
exactly. The solver starts at the origin, is never told `c`, and is never told which
seed it is running on.

Two frozen rules do the work:

- **One gradient per step.** `step(x, grad) -> x_new`. The solver cannot evaluate `f`,
  so there is no line search, and a step-size rule is a real decision.
- **`MAX_UPDATE_NORM = 0.0022`.** Every update is clipped to that euclidean norm. This
  is the reason a 24-dimensional bowl needs thousands of steps and still runs in under
  a second: it decouples step COUNT from step COST. It also defines what "better"
  means here — the budget buys a fixed amount of path length, and the shortest path to
  the floor runs along `-H^-1 g`, not along `-g`.

The rotation is load-bearing. On an axis-aligned bowl any per-coordinate step-size
rule recovers the curvature for free; on a rotated one it does not, which is why Adam
scores 0.00 below and L-BFGS scores 1.00.

## Measured headroom

Worst of seeds 0 and 1, produced by `environment/runner/train_mini.py` and scored by
`tests/grade.py`. The whole table takes 3.8 s to reproduce.

| optimizer | graded_step | reward |
|---|---|---|
| reference (`solution/optimizer.py`, L-BFGS m=8) | 2672 | **1.0000** |
| plain fixed-step GD, lr 1e-3 or 1e-2 | 3318 | 0.3033 |
| normalized steepest descent | 3318 | 0.3033 |
| heavy-ball momentum 0.9 | 3331 | 0.2817 |
| Nesterov 0.95 | 3345 | 0.2583 |
| Adam, default lr 1e-3 | never clears in 4200 steps | 0.0000 |
| Adam, tuned lr 0.032 | never clears in 4200 steps | 0.0000 |

So the ladder a campaign can actually climb is **0.30 -> 1.00**, with 0.00 available
below it for a genuinely wrong choice. A full two-seed graded run costs 0.7 s on the
host and 0.9 s in the container.

The learning rate barely matters and that is deliberate: the cap means every
gradient-following rule walks the same arc and lands at 0.30. The only way past it is
to use the gradient HISTORY, which is the insight the loop should be observed
discovering.

## Why the reward constants are copied from bia/track3nov rather than chosen

`BASELINE_STEPS = 3500`, `TARGET_STEPS = 2900`, `TARGET_LOSS = 3.28` are the same
numbers as the real task. That is not laziness, it is forced:

`runner/agentloop/trial_io.py:87` calls `reward_from_curve(curve)` with **no**
task-derived constants, so the loop scores every task with `reward.py`'s module
defaults. There is no code path by which a task bundle's own numbers reach the
loop's scorer — `constants_from_grade_py` exists to DETECT drift, and the module
docstring is explicit that detection is not rebinding.

A miniature whose curve lived on any other scale would therefore be scored against
3.28/2900/3500 anyway, cross it within a handful of steps, and return reward 1.0 on
every iteration — validating nothing. So the objective is calibrated to put its
crossing of 3.28 inside the 2900–3500 window instead. `MAX_UPDATE_NORM` is the knob
that does it: step counts scale as 1/`MAX_UPDATE_NORM`.

The consequence worth stating plainly: this task validates the loop's scoring path
**because** it agrees with the hardcoded constants, not because the loop learned to
read the bundle. That limitation is real and is unchanged by this bundle.

## grade.py agrees with reward.py exactly, on purpose

The track3nov grader adds a significance margin and a persistence check that
`reward.py` deliberately does not reproduce, so those two disagree by construction.
Here they must agree bit for bit, because agreement is the property under test: two
independent implementations of one rule, on one curve. Verified for the reference
(2672 / 1.0000), the naive baseline (3318 / 0.3033), a late crossing (3601 / 0.0000)
and a run that never crosses (None / 0.0000).

## Reason strings are a contract with classify.py

`runner/agentloop/classify.py` maps `reason` to an outcome by substring match, and an
unrecognised string becomes `unknown`, for which no guidance block fires — the next
iteration is told nothing about why it scored zero. Every reason this grader emits
carries a token from `RECOGNISED_REASON_TOKENS`:

| situation | reason | outcome |
|---|---|---|
| reward > 0 | `graded_step=<n>` | `graded_pass` |
| cleared, but at or after 3500 | `no_step_clears_baseline_reached_target_at_step_<n>` | `graded_miss` |
| a seed never cleared | `no_step_clears_target_loss` | `graded_miss` |
| fewer than 2 seeds | `need_at_least_2_seeds_got_<n>` | `agent_abandoned_run` |
| no parsable logs | `telemetry_absent` | `agent_abandoned_run` |

**The reason must not name the failing seed.** `classify` tests `'seed'` in its FIRST
branch, the ungradable one, so `no_step_clears_target_loss_seed_0` classifies as
`harness_incomplete` — the harness gets blamed for an optimizer that was merely too
slow, and `harness_incomplete` is the label under which the next iteration is told to
change nothing. This was hit during development and is why the failing seed is
reported in `detail` instead. This coupling had never been exercised by any bundle.

## Known defects this bundle surfaced but does not fix

1. **The loop cannot read a task's constants** (above). Fixing it means editing
   `runner/agentloop/`, which was out of scope here.
2. **A one-seed run is over-credited by the loop.** `classify` returns `graded_pass`
   on its first branch whenever `reward > 0`, before `n_seeds` is ever consulted, so a
   trial with one log scores off that single seed while `grade.py` correctly calls it
   ungradable (`need_at_least_2_seeds_got_1`). The trainer defaults to both seeds and
   the instruction requires both, so this only bites an agent that overrides `--seeds`.
3. **The objective is readable by the agent.** `train_mini.py` ships in the container,
   so a solver could in principle reconstruct `c` from `seed` and jump to the optimum.
   `bia/track3nov` is immune only because its objective is a language model on real
   data; no cheap CPU objective can be. instruction.md forbids it as a contract term,
   and nothing enforces it. If a campaign returns 1.0 on iteration 1, read the
   submitted `optimizer.py` before believing the ladder.

## Why the bundle declares schema_version = "1.4"

Harbor 0.21's `TaskConfig` field is `schema_version`, defaulting to `"1.4"`
(`harbor/models/task/config.py:796`), and a back-compat validator maps a legacy
top-level `version` onto it (same file, line 826). This bundle declares
`schema_version = "1.4"`, matching both bia/track3nov bundles and harbor itself.

Declaring `version = "1.0"` instead would tell harbor this bundle is schema 1.0 while
it uses 1.4 features (`artifacts`, phase-scoped `network_mode`).

An older test, `legacy/harness2/tests/test_task_toml.py::test_task_toml_parses`,
asserted a top-level `version = "1.0"` and so failed for every bundle using the current
spelling. It was removed rather than satisfied; `test_task_name_is_declared` keeps its
one still-valid assertion. See README.md for current suite counts.
