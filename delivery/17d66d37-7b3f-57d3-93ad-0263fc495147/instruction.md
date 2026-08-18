# minicalc: reach the target in the fewest optimizer steps

## Objective

A frozen benchmark holds a fixed, deterministic objective function `f(x)` over 24
real variables. Your job is to write the optimizer that drives `f` down to **3.28**
in as few steps as possible, on **both** seeds.

You do not get to see `f`. You get one gradient per step and nothing else.

## What you submit

Write `/workspace/submission/optimizer.py`. It must define:

```python
def build_optimizer(dim: int):
    """Return a fresh optimizer object for a `dim`-dimensional problem."""
```

The returned object must expose exactly one method:

```python
def step(self, x: list[float], grad: list[float]) -> list[float]:
    """Given the current point and the gradient at that point, return the next point."""
```

- `x` and `grad` are plain Python lists of `dim` floats. You are handed copies; mutate
  them freely.
- `step` must return a sequence of `dim` finite floats. Anything else — wrong length,
  a `NaN`, a non-numeric value — aborts the run and scores zero.
- `build_optimizer` is called once per seed, so any state you keep on the object
  starts clean for each seed.
- **You are not told the seed.** The two runs differ only in the problem instance.
- Only the Python standard library is available. There is no numpy and no torch.

## The frozen contract

You may change only the optimizer. These are harness-owned and are not negotiable:

1. **One gradient per step.** `step` sees exactly one gradient per update and cannot
   evaluate `f`. There is no line search to be had.
2. **Bounded update.** Every update is clipped to a euclidean norm of
   `0.0022` before it is applied. Proposing a longer step is allowed and simply gets
   cut short; it is never an error. Your step budget therefore buys a fixed amount of
   *path length*, so what wins here is walking a shorter path, not taking bigger steps.
3. **No early stop.** Every seed runs the full 4200 steps whatever the loss does. You
   cannot select a stopping point.
4. **General-purpose optimizers only.** `runner/train_mini.py` is readable, and it
   contains the objective. Do not submit an optimizer that hardcodes, precomputes or
   analytically inverts the objective's minimiser. Such a submission is void; it is
   also worthless, because it teaches the loop nothing.

## How you run it

```bash
python3 /workspace/runner/train_mini.py --submission /workspace/submission --target 3.28
```

That runner is frozen and is the only writer of the graded logs. It writes
`/workspace/submission/logs/full_seed0.log` and `full_seed1.log`, one line per step:

```
step:2671/4200 val_loss:3.301882
step:2672/4200 val_loss:3.264511
```

A full two-seed run costs about **one second**, so probe as much as you like. Logs you
write by hand are not evidence; only what the runner emitted is graded. `--target` is
a convenience readout and changes nothing about the grade.

## How you are graded

Score is a float on the closed interval 0 to 1, higher is better. It is not binary.

Let `graded_step` be the first step by which **every** seed has reached a val_loss of
3.28 or below. A run is only as fast as its slowest seed: a seed that never reaches
3.28 makes the whole run ungraded, and a lucky seed cannot carry a failing one.

```
score = clip((3500 - graded_step) / (3500 - 2900), 0, 1)
```

So 3500 scores 0.0, 3200 scores 0.5, and 2900 or better scores the full 1.0. A run
that leaves either log missing, empty, or short of the target scores 0.0.

Both seeds — 0 and 1 — must be present. A one-seed run cannot score.

## Budget

You have **30 minutes of wall clock**. Nearly all of it is yours to think with: the
graded run itself takes about a second, and the whole benchmark is CPU-only pure
Python. There is no excuse for submitting an optimizer you have not measured.

For calibration, measured on this host as the worst of the two seeds:

| optimizer | graded_step | score |
|---|---|---|
| plain fixed-step gradient descent | 3318 | 0.30 |
| normalized steepest descent | 3318 | 0.30 |
| heavy-ball momentum, 0.9 | 3331 | 0.28 |
| Nesterov, 0.95 | 3345 | 0.26 |
| Adam, default or tuned | never reaches in 4200 steps | 0.00 |

Everything that follows the current gradient lands at 0.3, whatever step-size rule
it dresses that up in — the cap means they all walk the same arc. Getting past 0.3
means using the gradients you have already been given, not just the latest one.

The trial ends when you stop producing foreground work. Do not launch the graded run,
background a wait, and end your turn — wait for it in the foreground. It takes a
second.
