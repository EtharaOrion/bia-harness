# track3_optimization_harness

Minimal, self-contained slice of `modded-nanogpt` required to run the
**Track-3 (Optimization) benchmark**. Everything outside this folder in the
parent repo is out of scope for Track-3 (main-speedrun trainer, Triton kernels,
HellaSwag evaluator, Track-1 records, etc. are not imported by any Track-3
code path).

## Contents

| Path | Role |
|---|---|
| `train_gpt_simple.py` | The Track-3 baseline trainer (result #36 hyperparameters). Single-file, self-contained. Copied verbatim from `records/track_3_optimization/train_gpt_simple.py`. |
| `data/cached_fineweb10B.py` | One-shot FineWeb-10B token downloader (Hugging Face). Writes shards to `data/fineweb10B/fineweb_{train,val}_*.bin` next to itself. |
| `requirements.txt` | Runtime deps per the Track-3 quickstart: `torch==2.11`, `huggingface_hub`. |
| `TRACK3_README.md` | The official Track-3 benchmark doc: rules, quickstart, results history, active techniques, statistical-significance formula. |

## Quickstart (single GPU or 1..8× A100/H100)

Run from **this folder** (both the data glob and the log path are CWD-relative
inside the trainer). Do not `cd` into `data/` or into a subdirectory.

```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py 20         # ~2B tokens; enough for ~4000 steps
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) train_gpt_simple.py
```

Baseline `train_steps = 3250` is hard-coded in `train_gpt_simple.py`. To sweep
seeds, pass a trailing integer: `torchrun ... train_gpt_simple.py 5` runs 5
trials sequentially in the same process.

Per-run output:

- Console: `step:{S}/{N} val_loss:{V:.5f} train_time:{T:.3f}s step_avg:{ms:.2f}ms`
- Log file: `logs/{uuid4()}.txt` (CWD-relative; the same content as console)

There is no `result.json`, no checkpoint, and no structured artifact. The log
file is the artifact.

## Reproducing a specific record (#4..#46)

Track-3 rule #4 requires every submission to be self-contained: submission
trainers live under `records/track_3_optimization/results/<slug>/` in the
parent repo and are *not* the same as `train_gpt_simple.py`. To reproduce a
specific record, copy that submission's `train_gpt_*.py` + `run.sh` +
`claim_stats.py` into this folder (or run them out of the parent repo) and
follow their local README. This harness folder covers the **baseline** and
gives you the data pipeline every submission depends on.

## Grader contract

The de-facto grader is `claim_stats.py` (stdlib-only: `math, re, sys`), shipped
per-submission in the parent repo. It reads plain log files and parses two
regexes:

- `step:(\d+)/\d+ val_loss:([\d.]+)`
- `readout step:(\d+) alpha:([\d.]+) val_loss:([\d.]+)` (tail-EMA sweeps only)

Statistical significance rule (Track-3 §Rules #3):

```
(3.28 - avg_loss) * sqrt(num_runs) >= 0.004
```

## Hidden invariants (read before running)

1. **CWD must be this folder.** `train_gpt_simple.py` uses `Path.cwd().glob(...)`
   for `data/fineweb10B/fineweb_{train,val}_*.bin` and writes logs to
   `logs/{uuid}.txt`. Running from a different CWD fails with `StopIteration`
   or a missing `logs/` write.

2. **Torch version matters.** Track-3 quickstart pins `torch==2.11`. The parent
   repo's root `requirements.txt` pins `torch==2.10`, which has a documented
   NaN bug on A100 with `torch.compile` (see Track-3 README). This folder's
   `requirements.txt` uses `torch==2.11` on purpose.

3. **Data shard count.** `python data/cached_fineweb10B.py 20` downloads 20
   train chunks (~2B tokens), enough for the baseline's 3250 steps. Submissions
   that run longer need more chunks (up to `100` for full FineWeb-10B).

4. **World-size assertion.** The trainer asserts `8 % world_size == 0`, i.e.
   `nproc_per_node` must be one of {1, 2, 4, 8}. Any other value aborts at setup.

5. **No CLI for hyperparameters.** `train_steps`, learning rates, weight decays,
   and betas are all edited in-source (§ `Init & Optim Hyperparams`, around
   line 269). This is by design per Track-3 §Guidelines ("hardcoded
   hyperparameters are to be preferred as compared to command line arguments").

6. **`sys.argv[-1]` = num_trials.** Line 257: `num_trials = int(sys.argv[-1]) if
   len(sys.argv) > 1 else 1`. Any positional arg to the script is coerced to
   int; there is no argparse. Under `torchrun`, put the count *after* the
   script path.

7. **No provenance beyond self-source.** Line 10 (`with open(sys.argv[0]) as f:
   code = f.read()`) snapshots the trainer's own source into the log header.
   That is the only artifact linking the log to the code that produced it.

## What this folder does NOT contain (and why)

- `train_gpt.py`, `train_gpt_medium.py`, `triton_kernels.py`,
  `dc_triton_kernels.py` — main speedrun (Track-1). Not imported by Track-3.
- `evals/hellaswag.py` — Track-1 auxiliary evaluator. `grep -rn hellaswag
  records/track_3_optimization/` returns zero matches.
- Root `run.sh`, root `Dockerfile` — target the main speedrun, wrong torch pin.
- `records/track_3_optimization/make_figures.py` — matplotlib visualization
  over finished logs; not on the run path.
- `records/track_3_optimization/results/<slug>/**` — historical logs and
  per-submission trainers. Copy only what you need to reproduce a specific
  record.
- `records/track_3_optimization/results/20260513_shampoo_1_4_power/distributed_shampoo/`
  — vendored 3rd-party repo. Its test tree is the source of the 41 pytest
  collection errors at the parent-repo root. Not required for any run.
