# nanogpt/track3-speedrun

## What you must do

Modify the training script at `/app/environment/train_gpt_simple.py` so that it reaches **val_loss ≤ 3.28** in as few steps as possible when launched via:

```bash
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) /app/environment/train_gpt_simple.py
```

The verifier will parse the stdout log for `step:S/N val_loss:V` lines and grade based on the earliest step at which `val_loss ≤ 3.28`.

## Constraints (hard — verifier will not check these but violations invalidate a submission)

1. Dataset, batch size, and architecture are fixed. Do not touch the `Model` section or the `distributed_data_generator` function.
2. One forward-backward pass per step. No multi-step inner loops.
3. Val loss must reach `≤ 3.28` with statistical significance across seeds: `(3.28 - mean_val_loss) * sqrt(N) ≥ 0.004`. The grader emits `stat_sig_at_n1` per single-seed run; multi-seed aggregation is the harness's job.
4. All modifications must be in the `Optimization` and `Init & Optim Hyperparams` sections. Hyperparameters must be hardcoded in the script — no CLI args.
5. Third-party optimizer libraries must be copied in whole. Do not add `pip install` calls that were not present.

## Success criteria

- Trainer runs to completion (no CUDA errors, no NaN).
- Log contains at least one `step:S/N val_loss:V` line with `V ≤ 3.28`.
- Reward is `clip((3500 - step_to_3.28) / (3500 - 2500), 0, 1)`. Reference: 3500 steps ≈ canonical Muon baseline (reward 0); 2500 steps = ambitious stretch (reward 1).

## Data

FineWeb-10B GPT-2-tokenized shards are pre-downloaded to `/app/environment/data/fineweb10B/`. The trainer reads them via `Path.cwd().glob("data/fineweb10B/fineweb_{train,val}_*.bin")` — so the working directory must be `/app/environment/` when `torchrun` is invoked.

## Verifier

The verifier is `tests/test.sh`, which invokes `tests/grader.py <log>` and writes `/logs/verifier/reward.json` with:

```json
{
  "reward": 0.78,
  "hit_target": true,
  "step_to_3_28": 2720,
  "final_val_loss": 3.27978,
  "min_val_loss": 3.27978,
  "mean_val_loss": 3.34567,
  "n_val_points": 26,
  "final_step": 2720,
  "total_steps_planned": 2900,
  "train_time_s": 872.4,
  "stat_sig_at_n1": 0.00022,
  "target_val_loss": 3.28,
  "baseline_steps": 3500
}
```

## Oracle

If you invoke `harbor run ... -a oracle`, the reference solution at `solution/solve.sh` runs the unmodified canonical baseline (Muon + AdamW at result #3 hyperparameters). Use this to sanity-check the container and calibrate wallclock.

## Reference

Full benchmark spec: https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization (also mirrored at `/app/environment/track3_README.md` if present).
