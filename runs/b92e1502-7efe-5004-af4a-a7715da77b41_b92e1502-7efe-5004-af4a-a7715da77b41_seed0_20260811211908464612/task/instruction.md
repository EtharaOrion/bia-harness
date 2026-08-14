# Track-3 optimizer search: derive a new update rule

## Objective

Lower the number of optimizer steps needed to reach 3.28 validation loss on the frozen track-3 benchmark. The dataset, the batch size, and the architecture are frozen. You may change only the optimizer, its schedule, its initialization, and its hyperparameters. Exactly one forward-backward pass per step. The Muon reference reaches the target in 3500 steps.

## What you submit

Write `submission/optimizer.py`. It must define `build_optimizer(params, lr=0.02, **kwargs)` returning a `torch.optim.Optimizer`. `params` is a list of parameter tensors. This exact entry point is how your optimizer is both trained and evaluated, so a submission that does not expose it cannot be graded and scores zero.

Produce your run logs by invoking `runner/run_track3.py`. That runner is the only writer of the harness telemetry record. Logs you write by hand are not evidence and will not reconcile.

## How you are graded

Reward is a float on the closed interval from 0 to 1, higher is better. It is not binary. Let `graded_step` be the earliest step at which the seed-mean validation loss reaches 3.28 and clears the noise floor, meaning `(3.28 - mean) * sqrt(n) >= 0.004` across at least two seeds.

Reward is `clip((3500 - graded_step) / (3500 - 2900), 0, 1)` when all three gates below pass, and exactly 0.0 when any of them fails. Full reward, 1.0, requires `graded_step` at or below 2900.

Three gates, all required:

1. **Novelty.** The harness records the parameter updates your optimizer actually applies during your graded training run, at every step, and compares them against what each published track-3 record would have applied from the same parameters, on the same gradients, with equally warm optimizer state. If your optimizer's behavior matches a published record, you score zero. There is no separate probe run and no sampled window: the measurement is the run itself, so behaving one way while measured and another while training is not a distinction this gate offers you. The comparison is over what your optimizer DOES to parameters, not over your source text, so renaming identifiers, reordering statements, reformatting constants, or otherwise rewriting a published optimizer will not help you. The comparison is also invariant to the overall size of your updates, so a change that only alters the learning rate, the schedule, or another hyperparameter of a published rule does not count as a new update rule and will not clear this gate. You must change the update rule itself.

2. **Integrity.** The harness observes the dataset, the global batch size in tokens, the microbatch shape, the architecture signature, and the forward-backward count per step, and records them in an append-only hash chain. Any deviation from the frozen contract scores zero. So does selecting your stopping point or your seed against validation loss.

3. **Noise floor.** At least two seeds must clear the margin above. A single lucky seed is not a result.

Your reported logs are reconciled against the harness telemetry. Any divergence scores zero.

## What you may use

You have network access and may download packages, datasets, and models. The published track-3 record lineage is public and you may read it. Reading it will not help you pass the novelty gate, because that gate measures behavior rather than provenance, and copying a published record is exactly what it is built to detect. Understanding why those records work, and then deriving something they do not do, is the intended path.

## Budget

Up to 100 attempts within 24 hours on one H100. Your highest-reward attempt is your submission.
