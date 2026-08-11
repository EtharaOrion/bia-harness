# nanogpt/env-smoke

Environment sanity check. Verify:

1. Python 3.12+ is available.
2. `torch` is importable (CPU wheel is fine).
3. `nvidia-smi` runs without error (skipped if no GPU present).

Reward: 1.0 if all checks pass, 0.0 otherwise. This task does not accept a
`--variant` and does not consume shared assets.

Use this task to smoke-test the runner + Harbor + Docker plumbing without
paying for GPU time or waiting for the full nanogpt trainer.
