# Track 3 optimizer search

The authoritative statement of the objective is `instruction.md`. This file exists so the image carries a pointer rather than a second, drifting copy of the task; where the two ever disagree, `instruction.md` wins.

`train_gpt_track3.py` is the frozen trainer. Dataset, batch size, sequence length, architecture, and the one forward-backward per step are fixed by it and are observed directly by the harness rather than taken from any claim you make. Your work is the optimizer and nothing else.

Write `submission/optimizer.py`. Run your own experiments however you like. The run that counts is executed by the harness after your session ends.
