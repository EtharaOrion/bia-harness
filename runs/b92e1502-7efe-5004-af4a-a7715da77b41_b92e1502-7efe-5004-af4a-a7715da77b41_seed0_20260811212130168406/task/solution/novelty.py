"""Private novelty scoring over a dense in-run capture. The measurement context IS the run.

An earlier bundle scored novelty with a standalone probe that drove the submitted optimizer
over synthetic tensors in a context disjoint from training. Attack A3 defeated it by
detecting that context and branching. The reply was to measure inside the run over a keyed
window, which was the right architectural call and is kept. What was wrong was resting the
window on a secret the runner handed to the submission, and resting the statistic on raw
update magnitude from a cold-started replay. This module fixes the second half; the runner
fixes the first.

Two defects are corrected here, and they had one root cause between them.

SCALE. The first element of the old feature vector was the raw Frobenius norm of the
update, so the statistic measured step size before it measured update rule. A four percent
learning-rate bump on the published baseline scored as novel, and at five percent it
out-scored the reference optimizer. instruction.md promises that changing only a
hyperparameter does not clear this gate, so that promise was false. Every feature is now
computed after dividing by ONE global scalar, the root-sum-square of every captured delta
in the run. That removes an overall learning-rate rescale exactly, because a uniform
rescale divides out, while preserving everything that encodes a rule: the magnitude ratio
BETWEEN tensors, the magnitude profile ACROSS steps, and the internal shape of each update.
Nayebi et al. (arXiv 2010.11765) report the same thing from the other direction, that
trajectory-averaged scale is insufficient to separate learning rules; Bernstein and
Newhouse frame the choice of norm as the rule identity and the step size as a separate
scalar. Absolute scale is still reported, as scale_ratio_log, and is deliberately excluded
from the score: a term that let step size manufacture novelty is the defect, not the fix.

STATE. The old comparison instantiated each corpus member with COLD optimizer state at the
window start and compared it against a submission carrying hundreds of steps of warm
momentum. With a decay of 0.95 the member never caught up, so the statistic measured state
warmth as much as update rule. That is why a verbatim copy scored between 0.039 and 0.095
against ITSELF, why the floor had to be raised to 0.09 to catch copies, and why that same
floor then rejected the bundle's own reference solution in a fifth of window placements.
The copy ceiling exceeded the reference floor and the margin was negative. Replay now
starts from the first captured step, so the member warms up on the same real gradient
sequence the submission saw, and its parameters are teacher-forced to the submission's
recorded parameters at each step so both answer the same question from the same footing.
This is a counterfactual one-step replay with warm state, in the sense of arXiv 2608.04408,
rather than an off-the-shelf named technique. A verbatim copy now reproduces the deltas it
is compared against and scores essentially zero.

Residual, stated rather than hidden: a submission may still spend real optimization quality
to look different, for instance by perturbing its updates. That trade is the point. It buys
divergence with training progress it must then make up before step 2900. Keeping this
statistic private, rather than shipping it in the agent-visible runner, is what stops that
perturbation from being cheaply optimized against the metric itself.
"""
from __future__ import annotations

import math

import torch

# Grading must not depend on how many cores the verifier host happens to have. Multithreaded
# BLAS reassociates reductions, which moved this statistic in the fourth decimal between a
# 64-core and a single-threaded run of the same bytes. A score that changes with host width
# is a VERIFIER-ROBUSTNESS defect, so thread count is pinned here rather than inherited.
torch.set_num_threads(1)

# Calibrated on the rebuilt gate. solution/separation_test.py measures the copy ceiling and
# the reference floor across the full independent-seed sweep and writes the table this
# number is read off. The floor is placed inside the measured gap, not chosen to make a
# fixture resolve, and separation_test.py fails if the gap it sits in is not positive.
#
# These numbers come from smoke-scale captures. They are a measured basis, not a full-scale
# calibration, and must be re-derived from full-scale captures before any pilot. See
# gap-novelty-floor-smoke-scale-only.
NOVELTY_FLOOR = 0.07
QUANT_DECIMALS = 6
SCORE_VERSION = "inrun/v2"


def _quantize(x: float) -> float:
    if not math.isfinite(x):
        return float("nan")
    return round(float(x), QUANT_DECIMALS)


def global_scale(all_deltas) -> float:
    """Root-sum-square of every captured delta in the run.

    One scalar for the whole capture, never one per step or per tensor. Dividing by a
    per-tensor norm would also erase the magnitude ratio between tensors, and dividing by a
    per-step norm would erase the profile across steps. Both of those are rule content. An
    overall learning rate is not, and it is the only thing this removes.
    """
    tot = 0.0
    for d in all_deltas:
        tot += float(d.detach().to(torch.float64).pow(2).sum())
    return math.sqrt(tot)


def delta_features(delta: torch.Tensor, scale: float, n_deltas: int) -> list:
    """Reduce one parameter delta to scale-free features.

    The leading feature is the delta's share of the run's total update energy, not its raw
    norm. Multiplying by the square root of the delta count keeps it order one so the
    quantization step means the same thing at every capture length.
    """
    d = delta.detach().to(torch.float64)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    fro = float(d.norm())
    if fro == 0.0 or scale == 0.0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    rel = fro / scale * math.sqrt(n_deltas)
    dn = d / fro
    return [
        _quantize(rel),
        _quantize(float(dn.norm(dim=1).std()) if dn.shape[0] > 1 else 0.0),
        _quantize(float(dn.norm(dim=0).std()) if dn.shape[1] > 1 else 0.0),
        _quantize(float(dn.abs().mean())),
        _quantize(float(dn.max() - dn.min())),
    ]


def vector_from_deltas(per_step_deltas, cumulative_deltas):
    """Returns the feature vector and the absolute scale it was normalized by."""
    flat = [d for step in per_step_deltas for d in step] + list(cumulative_deltas)
    scale = global_scale(flat)
    n = max(1, len(flat))
    v = []
    for step_deltas in per_step_deltas:
        for d in step_deltas:
            v.extend(delta_features(d, scale, n))
    for d in cumulative_deltas:
        v.extend(delta_features(d, scale, n))
    return v, scale


def divergence(vec_a, vec_b) -> float:
    """Normalized behavioral distance on the closed interval from 0 to 1."""
    if len(vec_a) != len(vec_b):
        return 1.0
    num = 0.0
    den = 0.0
    for a, b in zip(vec_a, vec_b):
        if math.isnan(a) or math.isnan(b):
            if math.isnan(a) and math.isnan(b):
                continue
            return 1.0
        num += abs(a - b)
        den += max(abs(a), abs(b))
    if den == 0.0:
        return 0.0
    return min(1.0, num / den)


def scale_ratio_log(scale_a: float, scale_b: float) -> float:
    """Reported, never scored.

    How much larger the submission's total update energy was than the member's. This is the
    quantity the old statistic scored first. It is retained as evidence for a reader and
    kept out of the score, because a gate that lets step size manufacture novelty
    contradicts what instruction.md promises.
    """
    if scale_a <= 0.0 or scale_b <= 0.0:
        return float("nan")
    return round(math.log(scale_a / scale_b), 6)


def submission_deltas(capture):
    """What the submission actually did to the real parameters, step by step.

    The capture holds the real parameter state entering the run and the real state after
    every captured step, so these deltas are the graded run's own updates. No optimizer is
    executed to produce them.
    """
    before = capture["params_before"]
    after = capture["params_after"]
    per_step = []
    prev = before
    for step_after in after:
        per_step.append([a - p for a, p in zip(step_after, prev)])
        prev = step_after
    cumulative = [a - b for a, b in zip(after[-1], before)]
    return per_step, cumulative


def submission_vector(capture):
    per_step, cumulative = submission_deltas(capture)
    return vector_from_deltas(per_step, cumulative)


def corpus_vector(build_optimizer, capture, lr: float):
    """What a corpus member would have done, from the same footing, equally warm.

    The member starts where the submission started and is driven by the run's own recorded
    gradient sequence from the first captured step, so its internal state builds up exactly
    as the submission's did. Before each step its parameters are teacher-forced to the
    submission's recorded parameters, so the two are always answering the same question
    from the same point rather than drifting apart into different regions of the landscape.
    Its cumulative delta is the sum of its one-step counterfactual deltas, which is the
    honest cumulative for a teacher-forced replay.
    """
    before = capture["params_before"]
    params = [torch.nn.Parameter(t.clone(), requires_grad=False) for t in before]
    names = list(capture["param_names"])
    opt = build_optimizer(params, names, lr)

    per_step = []
    running = [torch.zeros_like(t) for t in before]
    prev_state = before
    for si, grads in enumerate(capture["grads"]):
        with torch.no_grad():
            for p, t in zip(params, prev_state):
                p.data.copy_(t)
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()
        for p in params:
            p.grad = None
        step_delta = [p.data - t for p, t in zip(params, prev_state)]
        per_step.append(step_delta)
        for r, d in zip(running, step_delta):
            r += d
        prev_state = capture["params_after"][si]
    return vector_from_deltas(per_step, running)
