"""Dense in-run capture of what the submitted optimizer actually did.

This module is agent-visible on purpose, and unlike its predecessor it holds no secret and
makes no secrecy claim. It records EVERY optimizer step of the graded run, over EVERY
two-dimensional parameter, and digests the result so the telemetry chain can commit to it.

Why dense rather than a keyed window. The predecessor selected a window of consecutive
steps under a key the submission was assumed not to hold. That assumption was false as the
runner was written: the submission was exec'd inside the runner process, so it could read
the key out of the environment and recompute the window exactly. The response is not to
protect the key harder. A measurement whose soundness rests on a secret fails the moment
the secret leaks, and in an agentic setting the adversary moves last. Capturing every step
removes the branch target instead of hiding it. There is no window to compute, so there is
nothing to behave differently inside. A submission that wants to look novel to this capture
has to actually be novel for the whole run, which is the property the gate exists to
measure.

Cost, stated rather than hidden. Dense capture over every step and every 2D parameter is
cheap at smoke scale and expensive at full scale. See gap-capture-fullscale-footprint in
EDICT.md for the measured per-step figure and the streaming design that bounds it. The
scope is declared here rather than quietly narrowed, because narrowing it back to a subset
would reintroduce a branch target.

It deliberately contains no scoring. The statistic that turns a capture into a novelty
number lives in the private verifier. A visible metric would let a submission craft a
minimal-cost perturbation that maximizes the measured statistic while barely changing its
real update, which buys divergence without buying different behaviour.
"""
from __future__ import annotations

import hashlib
import json

import torch

CAPTURE_VERSION = "inrun/v2"
CAPTURE_SCOPE = "all_2d_parameters_every_step"
MIN_STEPS = 8


def capturable_indices(params) -> list:
    """Every two-dimensional parameter, in declaration order.

    Public and unkeyed by construction. A keyed or narrowed subset would hand the
    submission a second thing to branch on: behave novel on the watched tensors, copy a
    record on the rest. Taking all of them leaves no unwatched tensor to hide in.
    """
    return [i for i, p in enumerate(params) if p.dim() == 2]


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Exact tensor bytes without numpy.

    The verifier image ships CPU torch and no numpy, so reinterpreting the buffer through
    a uint8 view keeps the digest exact and dependency-free.
    """
    flat = t.detach().contiguous().to(torch.float64).flatten()
    return bytes(flat.view(torch.uint8).tolist())


def capture_digest(capture) -> str:
    """Digest over the captured tensors, committed into the keyed telemetry chain.

    Binding this into the chain is what stops the captured behaviour from being rewritten
    after the run. The chain is produced by the supervisor process, which holds the key and
    contains no submitted code, so the commitment is made outside the submission's reach.
    """
    h = hashlib.sha256()
    h.update(json.dumps({
        "version": CAPTURE_VERSION,
        "scope": CAPTURE_SCOPE,
        "names": list(capture["param_names"]),
        "steps": len(capture["params_after"]),
        "lr": capture["lr"],
    }, sort_keys=True, separators=(",", ":")).encode())
    for t in capture["params_before"]:
        h.update(_tensor_bytes(t))
    for step in capture["grads"]:
        for t in step:
            h.update(_tensor_bytes(t))
    for step in capture["params_after"]:
        for t in step:
            h.update(_tensor_bytes(t))
    return h.hexdigest()


def spec_digest() -> str:
    return hashlib.sha256(json.dumps({
        "version": CAPTURE_VERSION,
        "scope": CAPTURE_SCOPE,
        "min_steps": MIN_STEPS,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
