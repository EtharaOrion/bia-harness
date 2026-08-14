"""PLANTED DEFECT FIXTURE: a semantics-preserving rewrite of the baseline Muon.

This is the adversarial case the novelty gate exists to catch, and it is the case a
source-text comparison provably fails. Every identifier is renamed, the helper is inlined
differently, the loop is restructured, constants are written in a different literal form,
and the class layout is changed. The arithmetic is unchanged, so a correct behavioral gate
must place this at essentially zero divergence from baseline_muon and block it.

If this fixture ever passes the gate, the gate is inert and the bundle is BLOCK:INVALID_TASK.
"""
from __future__ import annotations

import torch

_COEFF_LINEAR = 2.0
_COEFF_QUAD = -3.0 / 2.0
_COEFF_CUBIC = 1.0 / 2.0
_ITERATIONS = 12
_EPSILON = 1e-7


def _orthogonalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    assert matrix.ndim >= 2
    tall = matrix.size(-2) > matrix.size(-1)
    work = matrix.bfloat16()
    if tall:
        work = work.mT
    denom = work.norm(dim=(-2, -1), keepdim=True) + _EPSILON
    work = work / denom
    iteration = 0
    while iteration < _ITERATIONS:
        gram = work @ work.mT
        shaped = _COEFF_QUAD * gram + _COEFF_CUBIC * (gram @ gram)
        work = _COEFF_LINEAR * work + shaped @ work
        iteration += 1
    if tall:
        work = work.mT
    return work


def _compute_update(gradient, velocity, decay=0.95):
    velocity.lerp_(gradient, 1.0 - decay)
    blended = gradient.lerp_(velocity, decay)
    directed = _orthogonalize_matrix(blended)
    aspect = max(1, gradient.size(-2) / gradient.size(-1)) ** 0.5
    directed *= aspect
    return directed


class OrthogonalMomentumSolver(torch.optim.Optimizer):
    def __init__(self, parameter_list, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(parameter_list, list) and len(parameter_list) >= 1
        ordered = sorted(parameter_list, key=lambda t: t.size(), reverse=True)
        super().__init__(ordered, dict(lr=lr, weight_decay=weight_decay, mu=mu))

    @torch.no_grad()
    def step(self):
        for bundle in self.param_groups:
            step_size = bundle["lr"]
            shrink = bundle["weight_decay"]
            decay = bundle["mu"]
            for tensor in bundle["params"]:
                if tensor.grad is None:
                    continue
                store = self.state[tensor]
                if not store:
                    store["momentum"] = torch.zeros_like(tensor)
                delta = _compute_update(tensor.grad, store["momentum"], decay=decay)
                tensor.mul_(1 - step_size * shrink)
                tensor.add_(delta, alpha=-step_size)


def build_optimizer(params, lr: float = 0.02, **kwargs):
    return OrthogonalMomentumSolver(list(params), lr=lr, **kwargs)
