import os

import sys

import uuid

import time

from pathlib import Path

import torch

from torch import Tensor, nn

import torch.nn.functional as F

import torch.distributed as dist

def init_psgd(grad: Tensor):
    assert grad.ndim == 2
    m, n = grad.shape
    s = grad.float().abs().pow(4).mean().clamp(min=1e-24).pow(-1/32).item()
    Q0 = torch.eye(m, dtype=torch.float32, device=grad.device) * s
    Q1 = torch.eye(n, dtype=torch.float32, device=grad.device) * s
    return Q0, Q1

def spectral_norm(A: Tensor) -> Tensor:
    # 1.5 power iterations; A is symmetric PSD at all call sites
    scale = A.norm(float("inf"))
    if scale == 0:
        return scale
    A = A / scale
    x = A[:, (A * A).sum(0).argmax()]
    x = A @ x
    x = x / torch.linalg.vector_norm(x)
    return scale * torch.linalg.vector_norm(A @ x)

def psgd_update_precond(Q0: Tensor, Q1: Tensor, U: Tensor, v: Tensor, precond_lr=1.0) -> None:
    # balance max absolute values of Q0, Q1
    rescale_factor = (Q1.abs().amax() / Q0.abs().amax()).sqrt()
    Q0.mul_(rescale_factor)
    Q1.div_(rescale_factor)

    # preconditioned update; A = Q0 U Q1^T
    A = Q0 @ U @ Q1.mT
    # preconditioned probe vector; B = Q0^{-T} v Q1^{-1}
    B = torch.linalg.solve_triangular(Q0.mT, v, upper=False, left=True)
    B = torch.linalg.solve_triangular(Q1, B, upper=True, left=False)

    AAt = A @ A.mT
    BBt = B @ B.mT
    ell0 = spectral_norm(AAt + BBt)
    Q0.sub_(precond_lr / ell0 * torch.triu(AAt - BBt) @ Q0)

    AtA = A.mT @ A
    BtB = B.mT @ B
    ell1 = spectral_norm(AtA + BtB)
    Q1.sub_(precond_lr / ell1 * torch.triu(AtA - BtB) @ Q1)

class PSGDKron(torch.optim.Optimizer):
    def __init__(self, params, lr=0.025, precond_lr=1.0, beta1=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.numel(), reverse=True)
        defaults = dict(lr=lr, precond_lr=precond_lr, beta1=beta1)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            beta1 = group["beta1"]
            precond_lr = group["precond_lr"]
            lr = group["lr"]
            params = group["params"]
            # pad so len is divisible by world_size
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params_pad), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    if grad is None:
                        continue
                    grad = grad.squeeze().float()
                    state = self.state[p]
                    if not state:
                        state["Q0"], state["Q1"] = init_psgd(grad)
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(grad)
                    state["step"] += 1
                    # adam-style beta correction to warm up preconditioner
                    state["exp_avg"].lerp_(grad, 1 - beta1)
                    update = state["exp_avg"] / (1 - beta1 ** state["step"])

                    v = torch.randn_like(update)
                    nu = torch.randn_like(update)
                    damping = torch.finfo(torch.float32).eps * update.abs()
                    Q0, Q1 = state["Q0"], state["Q1"]
                    psgd_update_precond(
                        Q0, Q1, update + damping * nu, v,
                        precond_lr=precond_lr,
                    )

                    step_update = (Q0.mT @ Q0) @ update @ (Q1.mT @ Q1)

                    # hyperball: step on the sphere, preserve param norm
                    step_update = step_update.view_as(p)
                    p_norm = p.float().norm()
                    u_norm = step_update.norm()
                    new_p = p.float() - lr * step_update * (p_norm / (u_norm + 1e-10))
                    new_p_norm = new_p.norm()
                    p.copy_((new_p / (new_p_norm + 1e-10) * p_norm).to(p.dtype))
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

precond_lr = 1.0
