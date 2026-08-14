import os

import sys

import uuid

import time

from pathlib import Path

import torch

from torch import Tensor, nn

from torch.optim import AdamW

import torch.nn.functional as F

import torch.distributed as dist

CONTRA_MUON = 0.4

TARGET_UW = 0.35

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def scale_to_unit_operator_norm(G: Tensor, eps: float = 1e-10) -> Tensor:
    X = G.float()
    v = torch.ones(X.size(-1), dtype=X.dtype, device=X.device)
    v = v / torch.clamp(v.norm(), min=eps)
    for _ in range(5):
        u = X @ v
        u = u / torch.clamp(u.norm(), min=eps)
        v = X.mT @ u
        v = v / torch.clamp(v.norm(), min=eps)
    op_norm = torch.clamp((X @ v).norm(), min=eps)
    return G / op_norm.to(G.dtype)

def muon_update(grad, momentum, second_moment, mu=0.95, beta2=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    normalized_grad = scale_to_unit_operator_norm(update.clone())
    update = zeropower_via_newtonschulz5(update)
    opower_frobenius_norm = update.norm()

    # https://github.com/nilin/contra-muon
    update = update - CONTRA_MUON / 2 * normalized_grad
    update = update * opower_frobenius_norm / torch.clamp(update.norm(), min=1e-10)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    # Per-row variance (or per-col if matrix is wide). Keep along the longer dim.
    if update.size(-2) >= update.size(-1):
        per_row_var = (update * update).mean(dim=-1, keepdim=True)  # shape (..., m, 1)
    else:
        per_row_var = (update * update).mean(dim=-2, keepdim=True)  # shape (..., 1, n)
    second_moment.lerp_(per_row_var.float(), 1 - beta2)
    vnorm = update.norm()
    update = update * second_moment.clamp_min(1e-10).rsqrt().to(update.dtype)
    vnorm_new = update.norm().clamp_min(1e-10)
    update = update * (vnorm / vnorm_new)
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                        # Second-moment buffer matches per-row-or-col shape.
                        if p.size(-2) >= p.size(-1):
                            state["second_moment"] = torch.zeros((*p.shape[:-1], 1),
                                dtype=torch.float32, device=p.device)
                        else:
                            state["second_moment"] = torch.zeros((*p.shape[:-2], 1, p.shape[-1]),
                                dtype=torch.float32, device=p.device)
                    update = muon_update(p.grad, state["momentum"], state["second_moment"],
                                         mu=group["mu"])
                    # u/w-floor. If u/w would be below TARGET (0.35), scale UP to maintain 0.35.
                    # If u/w >= TARGET (early training when weights are small), leave update alone.
                    p_fro = p.float().norm().clamp_min(1e-8)
                    u_fro = update.float().norm().clamp_min(1e-8)
                    cur_uw = u_fro / p_fro
                    scale = torch.where(cur_uw < TARGET_UW, TARGET_UW * p_fro / u_fro, torch.ones_like(p_fro))
                    update = update * scale.to(update.dtype)
                    # WD set to 0 — u/w target replaces wd's role (smaller updates as p grows).
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
