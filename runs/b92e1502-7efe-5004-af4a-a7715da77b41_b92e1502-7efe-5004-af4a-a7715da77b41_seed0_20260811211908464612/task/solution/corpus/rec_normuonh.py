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

def normuon_update(grad, momentum, v_buf, mu=0.95, beta2=0.95, eps=1e-10, nesterov=True):
    """NorMuon direction: NS-orthogonalised gradient followed by Adafactor-style variance
    preconditioning along the SHORT axis (per https://arxiv.org/pdf/2510.05491). The
    variance buffer `v_buf` is an EMA of squared post-NS values along the short axis,
    persistent across optimizer steps."""
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    # Variance per row (if rows >= cols) or per column (otherwise) — Adafactor-style.
    if grad.size(-2) >= grad.size(-1):
        v_new = update.square().mean(dim=-1, keepdim=True)
    else:
        v_new = update.square().mean(dim=-2, keepdim=True)
    # `update` is bf16 after NS5; cast `v_new` to `v_buf`'s dtype (float32) so lerp_ matches.
    v_buf.lerp_(v_new.to(v_buf.dtype), 1 - beta2)
    update = update * v_buf.clamp_min(eps).rsqrt()
    return update

def scale_invariant_update_(param: Tensor, update: Tensor, lr: float, eps: float = 1e-10) -> None:
    """Hyperball-constrained step: take a preconditioned update of size lr * ||param||,
    then renormalise back onto the Frobenius sphere of the parameter's initial radius. Preserves
    ||param|| exactly across training; the invariant lets us drop weight decay on hidden
    matrices entirely (the constraint already prevents norm growth)."""
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)

class NorMuonH(torch.optim.Optimizer):
    """NorMuonH: NS-orthogonalised gradient + Adafactor-style row/column variance
    preconditioning (NorMuon, https://arxiv.org/pdf/2510.05491) + hyperball Frobenius-
    norm-preserving step. Used here for ALL hidden 2D weight matrices — q, k, v, mlp.fc,
    attn.proj, mlp.proj — under non-zero (Kaiming-derived) init."""
    def __init__(self, params, lr=0.018, mu=0.95, beta2=0.95, eps=1e-10):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, mu=mu, beta2=beta2, eps=eps)
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
                        # variance buffer shape: keepdim along the short axis
                        v_shape = list(p.shape)
                        if p.size(-2) >= p.size(-1):
                            v_shape[-1] = 1
                        else:
                            v_shape[-2] = 1
                        state["v"] = torch.zeros(v_shape, dtype=p.dtype, device=p.device)
                    update = normuon_update(
                        p.grad, state["momentum"], state["v"],
                        mu=group["mu"], beta2=group["beta2"], eps=group["eps"],
                    )
                    scale_invariant_update_(p, update, group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
