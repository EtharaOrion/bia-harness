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

def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, refresh_interval=64):
        assert isinstance(params, list) and len(params) >= 1
        if isinstance(params[0], torch.nn.Parameter):
            params = sorted(params, key=lambda x: x.size(), reverse=True)
        else:
            assert isinstance(params[0], dict)
            params = [dict(group, params=sorted(list(group["params"]), key=lambda x: x.size(), reverse=True))
                      for group in params]
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self.global_step = 0
        self._precond_attached = self._precond_has_inv = False
        self._precond_stats = []
        self._precond_K = self._precond_count = None
        self._precond_cov = None
        self._precond_matrix_stat_idx = None
        self._precond_matrix_count = self._precond_matrix_count_clamped = None
        self._precond_alpha = None
        self.refresh_interval = refresh_interval

    @staticmethod
    def _eye_(x, value):
        x.zero_()
        x.diagonal(dim1=-2, dim2=-1).fill_(value)

    @torch.no_grad()
    def attach_preconditioner(self):
        self._precond_attached = True
        stats = {}
        for group in self.param_groups:
            for p in group["params"]:
                ref = getattr(p, "_stats_ref", None)
                assert ref is not None, f"Missing preconditioner stats for parameter {tuple(p.shape)}"
                key = id(ref["accum"])
                if key not in stats:
                    stats[key] = dict(accum=ref["accum"], count=ref["count"])
                self.state[p]["precond"] = stats[key]
        self._precond_stats = list(stats.values())
        d = self._precond_stats[0]["accum"].size(-1)
        n_by_stat = [s["accum"].numel() // (d * d) for s in self._precond_stats]
        n = sum(n_by_stat)
        device = self._precond_stats[0]["accum"].device
        self._precond_K = torch.empty(n, d, d, device=device)
        self._precond_cov = torch.empty_like(self._precond_K)
        self._precond_count = torch.empty(len(self._precond_stats), device=device)
        self._precond_matrix_count = torch.empty(n, device=device)
        self._precond_matrix_count_clamped = torch.empty(n, device=device)
        self._precond_matrix_stat_idx = torch.repeat_interleave(
            torch.arange(len(self._precond_stats), device=device),
            torch.tensor(n_by_stat, device=device),
        )
        self._precond_alpha = torch.empty(n, 1, 1, device=device)
        i = 0
        for s, stat_n in zip(self._precond_stats, n_by_stat):
            s["n"] = stat_n
            s["cov"] = self._precond_cov[i:i+stat_n].reshape_as(s["accum"])
            s["inv"] = self._precond_K[i:i+stat_n].reshape_as(s["accum"])
            self._eye_(s["cov"], 0.001)
            self._eye_(s["inv"], 1.0)
            s["accum"].zero_()
            s["count"].zero_()
            i += stat_n

    def precond_flag_for_step(self, step: int):
        return self._precond_attached and (int(step) + 1) % self.refresh_interval == 0

    @torch.no_grad()
    def _refresh_preconditioner(self):
        K, d, i = self._precond_K, self._precond_K.size(-1), 0
        for j, s in enumerate(self._precond_stats):
            n = s["n"]
            K[i:i+n].copy_(s["accum"].reshape(n, d, d))
            self._precond_count[j].copy_(s["count"])
            i += n
        dist.all_reduce(K, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._precond_count, op=dist.ReduceOp.SUM)

        torch.index_select(self._precond_count, 0, self._precond_matrix_stat_idx, out=self._precond_matrix_count)
        self._precond_matrix_count_clamped.copy_(self._precond_matrix_count).clamp_min_(1)
        K.div_(self._precond_matrix_count_clamped.view(-1, 1, 1))
        self._precond_alpha.copy_(self._precond_matrix_count.gt(0).view(-1, 1, 1)).mul_(0.05)
        self._precond_cov.lerp_(K, self._precond_alpha)

        diag = self._precond_cov.diagonal(dim1=-2, dim2=-1)
        reg = (diag.sum(-1) / d * 0.2 + 1e-8).unsqueeze(-1)
        diag.add_(reg)
        L, info = torch.linalg.cholesky_ex(self._precond_cov, upper=False, check_errors=False)
        diag.sub_(reg)
        torch.cholesky_inverse(L, upper=False, out=K)
        if info.any():
            self._eye_(K[info != 0], 1.0)

        for s in self._precond_stats:
            s["accum"].zero_()
            s["count"].zero_()
        self._precond_has_inv = True

    @torch.no_grad()
    def _precondition_grads(self):
        if not self._precond_has_inv:
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                if grad is not p.grad:
                    p.grad = grad
                inv = self.state[p]["precond"]["inv"]
                if inv.ndim == 3:
                    d = inv.size(-1)
                    g = grad.view(grad.size(0), 4, d).permute(1, 0, 2)
                    p.grad.copy_(torch.bmm(g, inv).permute(1, 0, 2).reshape_as(p.grad))
                else:
                    p.grad.copy_(grad @ inv)

    @torch.no_grad()
    def step(self):
        if self._precond_attached and self.precond_flag_for_step(self.global_step):
            self._refresh_preconditioner()
        if self._precond_attached:
            self._precondition_grads()

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.global_step += 1

device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
