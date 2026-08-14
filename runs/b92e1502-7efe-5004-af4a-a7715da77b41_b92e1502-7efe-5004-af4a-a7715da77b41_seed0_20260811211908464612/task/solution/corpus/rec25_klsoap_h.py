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

def scale_invariant_update_(param: Tensor, update: Tensor, lr: float, eps: float = 1e-10):
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    param.copy_(new_param / torch.clamp(new_param.norm(), min=eps) * p_norm)

def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.T)

def _initial_orthogonal_matrix(matrix: Tensor) -> Tensor:
    matrix = _symmetrize(matrix.float())
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=torch.float32)
    try:
        _, q = torch.linalg.eigh(matrix + 1e-30 * eye)
    except RuntimeError:
        _, q = torch.linalg.eigh((matrix + 1e-30 * eye).double())
        q = q.float()
    return torch.flip(q, dims=[1]).contiguous()

def project_to_klsoap_basis(grad: Tensor, state) -> Tensor:
    q_left, q_right = state["Q"]
    return q_left.T @ grad.float() @ q_right

def project_from_klsoap_basis(grad: Tensor, state) -> Tensor:
    q_left, q_right = state["Q"]
    return q_left @ grad.float() @ q_right.T

def init_2d_klsoap_state_(state, grad: Tensor, shampoo_beta: float, init_factor: float):
    grad = grad.detach().float()
    rows, cols = grad.shape
    state["step"] = 0
    state["GG"] = [
        (grad @ grad.T / cols).contiguous(),
        (grad.T @ grad / rows).contiguous(),
    ]
    state["Q"] = [
        _initial_orthogonal_matrix(state["GG"][0]),
        _initial_orthogonal_matrix(state["GG"][1]),
    ]
    inv = init_factor ** -0.5
    state["eigen_sqrt_inv"] = [
        torch.full((rows,), inv, device=grad.device, dtype=torch.float32),
        torch.full((cols,), inv, device=grad.device, dtype=torch.float32),
    ]
    state["exp_avg"] = torch.zeros_like(grad, dtype=torch.float32)
    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)
    state["shampoo_beta"] = shampoo_beta

def _update_eigen_sqrt_inv_(state, diag: Tensor, idx: int, beta: float):
    old_eigen = state["eigen_sqrt_inv"][idx].float().square().reciprocal()
    old_eigen = old_eigen.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    eigen = beta * old_eigen + (1.0 - beta) * diag.detach().float()
    inv_sqrt = eigen.clamp_min(1e-30).rsqrt().clamp(max=4000.0)
    state["eigen_sqrt_inv"][idx] = inv_sqrt.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).contiguous()

def update_2d_klsoap_preconditioner_(grad: Tensor, state):
    grad = grad.detach().float()
    q_left, q_right = state["Q"]
    inv_left, inv_right = state["eigen_sqrt_inv"]
    beta = state["shampoo_beta"]
    rows, cols = grad.shape

    right_whitened = (q_right.T @ grad.T) * inv_right.view(-1, 1)
    left_target = right_whitened.T @ right_whitened / cols
    left_whitened = (q_left.T @ grad) * inv_left.view(-1, 1)
    right_target = left_whitened.T @ left_whitened / rows
    state["GG"][0].mul_(beta).add_(left_target, alpha=1.0 - beta)
    state["GG"][1].mul_(beta).add_(right_target, alpha=1.0 - beta)
    state["GG"][0] = _symmetrize(state["GG"][0]).contiguous()
    state["GG"][1] = _symmetrize(state["GG"][1]).contiguous()

    projected = q_left.T @ grad @ q_right
    left_diag = (projected * inv_right.view(1, -1)).square().mean(dim=1)
    right_diag = (projected * inv_left.view(-1, 1)).square().mean(dim=0)
    _update_eigen_sqrt_inv_(state, left_diag, 0, beta)
    _update_eigen_sqrt_inv_(state, right_diag, 1, beta)

def refresh_klsoap_basis_(state):
    exp_avg_original = project_from_klsoap_basis(state["exp_avg"], state)
    refreshed = []
    for gg, q in zip(state["GG"], state["Q"]):
        new_q, _ = torch.linalg.qr(gg.float() @ q.float())
        refreshed.append(new_q.contiguous())
    state["Q"] = refreshed
    state["exp_avg"] = project_to_klsoap_basis(exp_avg_original, state).contiguous()

def klsoap_direction(state, grad: Tensor, beta1: float, beta2: float, eps: float) -> Tensor:
    grad_projected = project_to_klsoap_basis(grad.detach().float(), state)
    state["exp_avg"].mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
    state["exp_avg_sq"].mul_(beta2).addcmul_(grad_projected, grad_projected, value=1.0 - beta2)
    preconditioned = state["exp_avg"] / (state["exp_avg_sq"].sqrt() + eps)
    return project_from_klsoap_basis(preconditioned, state)

class KLSOAPH(torch.optim.Optimizer):
    def __init__(
        self, params, lr=0.018, beta1=0.95, beta2=0.9,
        shampoo_beta=0.9, eps=1e-8, precondition_frequency=1,
    ):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, shampoo_beta=shampoo_beta,
            eps=eps, precondition_frequency=precondition_frequency,
        )
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
                        init_2d_klsoap_state_(state, p.grad, group["shampoo_beta"], init_factor=0.1)
                        dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
                        continue
                    state["step"] += 1
                    update = klsoap_direction(state, p.grad, group["beta1"], group["beta2"], group["eps"])
                    update_2d_klsoap_preconditioner_(p.grad, state)
                    if state["step"] % group["precondition_frequency"] == 0:
                        refresh_klsoap_basis_(state)
                    scale_invariant_update_(p, update, group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
