import os

import sys

import argparse

import uuid

import time

from pathlib import Path

import torch

from torch import Tensor, nn

from torch.optim import AdamW

import torch.nn.functional as F

import torch.distributed as dist

CONTRA_MUON_COEFF = -0.2

CONTRA_HOLD_END_STEP = 0

CONTRA_TO_NORMAL_END_STEP = 2500

TARGET_UW = 0.3825

NOR_BETA2 = 1.0

SOAP_BETA2 = 0.90

SOAP_PRECONDITION_FREQUENCY = 10

SOAP_DENOM_POWER = 0.50

SOAP_BLEND = 1.00

V_SOAP_BLEND = 0.95

ATTN_SOAP_DENOM_FLOOR = 0.55

RADIAL_OUTWARD_SCALE = 0.5

RADIAL_INWARD_SCALE = 1.0

def gram_frobenius_norm_estimate(G: Tensor, keepdim: bool = False, eps: float = 1e-10) -> Tensor:
    X = G.float()
    gram = X.mT @ X if X.size(-2) > X.size(-1) else X @ X.mT
    return gram.norm(dim=(-2, -1), keepdim=keepdim).sqrt().clamp_min(eps)

def _ns_inner(X: Tensor) -> Tensor:
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

_AURORA_K = 3

_AURORA_BETA = 0.25

_AURORA_EPS = 1e-7

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    is_originally_wide = G.size(-2) < G.size(-1)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    if is_originally_wide:
        Xt = X.mT
        Xt32 = Xt.to(torch.float32)
        target_row_sq = Xt.size(-1) / Xt.size(-2)
        row_norm = Xt32.norm(dim=-1, keepdim=True).clamp_(min=_AURORA_EPS)
        D = 1.0 / row_norm
        U = None
        for k in range(_AURORA_K):
            scaled = (D * Xt32).to(Xt.dtype)
            scaled_wide = scaled.mT
            scaled_wide = scaled_wide / gram_frobenius_norm_estimate(scaled_wide, keepdim=True, eps=1e-7).to(scaled_wide.dtype)
            U_wide = _ns_inner(scaled_wide)
            U = U_wide.mT
            if k < _AURORA_K - 1:
                U32 = U.to(torch.float32)
                row_sq = U32.pow(2).sum(dim=-1, keepdim=True).clamp_(min=_AURORA_EPS * _AURORA_EPS)
                D = D * (target_row_sq / row_sq).pow(_AURORA_BETA)
        X = U.mT
    else:
        X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
        X = _ns_inner(X)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def scale_to_unit_operator_norm(G: Tensor, eps: float = 1e-10) -> Tensor:
    return G / gram_frobenius_norm_estimate(G, eps=eps).to(G.dtype)

def should_soap_param(name: str) -> bool:
    return (
        name.endswith(".mlp.fc.weight")
        or name.endswith(".mlp.proj.weight")
        or name.endswith(".attn.v.weight")
    )

def is_v_param(name: str) -> bool:
    return name.endswith(".attn.v.weight")

def is_attn_param(name: str) -> bool:
    return is_v_param(name)

def scale_radial_update(update: Tensor, param: Tensor, step: int, eps: float = 1e-12) -> Tensor:
    update_f = update.float()
    param_f = param.float()
    denom = (param_f * param_f).sum().clamp_min(eps)
    coeff = (update_f * param_f).sum() / denom
    radial = coeff * param_f
    tangential = update_f - radial
    # p.add_(update, alpha=-lr), so actual movement is -update.
    # Outward movement means (-update) is aligned with p, i.e. coeff < 0.
    outward_scale = RADIAL_OUTWARD_SCALE
    radial_scale = torch.where(
        coeff < 0,
        update_f.new_tensor(outward_scale),
        update_f.new_tensor(RADIAL_INWARD_SCALE),
    )
    return (tangential + radial_scale * radial).to(update.dtype)

def target_radius_after_update(param: Tensor, update: Tensor, lr: float, eps: float = 1e-8) -> Tensor:
    param_f = param.float()
    update_f = update.float()
    before_norm = param_f.norm().clamp_min(eps)
    # Use only the radial component's first-order radius change as the intended
    # radius change; the post-step rescale below removes finite tangent drift.
    radial_delta = -lr * (update_f * param_f).sum() / before_norm
    return (before_norm + radial_delta).clamp_min(eps)

def rescale_to_radius(param: Tensor, target_norm: Tensor, eps: float = 1e-8):
    after_norm = param.float().norm().clamp_min(eps)
    param.mul_((target_norm / after_norm).to(param.dtype))

def soap_eigenbasis(mat: Tensor) -> Tensor:
    try:
        _, q = torch.linalg.eigh(mat + 1e-30 * torch.eye(mat.size(0), device=mat.device))
    except RuntimeError:
        _, q = torch.linalg.eigh(mat.double() + 1e-30 * torch.eye(mat.size(0), device=mat.device))
        q = q.float()
    return torch.flip(q, [1])

def soap_basis_qr(row_gg, col_gg, q_row, q_col, exp_avg_sq):
    row_eig = torch.diag(q_row.T @ row_gg @ q_row)
    row_sort = torch.argsort(row_eig, descending=True)
    q_row = q_row[:, row_sort]
    exp_avg_sq = exp_avg_sq.index_select(0, row_sort)
    q_row, _ = torch.linalg.qr(row_gg @ q_row)

    col_eig = torch.diag(q_col.T @ col_gg @ q_col)
    col_sort = torch.argsort(col_eig, descending=True)
    q_col = q_col[:, col_sort]
    exp_avg_sq = exp_avg_sq.index_select(1, col_sort)
    q_col, _ = torch.linalg.qr(col_gg @ q_col)
    return q_row, q_col, exp_avg_sq

def soap_precondition_momentum(update, state, beta2=SOAP_BETA2, eps=1e-8,
                               blend=SOAP_BLEND, denom_floor_ratio=0.0):
    update_f = update.float()
    if state["q_row"] is None:
        return update
    q_row, q_col = state["q_row"], state["q_col"]
    projected = q_row.T @ update_f @ q_col
    state["exp_avg_sq"].mul_(beta2).add_(projected.square(), alpha=1 - beta2)
    denom = state["exp_avg_sq"].clamp_min(eps * eps).pow(SOAP_DENOM_POWER)
    if denom_floor_ratio > 0:
        denom_floor = denom.float().square().mean().sqrt().mul(denom_floor_ratio).clamp_min(eps)
        denom = denom.clamp_min(denom_floor.to(denom.dtype))
    precond = q_row @ (projected / denom) @ q_col.T
    if blend != 1.0:
        precond = blend * precond + (1 - blend) * update_f
    precond.mul_(gram_frobenius_norm_estimate(update_f, eps=eps) / gram_frobenius_norm_estimate(precond, eps=eps))
    return precond.to(update.dtype)

def soap_update_preconditioner(grad, state, shampoo_beta=SOAP_BETA2, precondition_frequency=SOAP_PRECONDITION_FREQUENCY):
    grad_f = grad.float()
    state["row_gg"].lerp_(grad_f @ grad_f.T, 1 - shampoo_beta)
    state["col_gg"].lerp_(grad_f.T @ grad_f, 1 - shampoo_beta)
    if state["q_row"] is None:
        state["q_row"] = soap_eigenbasis(state["row_gg"])
        state["q_col"] = soap_eigenbasis(state["col_gg"])
    elif state["soap_step"] > 0 and state["soap_step"] % precondition_frequency == 0:
        state["q_row"], state["q_col"], state["exp_avg_sq"] = soap_basis_qr(
            state["row_gg"], state["col_gg"], state["q_row"], state["q_col"], state["exp_avg_sq"]
        )
    state["soap_step"] += 1

def norm_preserving_soap_blend(raw: Tensor, soap: Tensor, eps: float = 1e-8) -> Tensor:
    raw_norm = gram_frobenius_norm_estimate(raw, eps=eps)
    soap_norm = gram_frobenius_norm_estimate(soap, eps=eps)
    return (soap * (raw_norm / soap_norm).to(soap.dtype)).to(raw.dtype)

def _linear_ramp(step: int, start_step: int, end_step: int) -> float:
    if end_step <= start_step:
        return 1.0 if step >= end_step else 0.0
    return min(1.0, max(0.0, (step - start_step) / (end_step - start_step)))

def contra_coeff_for_step(step: int) -> float:
    contra_to_normal = _linear_ramp(step, CONTRA_HOLD_END_STEP, CONTRA_TO_NORMAL_END_STEP)
    return CONTRA_MUON_COEFF * (1.0 - contra_to_normal)

def muon_update(update, second_moment, step, beta2=NOR_BETA2):
    normalized_grad = scale_to_unit_operator_norm(update.clone())
    ns_update = zeropower_via_newtonschulz5(update)
    update_norm_estimate = gram_frobenius_norm_estimate(ns_update)
    contra_coeff = contra_coeff_for_step(step)
    update = ns_update + contra_coeff * normalized_grad
    update = update * update_norm_estimate / gram_frobenius_norm_estimate(update)
    update *= max(1, update.size(-2) / update.size(-1))**0.5

    if update.size(-2) >= update.size(-1):
        per_row_var = (update * update).mean(dim=-1, keepdim=True)
    else:
        per_row_var = (update * update).mean(dim=-2, keepdim=True)
    second_moment.lerp_(per_row_var.float(), 1 - beta2)
    vnorm = gram_frobenius_norm_estimate(update)
    update = update * second_moment.clamp_min(1e-10).rsqrt().to(update.dtype)
    vnorm_new = gram_frobenius_norm_estimate(update)
    return update * (vnorm / vnorm_new)

class Muon(torch.optim.Optimizer):
    def __init__(self, named_params, lr=0.02, mu=0.95):
        assert isinstance(named_params, list) and len(named_params) >= 1
        self.param_names = {p: n for n, p in named_params}
        self.soap_params = {p for n, p in named_params if should_soap_param(n)}
        self.attn_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_param(n)}
        self.v_params = {p for n, p in named_params if is_v_param(n)}
        self.step_count = 0
        params = sorted([p for _, p in named_params], key=lambda x: x.size(), reverse=True)
        super().__init__(params, dict(lr=lr, mu=mu))

    def _init_state(self, p):
        state = self.state[p]
        if state:
            return state
        state["momentum"] = torch.zeros_like(p)
        if p in self.soap_params:
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["row_gg"] = torch.zeros(p.size(0), p.size(0), dtype=torch.float32, device=p.device)
            state["col_gg"] = torch.zeros(p.size(1), p.size(1), dtype=torch.float32, device=p.device)
            state["q_row"] = None
            state["q_col"] = None
            state["soap_step"] = 0
        if p.size(-2) >= p.size(-1):
            state["second_moment"] = torch.zeros((*p.shape[:-1], 1), dtype=torch.float32, device=p.device)
        else:
            state["second_moment"] = torch.zeros((*p.shape[:-2], 1, p.shape[-1]), dtype=torch.float32, device=p.device)
        return state

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
                    state = self._init_state(p)
                    grad = p.grad
                    state["momentum"].lerp_(grad, 1 - group["mu"])
                    momentum_update = grad.lerp(state["momentum"], group["mu"])
                    use_soap = p in self.soap_params
                    if use_soap:
                        if p in self.attn_soap_params:
                            soap_update = soap_precondition_momentum(
                                momentum_update,
                                state,
                                blend=V_SOAP_BLEND,
                                denom_floor_ratio=ATTN_SOAP_DENOM_FLOOR,
                            )
                            momentum_update = norm_preserving_soap_blend(momentum_update, soap_update)
                        else:
                            momentum_update = soap_precondition_momentum(momentum_update, state, blend=SOAP_BLEND)
                    update = muon_update(momentum_update, state["second_moment"], self.step_count)
                    update = scale_radial_update(update, p, self.step_count)
                    p_fro = p.float().norm().clamp_min(1e-8)
                    u_fro = update.float().norm().clamp_min(1e-8)
                    scale = torch.where(u_fro / p_fro < TARGET_UW, TARGET_UW * p_fro / u_fro, torch.ones_like(p_fro))
                    update = update * scale.to(update.dtype)
                    target_radius = target_radius_after_update(p, update, group["lr"])
                    p.add_(update, alpha=-group["lr"])
                    rescale_to_radius(p, target_radius)
                    if use_soap:
                        soap_update_preconditioner(grad, state)
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.step_count += 1
