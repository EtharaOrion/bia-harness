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

NOR_BETA2 = 0.95

SOAP_BETA2 = 0.90

SOAP_PRECONDITION_FREQUENCY = 10

SOAP_DENOM_POWER = 0.50

SOAP_BLEND = 1.00

SOAP_UPDATE_BEFORE_USE = False

SOAP_PARAM_MODE = "mlp_plus_attn_proj"

ATTN_SOAP_DENOM_FLOOR = float(os.environ.get("ATTN_SOAP_DENOM_FLOOR", "0.20"))

ATTN_EARLY_TRUST_FLOOR = float(os.environ.get("ATTN_EARLY_TRUST_FLOOR", "0.45"))

ATTN_EARLY_TRUST_CAP = float(os.environ.get("ATTN_EARLY_TRUST_CAP", "0.85"))

ATTN_TRUST_FLOOR_END_STEP = int(os.environ.get("ATTN_TRUST_FLOOR_END_STEP", "1375"))

ATTN_TRUST_FLOOR_FADE_END_STEP = int(os.environ.get("ATTN_TRUST_FLOOR_FADE_END_STEP", "1625"))

ATTN_TRUST_MIN_AGREE = float(os.environ.get("ATTN_TRUST_MIN_AGREE", "0.20"))

ATTN_TRUST_MIN_GRAD_ALIGN = float(os.environ.get("ATTN_TRUST_MIN_GRAD_ALIGN", "0.00"))

ATTN_TRUST_POWER = float(os.environ.get("ATTN_TRUST_POWER", "1.00"))

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

def should_soap_param(name: str) -> bool:
    is_mlp_fc = name.endswith(".mlp.fc.weight")
    is_mlp_proj = name.endswith(".mlp.proj.weight")
    is_attn_proj = name.endswith(".attn.proj.weight")
    is_qkv = (
        name.endswith(".attn.q.weight")
        or name.endswith(".attn.k.weight")
        or name.endswith(".attn.v.weight")
    )
    if SOAP_PARAM_MODE == "mlp_all":
        return is_mlp_fc or is_mlp_proj
    if SOAP_PARAM_MODE == "mlp_fc":
        return is_mlp_fc
    if SOAP_PARAM_MODE == "mlp_proj":
        return is_mlp_proj
    if SOAP_PARAM_MODE == "mlp_plus_attn_proj":
        return is_mlp_fc or is_mlp_proj or is_attn_proj
    if SOAP_PARAM_MODE == "mlp_plus_qkv":
        return is_mlp_fc or is_mlp_proj or is_qkv
    if SOAP_PARAM_MODE == "all_hidden":
        return is_mlp_fc or is_mlp_proj or is_attn_proj or is_qkv
    raise ValueError(f"unknown SOAP_PARAM_MODE={SOAP_PARAM_MODE}")

def is_attn_proj_param(name: str) -> bool:
    return name.endswith(".attn.proj.weight")

def tensor_cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a_f, b_f = a.float(), b.float()
    return (a_f * b_f).sum() / (a_f.norm() * b_f.norm()).clamp_min(eps)

def trust_gate(raw: Tensor, soap: Tensor, grad: Tensor, eps: float = 1e-8) -> Tensor:
    # SOAP is trusted when it still points with raw momentum and is at least as
    # gradient-aligned as raw momentum. This catches stale whitening bases.
    raw_grad = tensor_cosine(raw, grad, eps)
    soap_grad = tensor_cosine(soap, grad, eps)
    soap_raw = tensor_cosine(soap, raw, eps)

    agree_gate = ((soap_raw - ATTN_TRUST_MIN_AGREE) / (1 - ATTN_TRUST_MIN_AGREE)).clamp(0, 1)
    denom = (raw_grad - ATTN_TRUST_MIN_GRAD_ALIGN).clamp_min(eps)
    grad_gate = ((soap_grad - ATTN_TRUST_MIN_GRAD_ALIGN) / denom).clamp(0, 1)
    gate = (agree_gate * grad_gate).clamp(0, 1)
    if ATTN_TRUST_POWER != 1.0:
        gate = gate.pow(ATTN_TRUST_POWER)
    return gate

def early_trust_floor_for_step(step: int) -> float:
    if ATTN_TRUST_FLOOR_FADE_END_STEP <= ATTN_TRUST_FLOOR_END_STEP:
        return 0.0 if step >= ATTN_TRUST_FLOOR_FADE_END_STEP else ATTN_EARLY_TRUST_FLOOR
    if step < ATTN_TRUST_FLOOR_END_STEP:
        return ATTN_EARLY_TRUST_FLOOR
    if step >= ATTN_TRUST_FLOOR_FADE_END_STEP:
        return 0.0
    return ATTN_EARLY_TRUST_FLOOR * (
        ATTN_TRUST_FLOOR_FADE_END_STEP - step
    ) / (ATTN_TRUST_FLOOR_FADE_END_STEP - ATTN_TRUST_FLOOR_END_STEP)

def bounded_trust_gate(gate: Tensor, step: int) -> Tensor:
    floor = early_trust_floor_for_step(step)
    cap = ATTN_EARLY_TRUST_CAP if step < ATTN_TRUST_FLOOR_FADE_END_STEP else 1.0
    return gate.clamp(min=floor, max=cap)

def norm_preserving_blend(raw: Tensor, soap: Tensor, gate: Tensor, eps: float = 1e-8) -> Tensor:
    blended = raw + (soap - raw) * gate.to(raw.dtype)
    raw_norm = raw.float().norm().clamp_min(eps)
    blended_norm = blended.float().norm().clamp_min(eps)
    return (blended * (raw_norm / blended_norm).to(blended.dtype)).to(raw.dtype)

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
    precond.mul_(update_f.norm() / precond.norm().clamp_min(eps))
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

def muon_update(update, second_moment, beta2=NOR_BETA2):
    normalized_grad = scale_to_unit_operator_norm(update.clone())
    update = zeropower_via_newtonschulz5(update)
    opower_frobenius_norm = update.norm()

    # https://github.com/nilin/contra-muon
    update = update - CONTRA_MUON / 2 * normalized_grad
    update = update * opower_frobenius_norm / torch.clamp(update.norm(), min=1e-10)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
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
    def __init__(self, named_params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(named_params, list) and len(named_params) >= 1
        self.soap_params = {p for n, p in named_params if should_soap_param(n)}
        self.attn_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_proj_param(n)}
        self.step_count = 0
        params = sorted([p for _, p in named_params], key=lambda x: x.size(), reverse=True)
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
                        if p in self.soap_params:
                            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                            state["row_gg"] = torch.zeros(p.size(0), p.size(0), dtype=torch.float32, device=p.device)
                            state["col_gg"] = torch.zeros(p.size(1), p.size(1), dtype=torch.float32, device=p.device)
                            state["q_row"] = None
                            state["q_col"] = None
                            state["soap_step"] = 0
                        # Second-moment buffer matches per-row-or-col shape.
                        if p.size(-2) >= p.size(-1):
                            state["second_moment"] = torch.zeros((*p.shape[:-1], 1),
                                dtype=torch.float32, device=p.device)
                        else:
                            state["second_moment"] = torch.zeros((*p.shape[:-2], 1, p.shape[-1]),
                                dtype=torch.float32, device=p.device)
                    grad = p.grad
                    state["momentum"].lerp_(grad, 1 - group["mu"])
                    momentum_update = grad.lerp(state["momentum"], group["mu"])
                    is_attn_soap = p in self.attn_soap_params
                    use_soap = p in self.soap_params
                    if use_soap and SOAP_UPDATE_BEFORE_USE:
                        soap_update_preconditioner(grad, state)
                    if use_soap:
                        if is_attn_soap:
                            soap_update = soap_precondition_momentum(
                                momentum_update, state, blend=1.0,
                                denom_floor_ratio=ATTN_SOAP_DENOM_FLOOR
                            )
                            gate = bounded_trust_gate(
                                trust_gate(momentum_update, soap_update, grad),
                                self.step_count
                            )
                            momentum_update = norm_preserving_blend(momentum_update, soap_update, gate)
                        else:
                            momentum_update = soap_precondition_momentum(momentum_update, state, blend=SOAP_BLEND)
                    update = muon_update(momentum_update, state["second_moment"])
                    # u/w-floor. If u/w would be below TARGET (0.35), scale UP to maintain 0.35.
                    # If u/w >= TARGET (early training when weights are small), leave update alone.
                    p_fro = p.float().norm().clamp_min(1e-8)
                    u_fro = update.float().norm().clamp_min(1e-8)
                    cur_uw = u_fro / p_fro
                    scale = torch.where(cur_uw < TARGET_UW, TARGET_UW * p_fro / u_fro, torch.ones_like(p_fro))
                    update = update * scale.to(update.dtype)
                    # WD set to 0 — u/w target replaces wd's role (smaller updates as p grows).
                    p.add_(update, alpha=-group["lr"])
                    if use_soap and not SOAP_UPDATE_BEFORE_USE:
                        soap_update_preconditioner(grad, state)
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.step_count += 1
