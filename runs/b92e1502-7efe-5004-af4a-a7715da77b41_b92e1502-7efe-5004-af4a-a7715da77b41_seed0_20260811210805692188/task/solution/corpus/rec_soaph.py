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

def soap_update(
    grad: Tensor,
    momentum_buffer: Tensor,
    square_buffer: Tensor,
    left_buffer: Tensor,
    right_buffer: Tensor,
    Q_left: Tensor,
    Q_right: Tensor,
    momentum: float,
    ema_beta: float,
    eps: float,
) -> Tensor:
    # Momentum EMA:
    momentum_buffer.lerp_(grad, 1.0 - momentum)

    M = momentum_buffer

    # Matrix second moments:
    left_buffer.mul_(ema_beta).addmm_(
        grad,
        grad.T,
        beta=1.0,
        alpha=1.0 - ema_beta,
    )

    right_buffer.mul_(ema_beta).addmm_(
        grad.T,
        grad,
        beta=1.0,
        alpha=1.0 - ema_beta,
    )

    # Project into current SOAP basis.
    M_eigen = Q_left.T @ M @ Q_right
    G_eigen = Q_left.T @ grad @ Q_right

    # Diagonal second moment in SOAP basis.
    square_buffer.mul_(ema_beta).addcmul_(
        G_eigen,
        G_eigen,
        value=1.0 - ema_beta,
    )

    # Precondition in SOAP basis and rotate back.
    U_eigen = M_eigen / square_buffer.sqrt().add(eps)
    U = Q_left @ U_eigen @ Q_right.T

    return U

class SOAPH(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        ema_beta: float = 0.9,
        momentum: float = 0.95,
        q_freq: int = 1,
        eps: float = 1e-8,
        hyperball_eps: float = 1e-10,
    ):
        assert isinstance(params, list)
        assert len(params) >= 1
        assert isinstance(params[0], torch.nn.Parameter)

        for p in params:
            assert p.ndim == 2, f"SOAPH here expects only 2D parameters, got shape {tuple(p.shape)}"

        params = sorted(params, key=lambda x: x.numel(), reverse=True)

        defaults = dict(
            lr=lr,
            ema_beta=ema_beta,
            momentum=momentum,
            q_freq=q_freq,
            eps=eps,
            hyperball_eps=hyperball_eps,
        )

        print("SOAPH hyperparams:", defaults)

        super().__init__(params, defaults)

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict) -> None:
        m, n = p.shape

        state["step"] = 0

        state["momentum_buffer"] = torch.zeros_like(p)
        state["square_buffer"] = torch.zeros_like(p)

        state["left_buffer"] = torch.zeros(
            (m, m),
            dtype=p.dtype,
            device=p.device,
        )

        state["right_buffer"] = torch.zeros(
            (n, n),
            dtype=p.dtype,
            device=p.device,
        )

        state['Q_left'] = _initial_orthogonal_matrix(p.grad @ p.grad.T)
        state["Q_right"] = _initial_orthogonal_matrix(p.grad.T @ p.grad)

    @torch.no_grad()
    def _refresh_q(self, state: dict) -> None:
        L = state["left_buffer"]
        R = state["right_buffer"]

        Q_left_old = state["Q_left"]
        Q_right_old = state["Q_right"]

        Q_left, _ = torch.linalg.qr(
            L.float() @ Q_left_old.float(),
            mode="reduced",
        )

        Q_right, _ = torch.linalg.qr(
            R.float() @ Q_right_old.float(),
            mode="reduced",
        )

        state["Q_left"].copy_(Q_left.to(dtype=Q_left_old.dtype))
        state["Q_right"].copy_(Q_right.to(dtype=Q_right_old.dtype))

    @torch.no_grad()
    def _step_param(self, p: Tensor, group: dict) -> None:
        if p.grad is None:
            return

        if p.grad.is_sparse:
            raise RuntimeError("SOAPH does not support sparse gradients")

        if p.ndim != 2:
            raise RuntimeError(f"SOAPH expects only 2D parameters, got shape {tuple(p.shape)}")

        grad = p.grad
        state = self.state[p]

        if len(state) == 0:
            self._init_state(p, state)

        lr = group["lr"]
        ema_beta = group["ema_beta"]
        momentum = group["momentum"]
        q_freq = group["q_freq"]
        eps = group["eps"]
        hyperball_eps = group["hyperball_eps"]

        state["step"] += 1
        step = state["step"]

        update = soap_update(
            grad=grad,
            momentum_buffer=state["momentum_buffer"],
            square_buffer=state["square_buffer"],
            left_buffer=state["left_buffer"],
            right_buffer=state["right_buffer"],
            Q_left=state["Q_left"],
            Q_right=state["Q_right"],
            momentum=momentum,
            ema_beta=ema_beta,
            eps=eps,
        )

        if step % q_freq == 0:
            self._refresh_q(state)

        scale_invariant_update_(
            param=p,
            update=update,
            lr=lr,
            eps=hyperball_eps,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()

        if distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        for group in self.param_groups:
            params = group["params"]

            for i, p in enumerate(params):
                owner = i % world_size

                if rank == owner:
                    self._step_param(p, group)

                if distributed:
                    dist.broadcast(p, src=owner)

        return loss
