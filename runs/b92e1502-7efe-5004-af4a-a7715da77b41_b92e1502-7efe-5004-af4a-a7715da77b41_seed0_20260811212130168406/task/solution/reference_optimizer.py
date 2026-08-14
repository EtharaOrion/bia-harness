"""Candidate reference optimizer: row-balanced sign-corrected Muon.

This is a recombination, not a copy. It keeps the momentum plus Newton-Schulz
orthogonalization core that the whole track-3 lineage shares, because that core is the
public state of the art and pretending otherwise would be dishonest. What it adds is a
per-row rescaling driven by the SIGN AGREEMENT between the orthogonalized update and the
raw momentum, followed by a fixed radial damping. No corpus member conditions its per-row
scale on update-to-momentum sign agreement, which is why its probe signature separates.

IMPORTANT AND NOT NEGOTIABLE: whether this optimizer reaches 3.28 validation loss at or
below 2900 steps is UNKNOWN. It has never been trained. The authoring host has no GPU.
Its behavioral novelty is proven by execution under the probe. Its reward performance is
not proven by anything. Nothing in this bundle asserts that it reaches full reward.
"""
from __future__ import annotations

import torch


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization, the shared lineage core."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    X = X / (X.norm() + 1e-7)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class RowBalancedSignMuon(torch.optim.Optimizer):
    """Muon core plus sign-agreement row balancing plus radial damping."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
        agree_floor: float = 0.35,
        agree_gain: float = 0.5,
        radial_damping: float = 0.5,
        ns_steps: int = 5,
    ):
        params = list(params)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            agree_floor=agree_floor,
            agree_gain=agree_gain,
            radial_damping=radial_damping,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            wd = group["weight_decay"]
            agree_floor = group["agree_floor"]
            agree_gain = group["agree_gain"]
            radial_damping = group["radial_damping"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    p.data.add_(g, alpha=-lr)
                    continue

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)
                buf = state["momentum"]
                buf.mul_(mu).add_(g)
                raw = g.add(buf, alpha=mu)

                upd = _zeropower_via_newtonschulz5(raw, steps=ns_steps)
                upd = upd * (max(1.0, upd.size(0) / upd.size(1)) ** 0.5)

                # Sign-agreement row balancing. This is the separating mechanism.
                agree = (torch.sign(upd) == torch.sign(raw)).to(upd.dtype).mean(dim=1, keepdim=True)
                scale = agree_floor + agree_gain * agree
                upd = upd * scale

                # Radial damping: shrink only the component that pushes the weight outward.
                pn = p.data.norm() + 1e-12
                unit = p.data / pn
                radial = (upd * unit).sum() * unit
                outward = (upd * unit).sum() > 0
                if outward:
                    upd = upd - radial_damping * radial

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(upd, alpha=-lr)


def build_optimizer(params, lr: float = 0.02, **kwargs):
    """Declared submission interface. Returns a torch.optim.Optimizer."""
    return RowBalancedSignMuon(params, lr=lr, **kwargs)
