"""Deterministic predicates for nanogpt/track3-speedrun.

Each predicate reads only what the adapter surfaced (evidence.layout, evidence.observed_reward,
evidence.trajectory). Nothing here reaches into harness internals; this file must remain a plain
data-shape read against the discovered layout, so a new harness with a different on-disk tree can
be wired up by editing dataset.json alone.
"""
from __future__ import annotations

import os
import re

from bia_verifier.pipeline import Evidence, PredicateOutcome
from bia_verifier.schemas import CheckStatus

_NAN_VAL_LOSS_RE = re.compile(r"val_loss:\s*(?:nan|-?inf)\b", re.IGNORECASE)
_STEP_LINE_RE = re.compile(r"\bstep:\d+/\d+\s+val_loss:[\d.]+")
_CUDA_ERROR_MARKERS = (
    "cuda error",
    "cuda out of memory",
    "cudnn_status_execution_failed",
    "runtimeerror: cuda",
    "torch.cuda.outofmemoryerror",
    "device-side assert triggered",
)


def _read_all_logs(evidence: Evidence) -> str:
    parts: list[str] = []
    for p in evidence.layout.logs or []:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                parts.append(fh.read())
        except OSError as exc:
            parts.append(f"<<log open failed for {p}: {exc}>>")
    return "\n".join(parts)


TRAINER_RELPATH = os.path.join("environment", "train_gpt_simple.py")


def _variant_present(evidence: Evidence) -> PredicateOutcome:
    sub = evidence.layout.submission
    if not sub or not os.path.isdir(sub):
        return PredicateOutcome(
            CheckStatus.FAIL,
            "no submission directory discovered under run_dir",
            {"submission": sub, "run_dir": evidence.layout.run_dir},
        )
    trainer = os.path.join(sub, TRAINER_RELPATH)
    if not os.path.isfile(trainer):
        return PredicateOutcome(
            CheckStatus.FAIL,
            f"training script {TRAINER_RELPATH} is missing under the mounted task",
            {"looked_for": trainer, "submission": sub},
        )
    size = os.path.getsize(trainer)
    if size == 0:
        return PredicateOutcome(
            CheckStatus.FAIL,
            f"training script {TRAINER_RELPATH} is present but empty",
            {"path": trainer, "size": size},
        )
    return PredicateOutcome(
        CheckStatus.PASS,
        f"{TRAINER_RELPATH} present ({size} bytes) under the mounted task",
        {"path": trainer, "size": size},
    )


def _log_present(evidence: Evidence) -> PredicateOutcome:
    if not evidence.layout.logs:
        return PredicateOutcome(
            CheckStatus.FAIL,
            "no *.log files discovered under run dir",
            {"run_dir": evidence.layout.run_dir},
        )
    text = _read_all_logs(evidence)
    hits = _STEP_LINE_RE.findall(text)
    if not hits:
        return PredicateOutcome(
            CheckStatus.FAIL,
            "log(s) present but no `step:S/N val_loss:V` lines emitted",
            {"logs": evidence.layout.logs, "line_count": text.count("\n")},
        )
    return PredicateOutcome(
        CheckStatus.PASS,
        f"{len(hits)} val_loss checkpoint line(s) found across {len(evidence.layout.logs)} log(s)",
        {"logs": evidence.layout.logs, "checkpoint_lines": len(hits), "sample": hits[-1]},
    )


def _no_nan_or_cuda_error(evidence: Evidence) -> PredicateOutcome:
    text = _read_all_logs(evidence)
    if not text.strip():
        return PredicateOutcome(
            CheckStatus.PASS,
            "no log content to scan; absence of NaN / CUDA markers is vacuously true here "
            "(det.log_present carries the missing-log signal)",
            {"logs": evidence.layout.logs},
        )
    nan_hits = _NAN_VAL_LOSS_RE.findall(text)
    low = text.lower()
    cuda_hits = [m for m in _CUDA_ERROR_MARKERS if m in low]
    if nan_hits or cuda_hits:
        return PredicateOutcome(
            CheckStatus.FAIL,
            "log contains NaN val_loss and/or CUDA error markers",
            {"nan_val_loss_hits": len(nan_hits), "cuda_error_markers": cuda_hits},
        )
    return PredicateOutcome(
        CheckStatus.PASS,
        "no NaN val_loss / CUDA error markers detected",
        {"chars_scanned": len(text)},
    )


def _hit_target(evidence: Evidence) -> PredicateOutcome:
    r = evidence.observed_reward
    if not r:
        return PredicateOutcome(
            CheckStatus.FAIL,
            "no observed reward.json to consult for hit_target",
            {"looked_at": evidence.layout.observed_reward if evidence.layout else None},
        )
    hit = bool(r.get("hit_target"))
    step = r.get("step_to_3_28")
    final = r.get("final_val_loss")
    if hit and step is not None:
        return PredicateOutcome(
            CheckStatus.PASS,
            f"hit_target=true at step_to_3_28={step} (final_val_loss={final})",
            {"step_to_3_28": step, "final_val_loss": final, "reward": r.get("reward")},
        )
    return PredicateOutcome(
        CheckStatus.FAIL,
        f"hit_target={hit}, step_to_3_28={step}, final_val_loss={final}",
        {"observed_reward": r},
    )


def _reward_recorded(evidence: Evidence) -> PredicateOutcome:
    r = evidence.observed_reward
    if not r:
        return PredicateOutcome(
            CheckStatus.FAIL,
            "no reward.json discovered under seed workdir",
            {"looked_at": evidence.layout.observed_reward if evidence.layout else None},
        )
    required = {"reward"}
    missing = sorted(required - set(r))
    if missing:
        return PredicateOutcome(
            CheckStatus.FAIL,
            f"reward.json is missing required keys: {missing}",
            {"observed_reward": r},
        )
    return PredicateOutcome(
        CheckStatus.PASS,
        f"reward.json well-formed; reward={r.get('reward')}",
        {
            "path": evidence.layout.observed_reward if evidence.layout else None,
            "reward": r.get("reward"),
            "hit_target": r.get("hit_target"),
            "step_to_3_28": r.get("step_to_3_28"),
        },
    )


PREDICATES = {
    "det.variant_present": _variant_present,
    "det.log_present": _log_present,
    "det.no_nan_or_cuda_error": _no_nan_or_cuda_error,
    "det.hit_target": _hit_target,
    "det.reward_recorded": _reward_recorded,
}
