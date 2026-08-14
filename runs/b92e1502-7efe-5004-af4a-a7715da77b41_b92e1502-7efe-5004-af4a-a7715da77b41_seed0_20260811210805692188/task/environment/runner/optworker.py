"""Tier 3. The only process that ever executes submitted code.

The submitted optimizer runs here and nowhere else. This process holds no chain key, owns
no telemetry file, and never sees the loss, the model, or the recorder. It receives
gradients on stdin and returns updated parameter values on stdout. That is the whole
interface.

The reason for a third process, rather than only scrubbing the key out of the trainer's
environment, is that a keyed chain written by a process containing attacker code is not
evidence of anything. The predecessor exec'd the submission inside the process that wrote
the chain, so a submission could both read the key and rewrite the record. Moving the
submission behind a pipe means the bytes that get signed were produced by code the
submission cannot reach, which is what the chain claimed all along.

Frames are length-prefixed torch payloads. Parameters are created once at init and
persist, because a torch optimizer keys its state by parameter object identity and a fresh
tensor each step would silently reset momentum.
"""
from __future__ import annotations

import io
import os
import struct
import sys

import torch


def _read_frame(stream):
    head = stream.read(8)
    if not head or len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    buf = stream.read(n)
    if len(buf) < n:
        return None
    return torch.load(io.BytesIO(buf), weights_only=False)


def _write_frame(stream, obj):
    bio = io.BytesIO()
    torch.save(obj, bio)
    payload = bio.getvalue()
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)
    stream.flush()


def ensure_process_group():
    """Record-derived optimizers call collectives inside step().

    The real track-3 trainer runs under torchrun, so a process group always exists there.
    A single-process gloo group makes those calls valid here, so record code is exercised
    unmodified rather than edited to remove the collectives.
    """
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # Overrides rather than defaults: an inherited MASTER_PORT would collide with
        # whichever process exported it, and the worker must own its own rendezvous.
        os.environ["MASTER_PORT"] = os.environ.get("TRACK3_GLOO_PORT", "29518")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def load_submission(path):
    ns = {"__name__": "submitted_optimizer"}
    exec(compile(open(path).read(), "<submission>", "exec"), ns)  # noqa: S102
    if "build_optimizer" not in ns:
        raise SystemExit("submission/optimizer.py must define build_optimizer(params, lr=...)")
    return ns["build_optimizer"]


def build_with_conventions(build, params, names, lr):
    """Records and submissions use two constructor conventions.

    Plain parameters, and (name, parameter) pairs whose names drive role-conditional
    branches. Trying both here means a record optimizer can be replayed as a submission
    without editing it, which matters because the negative controls in the separation test
    are exactly unedited record optimizers.
    """
    attempts = [
        lambda: build(list(params), lr=lr),
        lambda: build(list(zip(names, params)), lr=lr),
        lambda: build(list(params)),
        lambda: build(list(zip(names, params))),
    ]
    last = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"no constructor convention matched: {type(last).__name__}: {last}")


def main():
    if os.environ.get("TRACK3_CHAIN_KEY"):
        raise SystemExit("optworker refuses to run with a chain key in its environment")

    # Framing moves off stdout before any submitted code can run. A submission, or a record
    # optimizer lifted verbatim, may print at construction time, and those bytes would
    # otherwise be read as a frame length prefix and desynchronize the protocol. One record
    # in the corpus does exactly this, and it presented as a MemoryError in the trainer.
    # Anything written to stdout from here on lands on stderr instead.
    framing = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)

    inp = sys.stdin.buffer
    out = framing
    params = None
    opt = None

    while True:
        msg = _read_frame(inp)
        if msg is None or msg.get("cmd") == "done":
            return
        if msg["cmd"] == "init":
            ensure_process_group()
            params = [torch.nn.Parameter(t.clone(), requires_grad=False) for t in msg["params"]]
            build = load_submission(msg["submission"])
            try:
                opt = build_with_conventions(build, params, list(msg["names"]), float(msg["lr"]))
            except Exception as e:  # noqa: BLE001
                _write_frame(out, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return
            _write_frame(out, {"ok": True})
        elif msg["cmd"] == "step":
            for p, g in zip(params, msg["grads"]):
                p.grad = g.clone()
            try:
                opt.step()
            except Exception as e:  # noqa: BLE001
                _write_frame(out, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return
            for p in params:
                p.grad = None
            _write_frame(out, {"ok": True, "params": [p.data.detach().clone() for p in params]})


if __name__ == "__main__":
    main()
