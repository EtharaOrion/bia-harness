"""Tier 2. Owns the model, the loss, and the dense capture. Holds no chain key.

This process runs the training loop and records what happened. It never executes submitted
code: the submitted optimizer lives in a tier-3 worker behind a pipe, and this process only
sends it gradients and receives updated parameter values. It writes RAW, unsigned records.
The tier-1 supervisor, which holds the key and likewise contains no submitted code, is what
turns them into the keyed chain.

The three-tier split exists because the predecessor collapsed all three roles into one
process. A single process that holds the key, writes the chain, and exec()s the submission
gives the submission both the key and the pen. Splitting them means the recorder observes
the submission from outside rather than hosting it.

This process asserts at startup that no chain key is reachable from it, including through
the parent's /proc entry, and refuses to run if one is. A misconfigured harness that leaks
the key must fail loudly rather than produce a run whose telemetry looks fine.
"""
from __future__ import annotations

import argparse
import ctypes
import io
import json
import os
import struct
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture  # noqa: E402

SMOKE_STEPS = 96


def set_non_dumpable():
    """Close this tier's /proc entries to the submission worker it spawns.

    The worker is a descendant, so ptrace_scope alone does not stop it reading
    /proc/<trainer>/mem. This tier holds no key, but it does hold the recorder buffers and
    the capture, and a submission that can read the recorder can plan against it.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(4, 0, 0, 0, 0)
    except OSError:
        pass


def assert_no_key_reachable():
    """Live control. The training tier must not be able to reach the chain key.

    Scrubbing the child environment alone is not sufficient, because a same-uid child can
    read the parent's environment block out of /proc. This is the leak class that defeated
    the predecessor, so it is asserted rather than assumed.
    """
    if os.environ.get("TRACK3_CHAIN_KEY"):
        raise SystemExit("KEY LEAK: TRACK3_CHAIN_KEY is present in the trainer environment")
    try:
        with open(f"/proc/{os.getppid()}/environ", "rb") as f:
            parent_env = f.read()
        if b"TRACK3_CHAIN_KEY=" in parent_env:
            raise SystemExit("KEY LEAK: TRACK3_CHAIN_KEY readable from the parent /proc environ")
    except (FileNotFoundError, PermissionError):
        pass


class Worker:
    """Pipe to the tier-3 process that executes the submitted optimizer."""

    def __init__(self, runner_dir, submission, gloo_port):
        env = {k: v for k, v in os.environ.items() if k != "TRACK3_CHAIN_KEY"}
        env["TRACK3_GLOO_PORT"] = str(gloo_port)
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(runner_dir, "optworker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, close_fds=True,
        )
        self.submission = submission

    def _send(self, obj):
        bio = io.BytesIO()
        torch.save(obj, bio)
        payload = bio.getvalue()
        self.proc.stdin.write(struct.pack("<Q", len(payload)))
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def _recv(self):
        head = self.proc.stdout.read(8)
        if not head or len(head) < 8:
            raise SystemExit("submission worker died without a reply")
        (n,) = struct.unpack("<Q", head)
        buf = self.proc.stdout.read(n)
        return torch.load(io.BytesIO(buf), weights_only=False)

    def init(self, params, names, lr):
        self._send({"cmd": "init", "params": params, "names": names, "lr": lr,
                    "submission": self.submission})
        r = self._recv()
        if not r.get("ok"):
            raise SystemExit(f"submission failed to build an optimizer: {r.get('error')}")

    def step(self, grads):
        self._send({"cmd": "step", "grads": grads})
        r = self._recv()
        if not r.get("ok"):
            raise SystemExit(f"submission optimizer step failed: {r.get('error')}")
        return r["params"]

    def close(self):
        try:
            self._send({"cmd": "done"})
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def smoke_model(seed: int):
    """Per-seed initialization.

    The predecessor seeded the loop and then called this function, which re-seeded to a
    constant and overwrote it, so every seed produced a bit-identical run. A two-seed
    noise floor built on one repeated sample measures nothing, and the novelty floor was
    calibrated on top of it. The seed is now a parameter and there is no constant reseed.
    """
    g = torch.Generator().manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64, bias=False),
    )
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.empty_like(p).uniform_(-0.125, 0.125, generator=g))
    return model, g


def run_smoke(seeds, steps, submission, logdir, rawdir, runner_dir):
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(rawdir, exist_ok=True)
    os.makedirs(os.path.join(rawdir, "novelty"), exist_ok=True)
    raw_path = os.path.join(rawdir, "raw_record.jsonl")
    open(raw_path, "w").close()

    def emit(rec):
        with open(raw_path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    for si, seed in enumerate(seeds):
        torch.manual_seed(int(seed))
        model, g = smoke_model(int(seed))
        named = [(n, p) for n, p in model.named_parameters() if p.ndim == 2]
        names = [n for n, _ in named]
        params = [p for _, p in named]
        for p in params:
            p.requires_grad_(False)
        lr = 0.02

        idx = capture.capturable_indices(params)
        cap = {
            "version": capture.CAPTURE_VERSION,
            "scope": capture.CAPTURE_SCOPE,
            "lr": lr,
            "param_names": [names[i] for i in idx],
            "params_before": [params[i].data.detach().clone().cpu().to(torch.float32) for i in idx],
            "grads": [],
            "params_after": [],
        }

        worker = Worker(runner_dir, submission, 20000 + (os.getpid() * 7 + si * 13) % 20000)
        worker.init([p.data.detach().clone() for p in params], names, lr)

        # Per-seed data. Shared data across seeds would leave the seed affecting only the
        # initialization, which is a weaker form of the collapse this replaces.
        x = torch.randn(32, 64, generator=g)
        y = torch.randn(32, 64, generator=g)

        logpath = os.path.join(logdir, f"smoke_seed{seed}.log")
        with open(logpath, "w") as lf:
            for step in range(1, steps + 1):
                for p in params:
                    p.requires_grad_(True)
                out = model(x)
                loss = torch.nn.functional.mse_loss(out, y)
                grads = torch.autograd.grad(loss, params)
                for p in params:
                    p.requires_grad_(False)

                updated = worker.step([grads[i].detach().clone() for i in range(len(params))])
                with torch.no_grad():
                    for p, u in zip(params, updated):
                        p.data.copy_(u)

                cap["grads"].append([grads[i].detach().clone().cpu().to(torch.float32) for i in idx])
                cap["params_after"].append(
                    [params[i].data.detach().clone().cpu().to(torch.float32) for i in idx]
                )

                val = float(loss.detach())
                lf.write(f"step:{step}/{steps} val_loss:{val:.5f} train_time:0.000s\n")
                emit({"seed": str(seed), "step": step, "val_loss": round(val, 5), "mode": "smoke",
                      "early_stop": False})

        worker.close()
        path = os.path.join(rawdir, "novelty", f"seed{seed}.pt")
        torch.save(cap, path)
        emit({"seed": str(seed), "step": steps + 1, "val_loss": round(val, 5), "mode": "smoke",
              "early_stop": False,
              "novelty_capture_digest": capture.capture_digest(cap),
              "novelty_capture_file": os.path.basename(path),
              "novelty_capture_steps": len(cap["params_after"]),
              "novelty_spec_digest": capture.spec_digest()})


def main():
    assert_no_key_reachable()
    set_non_dumpable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--max-steps", type=int, default=3500)
    ap.add_argument("--mode", choices=["full", "smoke"], default="full")
    ap.add_argument("--submission", default="submission/optimizer.py")
    ap.add_argument("--logdir", default="submission/logs")
    ap.add_argument("--rawdir", required=True)
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    runner_dir = os.path.dirname(os.path.abspath(__file__))

    if args.mode == "smoke":
        run_smoke(seeds, min(args.max_steps, SMOKE_STEPS), os.path.abspath(args.submission),
                  args.logdir, args.rawdir, runner_dir)
        return

    if not torch.cuda.is_available():
        raise SystemExit(
            "full mode requires CUDA. No accelerator is present on this host, so the real "
            "track-3 run cannot be executed here. This is a hard stop, not a fallback: "
            "silently degrading to smoke and reporting a reward would fabricate evidence."
        )
    raise SystemExit(
        "full mode wiring to the frozen trainer is not built in this authoring pass. "
        "See gap-runner-full-mode-unbuilt in EDICT.md."
    )


if __name__ == "__main__":
    main()
