"""Tier 1 supervisor. Holds the chain key and signs the run. Executes no submitted code.

The reference grader this bundle descends from parsed seed logs that the SOLVER produced,
which means it graded the solver's own claims about itself and had no mechanism to tell a
real training log from a hand-written one. This runner closes that hole: the harness
observes the frozen-contract facts directly and commits them to a keyed append-only chain.

WHAT CHANGED AND WHY. The predecessor kept the key, the chain writer, and a bare exec() of
the submission in ONE process. A submission could therefore read TRACK3_CHAIN_KEY straight
out of the environment, and the module comment claiming otherwise was false. Worse than
the novelty branch that exploited it, a submission holding the key could forge the chain
outright, so the chain authenticated nothing. Protecting the secret harder does not fix
that. The roles are now split across three processes:

  tier 1, this file    holds the key, signs raw records into the chain, runs no user code
  tier 2, trainer.py   owns the model, the loss, and the dense capture, holds no key
  tier 3, optworker.py executes the submitted optimizer, holds no key and owns no record

The key never enters tier 2 or tier 3. Because a same-uid child can read a parent's
environment block and memory through /proc, scrubbing the child environment is not enough
on its own: this process re-executes itself with the key moved out of the environment onto
an inherited pipe, and sets PR_SET_DUMPABLE to 0 so its /proc entries stop being readable
by a same-uid process. Tier 2 asserts the key is unreachable before it trains.

The measurement no longer depends on any of that holding. Capture is dense over every step,
so there is no keyed window to predict and nothing to branch into even if the key leaked.
Secrecy is defence in depth here, not the load-bearing member.

Two modes:
  full   the real track-3 trainer. Requires CUDA. Never reachable on a CPU-only host.
  smoke  a micro training loop on synthetic data, CPU only. It exists to exercise the
         instrumentation, capture, and grading path end to end. It CANNOT reach the
         target loss and is never a substitute for a real run or evidence about reward.
"""
from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys

TELEMETRY_DIR = os.environ.get("TRACK3_TELEMETRY_DIR", "/telemetry")
FROZEN = {
    "global_batch_tokens": 524288,
    "microbatch_sequences_per_forward": 64,
    "sequence_length": 1024,
    "forward_backward_per_step": 1,
}
KEY_ENV = "TRACK3_CHAIN_KEY"
KEY_FD_FLAG = "--chain-key-fd"


def set_non_dumpable():
    """Stop a same-uid process reading this one's /proc/<pid>/environ, maps, or mem.

    PR_SET_DUMPABLE is 4 and takes effect independently of kernel.yama.ptrace_scope, so it
    holds even where ptrace of a non-descendant is otherwise permitted. Best effort: a
    platform without prctl loses this layer, and the dense capture is what makes that
    survivable rather than fatal.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(4, 0, 0, 0, 0)
    except OSError:
        pass


def reexec_without_key_in_environ():
    """Move the key from the environment onto an inherited pipe, then re-exec.

    Python-level mutation of os.environ does not rewrite /proc/self/environ, so popping the
    variable would leave it readable by any same-uid process for the lifetime of this one.
    Only a fresh exec with a scrubbed environment actually removes it.
    """
    key = os.environ[KEY_ENV].encode()
    r, w = os.pipe()
    os.set_inheritable(r, True)
    os.write(w, key)
    os.close(w)
    env = {k: v for k, v in os.environ.items() if k != KEY_ENV}
    argv = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + [KEY_FD_FLAG, str(r)]
    os.execve(sys.executable, argv, env)


def read_key_from_fd(argv):
    i = argv.index(KEY_FD_FLAG)
    fd = int(argv[i + 1])
    with os.fdopen(fd, "rb") as f:
        key = f.read()
    del argv[i:i + 2]
    if not key:
        raise SystemExit("chain key pipe was empty")
    return key


class Chain:
    """HMAC-keyed append-only chain, written by the supervisor over raw tier-2 records.

    An unkeyed digest would be forgeable, because the recipe is agent-visible. The key is
    what makes the chain unforgeable rather than merely tamper-evident, and keeping it in a
    process that runs no submitted code is what makes that property real.
    """

    def __init__(self, path, key: bytes, optimizer_digest: str):
        self.path = path
        self.prev = "genesis"
        self.key = key
        self.optimizer_digest = optimizer_digest
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()

    def append(self, record):
        record = dict(record)
        record["optimizer_digest"] = self.optimizer_digest
        body = json.dumps(record, sort_keys=True)
        digest = hmac.new(self.key, (self.prev + body).encode(), hashlib.sha256).hexdigest()
        record = dict(record)
        record["chain"] = digest
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self.prev = digest


def main():
    argv = list(sys.argv)
    if KEY_FD_FLAG not in argv:
        if not os.environ.get(KEY_ENV):
            raise SystemExit(
                f"{KEY_ENV} absent. The harness must inject it; the agent never holds it."
            )
        reexec_without_key_in_environ()
        return

    key = read_key_from_fd(argv)
    set_non_dumpable()
    sys.argv = argv

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--max-steps", type=int, default=3500)
    ap.add_argument("--mode", choices=["full", "smoke"], default="full")
    ap.add_argument("--submission", default="submission/optimizer.py")
    ap.add_argument("--logdir", default="submission/logs")
    args = ap.parse_args(argv[1:])

    runner_dir = os.path.dirname(os.path.abspath(__file__))
    rawdir = os.path.join(TELEMETRY_DIR, "raw")
    if os.path.isdir(rawdir):
        shutil.rmtree(rawdir)
    os.makedirs(rawdir, exist_ok=True)

    child_env = {k: v for k, v in os.environ.items() if k != KEY_ENV}
    rc = subprocess.call(
        [sys.executable, os.path.join(runner_dir, "trainer.py"),
         "--seeds", args.seeds, "--max-steps", str(args.max_steps), "--mode", args.mode,
         "--submission", args.submission, "--logdir", args.logdir, "--rawdir", rawdir],
        env=child_env, close_fds=True,
    )
    if rc != 0:
        raise SystemExit(f"training tier exited {rc}")

    opt_digest = hashlib.sha256(open(args.submission, "rb").read()).hexdigest()
    chain = Chain(os.path.join(TELEMETRY_DIR, "run_record.jsonl"), key, opt_digest)

    novelty_out = os.path.join(TELEMETRY_DIR, "novelty")
    if os.path.isdir(novelty_out):
        shutil.rmtree(novelty_out)
    shutil.copytree(os.path.join(rawdir, "novelty"), novelty_out)

    with open(os.path.join(rawdir, "raw_record.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                chain.append({**json.loads(line), **FROZEN})

    if args.mode == "smoke":
        print("SMOKE COMPLETE. This exercises instrumentation only and reaches no target loss.")


if __name__ == "__main__":
    main()
