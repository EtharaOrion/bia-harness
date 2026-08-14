"""Authoring-time corpus builder. Runs inside the pinned verifier image.

Extracts each declared corpus member's optimizer from the real record bytes, proves it can
actually be replayed, and stages the lifted source privately under solution/corpus/.

The previous bundle precomputed a fixed signature vector per member against a standalone
synthetic probe. Novelty is now measured inside the graded training run, so a member's
comparison vector depends on that run's own parameters and gradients and cannot be
computed ahead of time. The verifier therefore needs the member's update rule at grading
time, not a frozen vector.

The lifted sources stay inside solution/, which never crosses the agent-visible boundary,
and they carry the upstream MIT licence. A corpus member that cannot be extracted is
recorded as a build failure with its reason. It is never silently dropped, because a
silently missing corpus member is a hole in the novelty gate that nobody would see.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import extract_symbol  # noqa: E402

def ensure_process_group():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("CORPUS_GLOO_PORT", "29788")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def replay_check(lifted_path, entry):
    """Prove the member can actually be replayed before admitting it to the corpus.

    A member that imports, constructs, or steps unsuccessfully at grading time makes the
    novelty checker return failure for EVERY submission, which is an instrument closed on
    every solution rather than closed on a wrong one. Invariant 17 calls that inert. The
    check therefore runs here, at build time, and a member that cannot be replayed is
    recorded as a named failure and left out of the corpus instead of poisoning the gate.
    """
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import novelty

    ensure_process_group()
    ns = {"__name__": f"corpus_probe_{entry}"}
    exec(compile(open(lifted_path, encoding="utf-8").read(), f"<{entry}>", "exec"), ns)  # noqa: S102
    cls = ns[entry]

    g = torch.Generator().manual_seed(0)
    names = ["probe.0.weight", "probe.1.weight"]
    before = [torch.empty(8, 8).uniform_(-0.1, 0.1, generator=g) for _ in names]
    cap = {
        "version": "probe", "scope": "probe", "lr": 0.02, "param_names": names,
        "params_before": before,
        "grads": [[torch.empty(8, 8).uniform_(-1, 1, generator=g) for _ in names] for _ in range(3)],
        "params_after": [[t.clone() for t in before] for _ in range(3)],
    }

    def build(params, nms, lr):
        attempts = [
            lambda: cls(list(params), lr=lr),
            lambda: cls(list(zip(nms, params)), lr=lr),
            lambda: cls(list(params)),
            lambda: cls(list(zip(nms, params))),
        ]
        last = None
        for fn in attempts:
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"no constructor convention matched: {type(last).__name__}: {last}")

    v1, _ = novelty.corpus_vector(build, cap, 0.02)
    v2, _ = novelty.corpus_vector(build, cap, 0.02)
    # A member that does not replay to itself sets a noise floor under every comparison it
    # takes part in, so the operating floor has to clear it. PSGD draws random probe vectors
    # inside its preconditioner update and is genuinely stochastic. Recording this here is
    # what makes the copy ceiling explainable rather than mysterious.
    return novelty.divergence(v1, v2)


RECORDS_ROOT = os.environ.get("RECORDS_ROOT", "/records")
GROUNDING = os.environ.get("GROUNDING", "/work/solution/grounding.yaml")
OUTDIR = os.environ.get("CORPUS_DIR", "/work/solution/corpus")
MANIFEST = os.environ.get("CORPUS_MANIFEST", "/work/solution/corpus_manifest.json")


def load_grounding(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def main():
    g = load_grounding(GROUNDING)
    members = g["novelty_gate"]["corpus_members"]
    os.makedirs(OUTDIR, exist_ok=True)

    out = {"members": {}, "failures": {}}
    for m in members:
        mid = m["id"]
        rel = m["source"].split("harness/records/track_3_optimization/", 1)[-1]
        path = os.path.join(RECORDS_ROOT, rel)
        try:
            src = open(path, encoding="utf-8", errors="replace").read()
            lifted = extract_symbol(src, m["entry"])
            dest = os.path.join(OUTDIR, f"{mid}.py")
            with open(dest, "w") as f:
                f.write(lifted)
            self_div = replay_check(dest, m["entry"])
            out["members"][mid] = {
                "replay_self_divergence_probe": round(self_div, 6),
                "replay_deterministic": self_div == 0.0,
                "record": m["record"],
                "entry": m["entry"],
                "source_rel": rel,
                "extracted_sha256": hashlib.sha256(lifted.encode()).hexdigest(),
                "module": f"{mid}.py",
            }
            print(f"OK   {mid:24} entry={m['entry']:12} sha={out['members'][mid]['extracted_sha256'][:16]} selfdiv={self_div:.6f}")
        except Exception as e:  # noqa: BLE001
            out["failures"][mid] = f"{type(e).__name__}: {e}"
            dead = os.path.join(OUTDIR, f"{mid}.py")
            if os.path.exists(dead):
                os.remove(dead)
            print(f"FAIL {mid:24} {type(e).__name__}: {e}")

    with open(MANIFEST, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"\nbuilt={len(out['members'])} failed={len(out['failures'])}")


if __name__ == "__main__":
    main()
