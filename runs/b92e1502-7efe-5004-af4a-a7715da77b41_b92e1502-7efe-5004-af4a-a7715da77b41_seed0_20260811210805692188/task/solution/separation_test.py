"""Measure what the novelty gate actually decides, by executing it.

This is the both-halves demonstration for the novelty gate. Every number below comes from
running a real training loop through the real capture path and scoring the real capture
with the real private statistic. Nothing is asserted.

The predecessor version of this file had never run against the gate it claimed to test. It
imported a probe module that had been deleted, read a corpus signature file that had been
deleted, and defaulted its floor to a value the gate no longer used. A both-halves proof
that cannot execute is not weaker evidence than a real one, it is no evidence, and it is
exactly the inert instrument invariant 17 caps at BLOCK.

Three families of case, each a real execution:

  reference        the candidate oracle. Must CLEAR the floor.
  verbatim copies  each corpus member resubmitted unchanged. Must be BLOCKED.
  paraphrase       a corpus member with identifiers renamed and statements reordered.
                   Must be BLOCKED. Phase R staged evidence that source-text similarity
                   detection collapses under exactly this transformation, so a behavioural
                   gate has to catch what a textual one cannot.
  rescales         a published record with nothing changed but its learning rate. Must be
                   BLOCKED. instruction.md promises that a hyperparameter change does not
                   clear this gate. The predecessor statistic broke that promise: a four
                   percent bump on the baseline cleared, and a five percent bump outscored
                   the reference optimizer.

The sweep is over independent seeds. There is no window sweep any more because there is no
window: capture is dense over every step, so a key realization no longer selects anything.
The predecessor's 841 key realizations were 841 different answers to a question this gate
no longer asks, and the spread across them was the cold-start artefact this rebuild removes.

The margin reported at the end is the real acceptance criterion: the lowest score the
reference reaches anywhere in the sweep, minus the highest score any planted copy reaches
anywhere in the sweep. A negative margin means no floor exists that both admits the
reference and blocks copies, and this script exits non-zero when that happens.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
RUNNER = os.path.join(BUNDLE, "environment", "runner")
CORPUS_DIR = os.environ.get("TRACK3_CORPUS_DIR", os.path.join(HERE, "corpus"))
MANIFEST = os.environ.get("TRACK3_CORPUS_MANIFEST", os.path.join(HERE, "corpus_manifest.json"))
WORK = os.environ.get("SEPARATION_WORK", "/tmp/separation")
SHARD = os.environ.get("SEPARATION_SHARD")
MERGE = os.environ.get("SEPARATION_MERGE")
PARTS = os.path.join(HERE, "fixtures", "parts")
SEEDS = [int(s) for s in os.environ.get("SEPARATION_SEEDS", "0,1,7,13,29").split(",")]
STEPS = int(os.environ.get("SEPARATION_STEPS", "96"))
LR_FACTORS = [1.02, 1.03, 1.04, 1.05, 1.10, 1.20, 1.35, 1.50]

sys.path.insert(0, HERE)
sys.path.insert(0, RUNNER)
import novelty  # noqa: E402


def shim_passthrough(member_path, entry):
    return (
        f"import runpy\n"
        f"_ns = runpy.run_path({member_path!r})\n"
        f"build_optimizer = _ns[{entry!r}]\n"
    )


def shim_rescaled(member_path, entry, factor):
    """A published record with its learning rate multiplied and nothing else touched.

    This is the submission a solver writes when it wants to clear a novelty gate without
    deriving anything. It is a hyperparameter change, not a new update rule.
    """
    return (
        f"import runpy\n"
        f"_ns = runpy.run_path({member_path!r})\n"
        f"_C = _ns[{entry!r}]\n"
        f"_F = {factor!r}\n"
        f"def build_optimizer(params, lr=0.02, **kw):\n"
        f"    params = list(params)\n"
        f"    for attempt in (0, 1):\n"
        f"        try:\n"
        f"            arg = params if attempt == 0 else list(params)\n"
        f"            return _C(arg, lr=lr * _F, **kw)\n"
        f"        except Exception:\n"
        f"            continue\n"
        f"    return _C(params, lr=lr * _F)\n"
    )


def make_capture(name, source, seed):
    """Run the real training tier over this candidate and return its dense capture."""
    d = os.path.join(WORK, name, str(seed))
    os.makedirs(d, exist_ok=True)
    sub = os.path.join(d, "optimizer.py")
    with open(sub, "w") as f:
        f.write(source)
    env = {k: v for k, v in os.environ.items() if k != "TRACK3_CHAIN_KEY"}
    r = subprocess.run(
        [sys.executable, os.path.join(RUNNER, "trainer.py"), "--mode", "smoke",
         "--seeds", str(seed), "--max-steps", str(STEPS), "--submission", sub,
         "--logdir", os.path.join(d, "logs"), "--rawdir", d],
        env=env, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{name} seed {seed} training failed: {r.stderr[-600:]}")
    return torch.load(os.path.join(d, "novelty", f"seed{seed}.pt"), weights_only=False)


def corpus_builders():
    manifest = json.load(open(MANIFEST))
    out = {}
    for mid, m in sorted(manifest["members"].items()):
        path = os.path.join(CORPUS_DIR, m["module"])
        ns = {"__name__": f"corpus_{mid}"}
        exec(compile(open(path).read(), f"<corpus:{mid}>", "exec"), ns)  # noqa: S102
        cls = ns[m["entry"]]

        def build(params, names, lr, cls=cls):
            attempts = [
                lambda: cls(list(params), lr=lr),
                lambda: cls(list(zip(names, params)), lr=lr),
                lambda: cls(list(params)),
                lambda: cls(list(zip(names, params))),
            ]
            last = None
            for fn in attempts:
                try:
                    return fn()
                except Exception as e:  # noqa: BLE001
                    last = e
            raise RuntimeError(f"no constructor convention matched: {last}")

        out[mid] = (build, m)
    return out


def score(cap, builders):
    sub_vec, sub_scale = novelty.submission_vector(cap)
    best, who, who_scale = 1.0, None, None
    per_member = {}
    for mid, (build, _m) in builders.items():
        cvec, cscale = novelty.corpus_vector(build, cap, float(cap["lr"]))
        d = novelty.divergence(sub_vec, cvec)
        per_member[mid] = round(d, 5)
        if d < best:
            best, who, who_scale = d, mid, cscale
    return best, who, per_member, novelty.scale_ratio_log(sub_scale, who_scale)


def ensure_process_group():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def build_cases():
    manifest = json.load(open(MANIFEST))
    cases = [("reference", open(os.path.join(HERE, "reference_optimizer.py")).read(), "clear")]
    for mid, m in sorted(manifest["members"].items()):
        path = os.path.join(CORPUS_DIR, m["module"])
        cases.append((f"verbatim_{mid}", shim_passthrough(path, m["entry"]), "block"))
    cases.append(("paraphrased_baseline",
                  open(os.path.join(HERE, "fixtures", "paraphrased_copy.py")).read(), "block"))
    base = manifest["members"]["baseline_muon"]
    for f in LR_FACTORS:
        cases.append((f"lrscale_baseline_x{f}",
                      shim_rescaled(os.path.join(CORPUS_DIR, base["module"]), base["entry"], f),
                      "block"))
    r46 = manifest["members"]["rec46_cwd_sota"]
    for f in (1.05, 1.20, 1.50):
        cases.append((f"lrscale_rec46_x{f}",
                      shim_rescaled(os.path.join(CORPUS_DIR, r46["module"]), r46["entry"], f),
                      "block"))
    return cases


def run_shard(index, total):
    ensure_process_group()
    builders = corpus_builders()
    os.makedirs(PARTS, exist_ok=True)
    results = {}
    for i, (name, source, expect) in enumerate(build_cases()):
        if i % total != index:
            continue
        per_seed = {}
        for seed in SEEDS:
            cap = make_capture(name, source, seed)
            best, who, per_member, srl = score(cap, builders)
            per_seed[str(seed)] = {"min_divergence": round(best, 5), "nearest": who,
                                   "scale_ratio_log_vs_nearest": srl,
                                   "per_member": per_member}
        vals = [v["min_divergence"] for v in per_seed.values()]
        results[name] = {"expect": expect, "per_seed": per_seed,
                         "min_over_seeds": min(vals), "max_over_seeds": max(vals)}
        print(f"{name:32} expect={expect:5} min={min(vals):.5f} max={max(vals):.5f} "
              f"nearest={per_seed[str(SEEDS[0])]['nearest']}", flush=True)
    with open(os.path.join(PARTS, f"part_{index}.json"), "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)
    return 0


def main():
    if SHARD:
        i, n = SHARD.split("/")
        return run_shard(int(i), int(n))

    if MERGE:
        results = {}
        for p in sorted(os.listdir(PARTS)):
            results.update(json.load(open(os.path.join(PARTS, p))))
        expected = {c[0] for c in build_cases()}
        missing = sorted(expected - set(results))
        if missing:
            print(f"MERGE INCOMPLETE, missing cases: {missing}")
            return 1
    else:
        ensure_process_group()
        builders = corpus_builders()
        results = {}
        for name, source, expect in build_cases():
            per_seed = {}
            for seed in SEEDS:
                cap = make_capture(name, source, seed)
                best, who, per_member, srl = score(cap, builders)
                per_seed[str(seed)] = {"min_divergence": round(best, 5), "nearest": who,
                                       "scale_ratio_log_vs_nearest": srl,
                                       "per_member": per_member}
            vals = [v["min_divergence"] for v in per_seed.values()]
            results[name] = {"expect": expect, "per_seed": per_seed,
                             "min_over_seeds": min(vals), "max_over_seeds": max(vals)}
            print(f"{name:32} expect={expect:5} min={min(vals):.5f} max={max(vals):.5f}",
                  flush=True)

    blocked = {k: v for k, v in results.items() if v["expect"] == "block"}
    copy_ceiling = max(v["max_over_seeds"] for v in blocked.values())
    ceiling_who = max(blocked.items(), key=lambda kv: kv[1]["max_over_seeds"])[0]
    reference_floor = results["reference"]["min_over_seeds"]
    margin = reference_floor - copy_ceiling

    floor = novelty.NOVELTY_FLOOR
    ref_clears = reference_floor >= floor
    all_blocked = all(v["max_over_seeds"] < floor for v in blocked.values())
    leakers = [k for k, v in blocked.items() if v["max_over_seeds"] >= floor]

    print("\nSEPARATION SUMMARY")
    print(f"  seeds swept                        : {SEEDS}")
    print(f"  steps per run                      : {STEPS}")
    print(f"  corpus members                     : {len(json.load(open(MANIFEST))['members'])}")
    print(f"  copy ceiling (worst planted copy)  : {copy_ceiling:.5f}  [{ceiling_who}]")
    print(f"  reference floor (worst seed)       : {reference_floor:.5f}")
    print(f"  margin                             : {margin:+.5f}")
    print(f"  operating floor NOVELTY_FLOOR      : {floor}")

    print("\nBOTH-HALVES CHECK")
    print(f"  clean fixture (reference) clears floor : {ref_clears}")
    print(f"  every planted copy blocked             : {all_blocked}"
          + (f"  leakers={leakers}" if leakers else ""))
    print(f"  margin strictly positive               : {margin > 0}")

    out = {
        "score_version": novelty.SCORE_VERSION,
        "floor": floor,
        "seeds": SEEDS,
        "steps": STEPS,
        "copy_ceiling": copy_ceiling,
        "copy_ceiling_case": ceiling_who,
        "reference_floor": reference_floor,
        "margin": margin,
        "reference_clears": ref_clears,
        "all_copies_blocked": all_blocked,
        "leakers": leakers,
        "results": results,
    }
    os.makedirs(os.path.join(HERE, "fixtures"), exist_ok=True)
    with open(os.path.join(HERE, "fixtures", "separation_result.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)

    # Shard scratch is intermediate, not evidence. separation_result.json is the artifact
    # that ships; leaving 31 part files in the bundle would put sweep plumbing into the
    # frozen bytes and the bundle identity derived from them.
    if MERGE and os.path.isdir(PARTS):
        for p in os.listdir(PARTS):
            os.remove(os.path.join(PARTS, p))
        os.rmdir(PARTS)

    if not (ref_clears and all_blocked and margin > 0):
        print("\nSEPARATION FAILED")
        return 1
    print("\nSEPARATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
