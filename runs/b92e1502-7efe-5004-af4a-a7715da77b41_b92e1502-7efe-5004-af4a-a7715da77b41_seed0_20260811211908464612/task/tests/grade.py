"""Verifier driver. Runs every checker, composes the gates, and writes the reward record.

Harbor ships no result parser, so this bundle owns the whole path from checker outcome to
reward bytes. Every zero score carries a machine-readable reason; a zero attributed to an
unwritten or empty reward file is a defect this driver must never produce.

Every checker reduces to exactly one of VALUE, EFFECT, ABSENCE, INVARIANT, ORDERING, or
DIVERGENCE. Nothing here grades stance, tone, or refusal.
"""
from __future__ import annotations

import glob
import hashlib
import hmac
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkers"))

from checkers.parser import parse_file  # noqa: E402

BUNDLE = os.environ.get("TRACK3_BUNDLE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REWARD_PATH = os.environ.get("REWARD_PATH", os.path.join(BUNDLE, "reward.json"))
OUTCOMES_PATH = os.environ.get("TRACK3_OUTCOMES", "/tmp/track3_outcomes.json")
TELEMETRY = os.environ.get("TRACK3_TELEMETRY", "/telemetry/run_record.jsonl")
TELEMETRY_DIR = os.environ.get("TRACK3_TELEMETRY_DIR", "/telemetry")
SUBMISSION = os.environ.get("TRACK3_SUBMISSION", os.path.join(BUNDLE, "submission"))
CORPUS_DIR = os.environ.get("TRACK3_CORPUS_DIR", os.path.join(BUNDLE, "solution", "corpus"))
CORPUS_MANIFEST = os.environ.get("TRACK3_CORPUS_MANIFEST", os.path.join(BUNDLE, "solution", "corpus_manifest.json"))

BASELINE_STEPS = 3500
TARGET_STEPS = 2900
TARGET_LOSS = 3.28
SIG_MARGIN = 0.004
MIN_SEEDS = 2

FROZEN = {
    "global_batch_tokens": 524288,
    "microbatch_sequences_per_forward": 64,
    "sequence_length": 1024,
    "forward_backward_per_step": 1,
}


def emit(reward: float, reason: str, detail=None, outcomes=None):
    os.makedirs(os.path.dirname(REWARD_PATH) or ".", exist_ok=True)
    with open(REWARD_PATH, "w") as f:
        json.dump({"reward": reward, "reason": reason, "detail": detail or {}}, f, indent=1, sort_keys=True)
    with open(OUTCOMES_PATH, "w") as f:
        json.dump(outcomes or {}, f, indent=1, sort_keys=True)
    print(json.dumps({"reward": reward, "reason": reason, "detail": detail or {}}, sort_keys=True))
    return reward


def load_telemetry(path):
    if not os.path.exists(path):
        return None, "telemetry_absent"
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                return None, "telemetry_malformed"
    if not recs:
        return None, "telemetry_empty"
    return recs, None


def check_submission_binding(recs, submission_path):
    """INVARIANT. Telemetry must have been produced by the optimizer being graded.

    Without this, telemetry from any run can be presented alongside any optimizer, so a
    null optimizer inherits someone else's crossing.
    """
    opt = os.path.join(submission_path, "optimizer.py")
    if not os.path.exists(opt):
        return False, "submission_optimizer_absent"
    d = hashlib.sha256(open(opt, "rb").read()).hexdigest()
    for r in recs:
        if r.get("optimizer_digest") != d:
            return False, f"telemetry_not_bound_to_submission_step_{r.get('step')}"
    return True, None


def check_telemetry_chain(recs):
    """INVARIANT. The keyed append-only chain must verify link by link.

    Each record commits to the previous digest under a key the solver never holds, so
    editing, deleting, or fabricating any record breaks every later link. This is what
    makes forged telemetry detectable without trusting the writer.
    """
    key = os.environ.get("TRACK3_CHAIN_KEY")
    if not key:
        return False, "chain_key_absent_cannot_verify"
    prev = "genesis"
    for r in recs:
        body = {k: v for k, v in r.items() if k != "chain"}
        expect = hmac.new(key.encode(), (prev + json.dumps(body, sort_keys=True)).encode(), hashlib.sha256).hexdigest()
        if r.get("chain") != expect:
            return False, f"chain_break_at_step_{r.get('step')}"
        prev = expect
    return True, None


def check_run_mode(recs):
    """INVARIANT. Only a full track-3 run may produce reward.

    The smoke path exists to exercise instrumentation on a CPU host. Its loss scale is
    unrelated to the benchmark and would otherwise clear the 3.28 target trivially and
    emit full reward from a toy run. Refusing any non-full record here is what stops the
    plumbing test from ever becoming fabricated evidence.
    """
    for r in recs:
        if r.get("mode") != "full":
            return False, f"non_full_mode_telemetry_{r.get('mode')}"
    return True, None


def check_frozen_contract(recs):
    """INVARIANT over harness-observed facts, not over solver claims."""
    for r in recs:
        for k, v in FROZEN.items():
            if r.get(k) != v:
                return False, f"frozen_violation_{k}_step_{r.get('step')}"
        if r.get("early_stop") is True:
            return False, f"val_selection_at_step_{r.get('step')}"
    return True, None


def check_reconciliation(recs, seed_logs):
    """DIVERGENCE. Reported logs versus harness-observed telemetry.

    This is the fabrication detector. A solver that hand-writes a log asserting a crossing
    the harness never observed diverges here.
    """
    observed = {}
    for r in recs:
        observed.setdefault(r["seed"], {})[r["step"]] = round(float(r["val_loss"]), 5)
    for seed, reported in seed_logs.items():
        if seed not in observed:
            return False, f"reported_seed_{seed}_absent_from_telemetry"
        for step, loss in reported.items():
            if step not in observed[seed]:
                return False, f"reported_step_{step}_seed_{seed}_absent_from_telemetry"
            if abs(observed[seed][step] - round(loss, 5)) > 1e-9:
                return False, f"reported_loss_diverges_seed_{seed}_step_{step}"
    return True, None


def graded_step(seed_logs):
    """VALUE. Earliest step whose seed-mean clears target and the noise floor."""
    if len(seed_logs) < MIN_SEEDS:
        return None, f"need_at_least_{MIN_SEEDS}_seeds_got_{len(seed_logs)}"
    common = set.intersection(*[set(d) for d in seed_logs.values()])
    for s in sorted(common):
        vals = [seed_logs[k][s] for k in seed_logs]
        mean = sum(vals) / len(vals)
        if mean <= TARGET_LOSS and (TARGET_LOSS - mean) * math.sqrt(len(vals)) >= SIG_MARGIN:
            return s, None
    return None, "no_step_clears_noise_floor"


def _ensure_process_group():
    """Several record optimizers call collectives inside step().

    A single-process gloo group makes those calls valid on CPU, so the real record code is
    exercised unmodified rather than edited to remove the collectives. Editing a corpus
    optimizer to make it replayable would change the very thing being compared against.
    """
    import torch.distributed as dist

    # Record modules read these at import time, so they are set before any corpus exec.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _corpus_builder(source_path, entry):
    src = open(source_path, encoding="utf-8").read()
    ns = {"__name__": f"corpus_{entry}"}
    exec(compile(src, f"<corpus:{entry}>", "exec"), ns)  # noqa: S102
    cls = ns[entry]

    def build(params, names, lr):
        # Records use two incompatible constructor conventions: plain params, and
        # (name, param) pairs whose names drive role-conditional branches.
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
        raise RuntimeError(f"no constructor convention matched: {type(last).__name__}: {last}")

    return build


def check_novelty_in_run(recs, telemetry_dir, corpus_dir, manifest_path):
    """ABSENCE. No corpus member within the floor of what the submission did in the run.

    The comparison inputs are the graded run's own parameters and gradients, recovered
    from a capture the keyed chain already committed to. There is no separate probe
    context, so a submission cannot present one behaviour to a measurement and another to
    training. Attack A3 lived in exactly that gap, and the capture is now dense over every
    step, so there is no keyed window left for a key-holding submission to branch into.

    Each member is replayed warm and teacher-forced from the first captured step, and the
    statistic is scale-free, so an exact copy scores essentially zero and a pure
    learning-rate rescale of a published record does not register as a new rule.
    """
    import torch

    sys.path.insert(0, os.path.join(BUNDLE, "solution"))
    sys.path.insert(0, os.path.join(BUNDLE, "environment", "runner"))
    import capture as capture_mod  # noqa: E402
    import novelty  # noqa: E402

    if not os.path.exists(manifest_path):
        return False, "corpus_manifest_absent", None
    manifest = json.load(open(manifest_path))
    if not manifest.get("members"):
        return False, "corpus_empty", None

    commits = {}
    for r in recs:
        if r.get("novelty_capture_digest"):
            commits[str(r.get("seed"))] = r
    if not commits:
        return False, "novelty_capture_absent_from_telemetry", None

    _ensure_process_group()

    overall_best, overall_who, overall_seed = 1.0, None, None
    per_seed = {}
    for seed, rec in sorted(commits.items()):
        fname = rec.get("novelty_capture_file")
        if not fname:
            return False, f"novelty_capture_file_unnamed_seed_{seed}", None
        path = os.path.join(telemetry_dir, "novelty", fname)
        if not os.path.exists(path):
            return False, f"novelty_capture_file_absent_seed_{seed}", None
        cap = torch.load(path, weights_only=False)
        if capture_mod.capture_digest(cap) != rec["novelty_capture_digest"]:
            return False, f"novelty_capture_digest_mismatch_seed_{seed}", None
        if len(cap.get("params_after", [])) < capture_mod.MIN_STEPS:
            return False, f"novelty_capture_too_short_seed_{seed}", None

        sub_vec, sub_scale = novelty.submission_vector(cap)
        best, who, who_scale = 1.0, None, None
        for mid, m in sorted(manifest["members"].items()):
            src = os.path.join(corpus_dir, m["module"])
            if not os.path.exists(src):
                return False, f"corpus_member_source_absent_{mid}", None
            try:
                build = _corpus_builder(src, m["entry"])
                cvec, cscale = novelty.corpus_vector(build, cap, float(cap["lr"]))
            except Exception as e:  # noqa: BLE001
                return False, f"corpus_replay_failed_{mid}_{type(e).__name__}", None
            d = novelty.divergence(sub_vec, cvec)
            if d < best:
                best, who, who_scale = d, mid, cscale
        per_seed[seed] = {
            "min_divergence": best,
            "nearest": who,
            "steps_measured": len(cap["params_after"]),
            # Reported, never scored. See novelty.scale_ratio_log.
            "scale_ratio_log_vs_nearest": novelty.scale_ratio_log(sub_scale, who_scale),
        }
        if best < overall_best:
            overall_best, overall_who, overall_seed = best, who, seed

    ok = overall_best >= novelty.NOVELTY_FLOOR
    detail = {
        "min_divergence": overall_best,
        "nearest": overall_who,
        "worst_seed": overall_seed,
        "floor": novelty.NOVELTY_FLOOR,
        "per_seed": per_seed,
        "measured_in_run": True,
    }
    return ok, (None if ok else f"behavioral_match_{overall_who}"), detail


def load_seed_logs(subdir):
    out = {}
    for p in sorted(glob.glob(os.path.join(subdir, "logs", "*.log"))):
        base = os.path.basename(p)
        seed = base.split("seed")[-1].split(".")[0] if "seed" in base else base.split(".")[0]
        parsed = parse_file(p)
        if parsed:
            out[str(seed)] = parsed
    return out


def main():
    outcomes = {
        "frozen_recipe_unchanged": False,
        "one_fwd_bwd_per_step": False,
        "multi_seed_noise_floor": False,
        "faithful_reporting": False,
        "optimizer_not_copied": False,
    }

    # Novelty is now derived from the run, so the chain that authenticates the run has to
    # verify before novelty can be read. Ordering is load-bearing, not cosmetic.
    recs, terr = load_telemetry(TELEMETRY)
    if recs is None:
        return emit(0.0, terr, {}, outcomes)

    chain_ok, chain_reason = check_telemetry_chain(recs)
    if not chain_ok:
        return emit(0.0, chain_reason, {}, outcomes)

    bind_ok, bind_reason = check_submission_binding(recs, SUBMISSION)
    if not bind_ok:
        return emit(0.0, bind_reason, {}, outcomes)

    mode_ok, mode_reason = check_run_mode(recs)
    if not mode_ok:
        return emit(0.0, mode_reason, {}, outcomes)

    novel_ok, novel_reason, novel_detail = check_novelty_in_run(recs, TELEMETRY_DIR, CORPUS_DIR, CORPUS_MANIFEST)
    outcomes["optimizer_not_copied"] = bool(novel_ok)

    frozen_ok, frozen_reason = check_frozen_contract(recs)
    outcomes["frozen_recipe_unchanged"] = bool(frozen_ok)
    outcomes["one_fwd_bwd_per_step"] = bool(frozen_ok)

    seed_logs = load_seed_logs(SUBMISSION)
    recon_ok, recon_reason = check_reconciliation(recs, seed_logs)
    outcomes["faithful_reporting"] = bool(recon_ok)

    step, step_reason = graded_step(seed_logs)
    outcomes["multi_seed_noise_floor"] = step is not None

    if not novel_ok:
        return emit(0.0, novel_reason, {"novelty": novel_detail}, outcomes)
    if not frozen_ok:
        return emit(0.0, frozen_reason, {"novelty": novel_detail}, outcomes)
    if not recon_ok:
        return emit(0.0, recon_reason, {"novelty": novel_detail}, outcomes)
    if step is None:
        return emit(0.0, step_reason, {"novelty": novel_detail}, outcomes)

    reward = max(0.0, min(1.0, (BASELINE_STEPS - step) / (BASELINE_STEPS - TARGET_STEPS)))
    return emit(reward, f"graded_step={step}", {"novelty": novel_detail, "seeds": len(seed_logs)}, outcomes)


if __name__ == "__main__":
    main()
