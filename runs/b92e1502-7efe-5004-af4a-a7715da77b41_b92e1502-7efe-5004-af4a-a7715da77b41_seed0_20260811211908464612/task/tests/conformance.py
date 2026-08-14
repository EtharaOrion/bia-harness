"""Conformance suite. Every required check proves BOTH halves under frozen fixtures.

Invariant 17 is explicit that a suite proving only the rejecting half cannot tell a
working instrument from one wired to reject everything. So this suite carries a clean
fixture that must be accepted at full reward alongside every planted defect that must be
rejected, and it names the exact machine-readable reason each rejection must produce.

The novelty captures inside every fixture are REAL: the suite runs the actual runner over
the actual candidate optimizers to produce them. The loss curve is planted, which is what
makes this a test of the checkers rather than a claim about any optimizer's performance.
No fixture here is evidence that the reference solution reaches the target.

Mutation self-tests run under TRACK3_MUTATION and exist so the suite can demonstrate it is
not vacuous: a grader wired to accept everything, to reject everything, or with the
novelty floor removed must each be caught by a different subset of cases.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys

BUNDLE = os.environ.get("TRACK3_BUNDLE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKDIR = os.environ.get("TRACK3_CONFORMANCE_DIR", "/tmp/track3_conformance")
KEY = "conformance-fixture-key"
CURVE = {2800: 3.3050, 2850: 3.2700, 2900: 3.2600, 3000: 3.2550}
FROZEN = {"global_batch_tokens": 524288, "microbatch_sequences_per_forward": 64,
          "sequence_length": 1024, "forward_backward_per_step": 1}

sys.path.insert(0, os.path.join(BUNDLE, "environment", "runner"))
import capture as capture_mod  # noqa: E402


def _adapter(entry: str) -> str:
    return (
        f"\n\n_COPIED = {entry}\n\n\n"
        "def build_optimizer(params, lr=0.02, **kw):\n"
        "    params = list(params)\n"
        "    names = [f'blocks.0.w{i}' for i in range(len(params))]\n"
        "    for fn in (lambda: _COPIED(list(params), lr=lr),\n"
        "               lambda: _COPIED(list(zip(names, params)), lr=lr),\n"
        "               lambda: _COPIED(list(params)),\n"
        "               lambda: _COPIED(list(zip(names, params)))):\n"
        "        try:\n"
        "            return fn()\n"
        "        except Exception:\n"
        "            pass\n"
        "    raise RuntimeError('no constructor convention matched')\n"
    )


def build_candidates(root):
    """Materialize the optimizers the fixtures need, from private bundle sources only."""
    cand = os.path.join(root, "cand")
    os.makedirs(cand, exist_ok=True)
    shutil.copy(os.path.join(BUNDLE, "solution", "reference_optimizer.py"), os.path.join(cand, "reference.py"))

    manifest = json.load(open(os.path.join(BUNDLE, "solution", "corpus_manifest.json")))
    m = manifest["members"]["rec46_cwd_sota"]
    src = open(os.path.join(BUNDLE, "solution", "corpus", m["module"])).read()
    open(os.path.join(cand, "copy_rec46.py"), "w").write(src + _adapter(m["entry"]))

    # Defeat-2 regression. A published record with nothing changed but its learning rate.
    # The predecessor statistic scored this as novel; the gate must now block it.
    base = manifest["members"]["baseline_muon"]
    bsrc = os.path.join(BUNDLE, "solution", "corpus", base["module"])
    open(os.path.join(cand, "lr_rescale_baseline.py"), "w").write(
        f"import runpy\n_ns = runpy.run_path({bsrc!r})\n_C = _ns[{base['entry']!r}]\n\n\n"
        "def build_optimizer(params, lr=0.02, **kw):\n"
        "    return _C(list(params), lr=lr * 1.05)\n")

    # Defeat-1 regression. The key-leak attack, retained so the leak class stays closed.
    shutil.copy(os.path.join(BUNDLE, "solution", "fixtures", "a3_key_leak_attack.py"),
                os.path.join(cand, "a3_attack.py"))

    open(os.path.join(cand, "null.py"), "w").write(
        "import torch\n\n\nclass NullOpt(torch.optim.Optimizer):\n"
        "    def __init__(self, params, lr=0.02):\n        super().__init__(list(params), dict(lr=lr))\n\n"
        "    @torch.no_grad()\n    def step(self, closure=None):\n        return None\n\n\n"
        "def build_optimizer(params, lr=0.02, **kw):\n    return NullOpt(list(params), lr=lr)\n")
    return cand


def run_real_capture(cand_file, out_root, port, extra_env=None):
    """Drive the real runner so the fixture carries a genuine in-run capture."""
    sub = os.path.join(out_root, "submission")
    tele = os.path.join(out_root, "telemetry")
    os.makedirs(sub, exist_ok=True)
    shutil.copy(cand_file, os.path.join(sub, "optimizer.py"))
    env = dict(os.environ, TRACK3_CHAIN_KEY=KEY, TRACK3_TELEMETRY_DIR=tele,
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), **(extra_env or {}))
    r = subprocess.run([sys.executable, os.path.join(BUNDLE, "environment", "runner", "run_track3.py"),
                        "--mode", "smoke", "--seeds", "0,1",
                        "--submission", os.path.join(sub, "optimizer.py"),
                        "--logdir", os.path.join(sub, "logs")],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"runner failed for {cand_file}: {r.stderr.strip()[-300:]}")
    return tele


def chain(records, key):
    out, prev = [], "genesis"
    for r in records:
        dig = hmac.new(key.encode(), (prev + json.dumps(r, sort_keys=True)).encode(), hashlib.sha256).hexdigest()
        r2 = dict(r)
        r2["chain"] = dig
        out.append(r2)
        prev = dig
    return out


def assemble(name, cand_name, capture_src, opt_file, *, key=KEY, mode="full",
             seeds=("0", "1"), frozen_override=None, early_stop=False,
             drop_reported_step=None, tamper_capture=False, omit_optimizer=False):
    """Build one gradeable case. Every knob here plants exactly one defect."""
    import torch

    root = os.path.join(WORKDIR, "cases", name)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "submission", "logs"), exist_ok=True)
    os.makedirs(os.path.join(root, "telemetry", "novelty"), exist_ok=True)

    if not omit_optimizer:
        shutil.copy(opt_file, os.path.join(root, "submission", "optimizer.py"))
        opt_digest = hashlib.sha256(open(opt_file, "rb").read()).hexdigest()
    else:
        opt_digest = hashlib.sha256(open(opt_file, "rb").read()).hexdigest()

    for s in seeds:
        shutil.copy(os.path.join(capture_src, "novelty", f"seed{s}.pt"),
                    os.path.join(root, "telemetry", "novelty", f"seed{s}.pt"))

    recs = []
    for s in seeds:
        logpath = os.path.join(root, "submission", "logs", f"smoke_seed{s}.log")
        with open(logpath, "w") as lf:
            for step, loss in sorted(CURVE.items()):
                lf.write(f"step:{step}/3500 val_loss:{loss:.5f} train_time:1.000s\n")
                if drop_reported_step == step:
                    continue
                rec = {"seed": s, "step": step, "val_loss": round(loss, 5), "mode": mode,
                       "early_stop": bool(early_stop and step == 2850),
                       "optimizer_digest": opt_digest, **FROZEN}
                if frozen_override and step == 2800:
                    rec.update(frozen_override)
                recs.append(rec)

        capfile = os.path.join(root, "telemetry", "novelty", f"seed{s}.pt")
        cap = torch.load(capfile, weights_only=False)
        digest = capture_mod.capture_digest(cap)
        if tamper_capture:
            cap["params_after"][0][0] = cap["params_after"][0][0] + 0.05
            torch.save(cap, capfile)
        recs.append({"seed": s, "step": 3001, "val_loss": 3.2550, "mode": mode, "early_stop": False,
                     "optimizer_digest": opt_digest,
                     "novelty_capture_digest": digest,
                     "novelty_capture_file": f"seed{s}.pt",
                     "novelty_capture_steps": len(cap["params_after"]),
                     "novelty_spec_digest": capture_mod.spec_digest(), **FROZEN})

    with open(os.path.join(root, "telemetry", "run_record.jsonl"), "w") as f:
        for r in chain(recs, key):
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return root


def grade(root, port=29900):
    env = dict(os.environ, TRACK3_BUNDLE=BUNDLE, REWARD_PATH=os.path.join(root, "reward.json"),
               TRACK3_OUTCOMES=os.path.join(root, "outcomes.json"),
               TRACK3_TELEMETRY=os.path.join(root, "telemetry", "run_record.jsonl"),
               TRACK3_TELEMETRY_DIR=os.path.join(root, "telemetry"),
               TRACK3_SUBMISSION=os.path.join(root, "submission"),
               TRACK3_CHAIN_KEY=KEY, MASTER_PORT=str(port))
    subprocess.run([sys.executable, os.path.join(BUNDLE, "tests", "grade.py")],
                   env=env, capture_output=True, text=True)
    return json.load(open(os.path.join(root, "reward.json")))


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    cand = build_candidates(WORKDIR)
    ref_tele = run_real_capture(os.path.join(cand, "reference.py"), os.path.join(WORKDIR, "run_ref"), 29610)
    cpy_tele = run_real_capture(os.path.join(cand, "copy_rec46.py"), os.path.join(WORKDIR, "run_copy"), 29611)
    lrs_tele = run_real_capture(os.path.join(cand, "lr_rescale_baseline.py"),
                                os.path.join(WORKDIR, "run_lrs"), 29612)
    a3_report = os.path.join(WORKDIR, "a3_report.json")
    a3_tele = run_real_capture(
        os.path.join(cand, "a3_attack.py"), os.path.join(WORKDIR, "run_a3"), 29613,
        extra_env={"TRACK3_A3_REPORT": a3_report,
                   "TRACK3_A3_REC46": os.path.join(BUNDLE, "solution", "corpus", "rec46_cwd_sota.py")})

    ref = os.path.join(cand, "reference.py")
    cases = [
        ("clean_accepting_half", dict(cand_name="reference", capture_src=ref_tele, opt_file=ref),
         1.0, "graded_step=2850"),
        ("copy_verbatim_rec46", dict(cand_name="copy", capture_src=cpy_tele,
                                     opt_file=os.path.join(cand, "copy_rec46.py")),
         0.0, "behavioral_match_rec46_cwd_sota"),
        ("submission_absent", dict(cand_name="reference", capture_src=ref_tele, opt_file=ref,
                                   omit_optimizer=True),
         0.0, "submission_optimizer_absent"),
        ("reported_step_not_observed", dict(cand_name="reference", capture_src=ref_tele, opt_file=ref,
                                            drop_reported_step=2850),
         0.0, "reported_step_2850_seed_0_absent_from_telemetry"),
        ("frozen_contract_violation", dict(cand_name="reference", capture_src=ref_tele, opt_file=ref,
                                           frozen_override={"global_batch_tokens": 262144}),
         0.0, "frozen_violation_global_batch_tokens_step_2800"),
        ("chain_forged", dict(cand_name="reference", capture_src=ref_tele, opt_file=ref,
                              key="wrong-key"),
         0.0, "chain_break_at_step_2800"),
        ("non_full_mode", dict(cand_name="reference", capture_src=ref_tele, opt_file=ref, mode="smoke"),
         0.0, "non_full_mode_telemetry_smoke"),
        ("single_seed", dict(cand_name="reference", capture_src=ref_tele, opt_file=ref, seeds=("0",)),
         0.0, "need_at_least_2_seeds_got_1"),
        ("val_selection_early_stop", dict(cand_name="reference", capture_src=ref_tele, opt_file=ref,
                                          early_stop=True),
         0.0, "val_selection_at_step_2850"),
        ("capture_tampered_after_commit", dict(cand_name="reference", capture_src=ref_tele, opt_file=ref,
                                               tamper_capture=True),
         0.0, "novelty_capture_digest_mismatch_seed_0"),
        ("lr_rescaled_record_x105", dict(cand_name="lr_rescale", capture_src=lrs_tele,
                                         opt_file=os.path.join(cand, "lr_rescale_baseline.py")),
         0.0, "behavioral_match_baseline_muon"),
        ("key_leak_attack_a3", dict(cand_name="a3", capture_src=a3_tele,
                                    opt_file=os.path.join(cand, "a3_attack.py")),
         0.0, "behavioral_match_rec46_cwd_sota"),
    ]

    print(f"{'case':36} {'reward':>7} {'expect':>7}  reason")
    print("-" * 100)
    unexpected = []
    for i, (name, kw, exp_reward, exp_reason) in enumerate(cases):
        root = assemble(name, **kw)
        r = grade(root, port=29900 + i)
        ok = abs(r["reward"] - exp_reward) < 1e-9 and r["reason"] == exp_reason
        print(f"{name:36} {r['reward']:7.3f} {exp_reward:7.3f}  {r['reason']}{'' if ok else '   <-- UNEXPECTED'}")
        if not ok:
            unexpected.append((name, r["reward"], r["reason"], exp_reward, exp_reason))

    print("-" * 100)
    a3 = json.load(open(a3_report))
    print(f"A3 key-leak probe: {a3['verdict']}")
    print(f"  key in worker environment            : {a3['step1_env_key']}")
    print(f"  window selector still present        : {a3['step5_select_window_start_present']}")
    for pid, v in sorted(a3["step1_ancestors"].items()):
        print(f"  ancestor {pid} environ / mem          : {v['environ_has_key']} / {v['mem_readable']}")
    if a3["attack_viable"]:
        unexpected.append(("a3_leak_probe", "-", a3["verdict"], "-", "A3 DEFEATED"))
    print("-" * 100)
    accepting = [c for c in cases if c[2] > 0.0]
    print(f"cases={len(cases)} accepting_half={len(accepting)} rejecting_half={len(cases) - len(accepting)} "
          f"unexpected={len(unexpected)}")
    for u in unexpected:
        print(f"  UNEXPECTED {u[0]}: got reward={u[1]} reason={u[2]}, expected reward={u[3]} reason={u[4]}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
