# Auto-NanoGPT Harness Bisection — Skeptical Veteran's Report

**Subject:** `PrimeIntellect-ai/experiments-autonomous-speedrunning`
**Blog:** https://www.primeintellect.ai/auto-nanogpt
**Basis:** `KellerJordan/modded-nanogpt` → `records/track_3_optimization/`
**Audit date:** 2026-08-11

---

## 0. Verdict Up Front

Prime Intellect's harness is **not code. It is 5 markdown files per agent, per wave — plus one unchanged upstream Python trainer they never wrote.** The intelligence lives in the prose: it defines conduct, gates, budgets, and file layouts that a general-purpose CLI agent (Claude Code / Codex) is expected to obey. The Python that actually ran (`variants/*.py`) was written *by the agent* at run-time, forking `train_gpt_simple.py`.

That is elegant, and it is also the reason you cannot "convert the harness to code" as a straight port. You port it by (a) declaring the *policy* it encodes, and (b) wiring the *execution* it presumes (SLURM `torchrun` on the modded-nanogpt Track-3 trainer + a log parser) into whatever runner you choose — Harbor, standalone, or hybrid.

Everything below is evidence-first. If I can't cite it, I mark it UNVERIFIED.

---

## 1. What They Took From `modded-nanogpt` (Provenance)

Verbatim, one file, no code changes:

| From modded-nanogpt | Used in auto-nanogpt as | Evidence |
|---|---|---|
| `records/track_3_optimization/train_gpt_simple.py` (372 lines, 14 KB) | The baseline every agent fork descends from. Every `data/runs_self_contained/agents/<agent>/runs/<id>/launched_script.py` opens with `# train_gpt_simple.py \n # This file descends from the [NanoGPT speedrun]...` | Header string verified in librarian sample |
| `records/track_3_optimization/README.md` (rules, statsig formula, quickstart) | The *substance* of the "hard rules" section inside every `AGENTS.md` (val≤3.28, 1 fwd-bwd/step, hardcoded HPs, dataset immutable) | Text overlap in v1/claude-code/AGENTS.md "Benchmark hard rules" |
| `data/cached_fineweb10B.py` + `data/fineweb10B/*.bin` | Data pipeline unchanged; run as `python data/cached_fineweb10B.py 20` before any agent loop | Track-3 README quickstart |
| Self-source snapshot pattern (`with open(sys.argv[0]) as f: code = f.read()`) | Reused as the source of `source_snapshot.py` in every exported run — it's the log header extract | `train_gpt_simple.py:10`; librarian export mapping |

They **did not** fork the trainer, add CLI args, wrap it in a runner, or write any harness Python. The trainer stays "hardcoded HPs, edit source" — because the *agent* is the one editing the source.

Everything else in `PrimeIntellect-ai/experiments-autonomous-speedrunning` is authored by them (or by their agents at run-time).

---

## 2. The Harness, File by File (Bisection)

### 2.1 Prescriptive layer — hand-written markdown per agent (~5 files)

For each `<wave>/<agent>/` (there are 8 combos: `v1|novelty|v2|v3` × `claude-code|codex`):

| File | Bytes / lines | Purpose | Who reads it | Editable? | Notes |
|---|---|---|---|---|---|
| `AGENTS.md` | ~400 lines pure prose, zero imports | **Conduct spec.** 6 always-law rules (hard benchmark rules, noise-floor gate, stuck detector, slug-stack ≤3, 2-seed reproduction, pruning-before-submit); autonomy doctrine ("run autonomously, don't ask"); subagent doctrine (spawn ultra-frequently; Opus for synthesis, Sonnet for scan); paper-reading protocol (deeply, not skimmingly, 800–2000 words, ends "To port to our setup:"); ideas doctrine (7-part template, math + intuition + web ref + ablation plan + failure prediction); polling/wakeup schedule; git rules (never push origin; commit code and `ai/` separately) | Orchestrator agent at startup + every re-entry | No (law) | The load-bearing file |
| `goal.md` | ~1 page | **Mission context.** Target = 3.28 val loss under 3500 steps on `train_gpt_simple.py`. Explicit optimizer/technique menu (MuonClip, AdEMAMix, Adam-mini, Lion, Sophia, SOAP, Kron, Shampoo, sign-based, K-FAC, Lookahead, RAdam, Sophia-G, Adan, μP/muTransfer, DeepNet, σReparam, qk-norm, WSD, trapezoid, cyclical-WSD, schedule-free, BS-increase, EMA averaging…). Scope IN (optimizer, HPs, schedules, init, instrumentation). Scope OUT (arch, batch, data, multi-step inner loops). Budget: 1 node 8×H{100,200}, ~15 min/run, one run at a time, no calendar deadline. | Orchestrator agent | No (mission) | Frames the search space |
| `plan.md` | mutable, small | **Current state.** Overwrite-in-place: target, best reproduced, active family, stuck counter, runs since last pruning, noise-floor estimates, delta vs canonical. Rank-ordered "Active picklist" table. Append-only "Lessons" log. Explicitly designed to be regenerable from `runs.jsonl` after context compaction. | Orchestrator agent | Yes (agent rewrites) | The state file |
| `scratchpad/THREAD.md` | grows over run | **Durable mission log.** So a fresh orchestrator can recover after context compaction. Linked in blog. | Orchestrator + human reviewers | Append | Recovery ballast |
| `scratchpad/picklist.md` + `scratchpad/audits.md` | small | Working shortlist of next candidates; audit log for rule-violation checks. | Orchestrator | Yes | Bookkeeping |

Every one of these files is markdown. No Python imports it. Nothing "loads" it. It is *literally passed to the LLM as context* at the start of each session. That is the entire harness surface.

### 2.2 Agent-generated work product — Python + shell + JSONL per run

Inside `<wave>/<agent>/scratchpad/`:

| Path | Kind | Written by | Purpose |
|---|---|---|---|
| `variants/<slug>.py` | Python (5–20 KB) | Agent | Fork of `train_gpt_simple.py` with the proposed modification. This *is* the code that runs. |
| `sbatch-stubs/<slug>.sh` OR top-level `run_*.sh` | Bash | Agent | SLURM inner stub: `#SBATCH --partition=preempt --gres=gpu:8 --time=00:30:00` + `torchrun --standalone --nproc_per_node=8 <variant>.py` |
| `runs/<slug>.log` | Text (50–500 KB) | Trainer stdout | Full training log — this is what the grader parses |
| `runs.jsonl` | JSONL | Agent + probably a helper | Ledger row per run: slug, final_val_loss, step_to_3_28, train_steps, train_time_s, seed(s), status |
| `sweeps/<name>/` | dir | Agent | Grouped multi-cell HP sweep artifacts |
| `ideas/<slug>.md` | md | Agent (or spawned subagent) | 500–1500-word idea write-up per the 7-part template |
| `papers/<arxiv-id>.md` | md | Agent (or spawned subagent) | 800–2000-word deep read per the paper doctrine, always ends `**To port to our setup:** <paragraph>` |

Everything here is created *at run-time by the agent*. The maintainers do not check any of it in as a starting point; they check it in as an archive after the run.

### 2.3 Post-hoc archive — the export they published

`data/runs_self_contained/` — the ~41k-file bulk portion. This is not part of the harness; it is the cleaned dataset of runs so the community can reproduce/inspect. Structure verified:

- Root: `manifest.json`, `runs.jsonl` (~50 MB), `runs.csv` (~40 MB), `dropped_runs.jsonl`, `README.md`
- Per run: `metadata.json` (structured metrics), `train.log` (raw stdout), `launched_script.py` (the variant Python), `source_snapshot.py` (log-header extract), `launch_stub.sh`, `console.log`
- Run counts: cc_v1=605, codex_v1=2165, cc_novelty=81, codex_novelty=254, cc_v2=459, codex_v2=2729, cc_v3=1059, codex_v3=3076 → **10,428 total**, 57 dropped (missing `config_path`)

**Skeptic's note**: `metadata.json` is the closest thing they have to a Harbor-style structured result. Its fields (`final_val_loss, min_val_loss, final_step, train_steps, step_to_3_28, num_val_points, train_time_s, step_avg_ms`) are basically the Harbor `metrics` dict — just not written by the trainer, computed by an extractor from `train.log`.

---

## 3. How It Actually Runs (Execution Bisection)

There is no runner binary. The sequence is:

1. **Human** starts a Claude Code (or Codex) session in `/beegfs/elie/modded-nanogpt-agent/` (a git worktree on branch `agent/track_3_speedrun`, deliberately outside the repo tree per AGENTS.md).
2. **Agent** reads `AGENTS.md` + `goal.md` + `plan.md` + `scratchpad/THREAD.md` at session start.
3. **Agent** picks a candidate from the picklist, writes/edits `scratchpad/variants/<slug>.py`, writes `scratchpad/sbatch-stubs/<slug>.sh`.
4. **Agent** launches:
   - Main-thread (single run): `cd /beegfs/elie/modded-nanogpt && torchrun --standalone --nproc_per_node=8 records/track_3_optimization/train_gpt_simple.py 2>&1 | tee records/track_3_optimization/ai/scratchpad/runs/<run-id>.log` (path verbatim from AGENTS.md §Benchmark run command)
   - Fan-out (sweeps): `sbatch scratchpad/sbatch-stubs/<slug>.sh` into SLURM `preempt` partition (idle nodes); polls via `sinfo -p preempt -h -o '%T %D'` and `sacct -j <id> -o State -X -n`
5. **Agent** waits ~15 min (armed wakeup: 20 min single, 60 min sweep of ≥5 cells, ≤90 min).
6. **Agent** parses `runs/<run-id>.log` for `step:{S}/{N} val_loss:{V:.5f}` lines, computes step_to_3.28, appends row to `runs.jsonl`, updates `plan.md`.
7. **Agent** applies the gates (noise-floor, stuck detector, 2-seed repro, pruning) *itself*. No external verifier.
8. On promotion: agent opens a PR against `eliebak/modded-nanogpt` (fork, not upstream). Human approves. This is one of the ~100 human interventions.

**What's automated, what isn't:**
- Fully automated: variant generation, sbatch dispatch, log parsing, gate application, `runs.jsonl` maintenance, `plan.md` state update, subagent spawning for paper reads / sweeps.
- Not automated: `git push origin` (blocked by rules), submission PR (requires human OK), initial data download, rule-violation correction (one of the ~100 human touches).
- Failure mode observed: Claude Code repeatedly quits ("Stopping the autonomous loop here") → ~22h idle in v1; Codex never stops but grinds the same HP surface.

**Compute infra dependencies (hard):**
- SLURM cluster with two partitions: `cluster` (reserved for main-thread) and `preempt` (idle-node fan-out)
- Shared filesystem at `/beegfs/elie/`
- `torchrun --standalone --nproc_per_node=8` (H100/H200)
- Cached FineWeb-10B shards at `data/fineweb10B/fineweb_{train,val}_*.bin`

None of that is Harbor-shaped. All of it is trivially wrappable.

---

## 4. What This Means for "Convert to Code / Our Requirements"

There are two things the auto-nanogpt harness bundles that we have to separate cleanly before we can port:

**A. Policy** (the AGENTS.md rules) — this is genuinely generic and reusable. It is a *conduct constitution* for an autonomous optimization loop.

**B. Execution** (the torchrun + log-parse + gate loop) — this is specific to nanogpt-style benchmarks and is where Harbor slots in.

If you try to port them as one thing you'll end up with a Harbor task whose `instruction.md` is a 400-line AGENTS.md copy and whose `test.sh` awkwardly re-implements the noise-floor gate. That is wrong. Split them:

- **Policy → prompt bundle** loaded by the *outer* orchestrator (agent process, not the task). Harbor doesn't own it; the agent runtime does.
- **Execution → Harbor task** with `train_gpt_simple.py` in the container, `torchrun` in `test.sh` (or a two-step task with the agent phase producing `variant.py` and the verifier phase running it), reward = a normalized score derived from `step_to_3.28`.

That mapping is what makes the three integration paths below well-defined instead of hand-wavy.

---

## 5. Three Integration Paths — Pros, Cons, Effort, When to Pick

For each: what you build, what changes, what breaks, honest effort.

### Path 1 — Harbor task per candidate variant (`environment_mode = "shared"`)

**Shape.** One Harbor task = one training run. Agent gets a candidate variant in the environment; agent phase invokes `torchrun train_gpt_simple.py` (or a variant) and dumps a log; verifier phase parses the log and writes `reward.txt`.

**Directory:**
```
tasks/nanogpt-speedrun/
├── task.toml
├── instruction.md          # short: "beat 3.28 val loss under N steps on this trainer"
├── environment/
│   ├── Dockerfile          # CUDA 12.6 + torch==2.11 + hf_hub + cached fineweb10B
│   └── train_gpt_simple.py # verbatim copy from track3_optimization_harness/
├── solution/
│   └── solve.sh            # optional: canonical Muon reference for oracle mode
└── tests/
    ├── test.sh             # bash: parse log, compute step_to_3.28, write reward.txt
    └── grader.py           # descendant of records/track_3_optimization/results/*/claim_stats.py
```

**task.toml (concrete):**
```toml
version = "1.0"

[task]
name = "nanogpt/track3-speedrun"

[metadata]
author_name = "you"
difficulty = "hard"
category = "ml-optimization"
tags = ["nanogpt", "muon", "speedrun"]

[agent]
timeout_sec = 3600.0            # 60 min for a single run; bump for slower variants

[verifier]
timeout_sec = 120.0
environment_mode = "shared"     # verifier runs in same container as agent, sees /logs/

[environment]
build_timeout_sec = 1800.0      # torch install is heavy
docker_image = "nanogpt-track3:cuda126"
cpus = 16
memory_mb = 131072
storage_mb = 51200
gpus = 8
gpu_types = ["H100", "H200"]
network_mode = "allowlist"
allowed_hosts = ["huggingface.co", "*.huggingface.co", "pypi.org", "*.pypi.org"]
```

**test.sh:**
```bash
#!/bin/bash
set -e
LOG=$(ls -t /logs/artifacts/runs/*.log | head -n1)
python /app/tests/grader.py "$LOG" > /tmp/score.json
mkdir -p /logs/verifier
cp /tmp/score.json /logs/verifier/reward.json
exit 0
```

**grader.py** — descendant of `claim_stats.py`. Emit:
```json
{"reward": 0.87, "step_to_3_28": 2820, "final_val_loss": 3.2794, "train_time_s": 872.4}
```

Reward mapping (concrete, defensible): `reward = clip((3500 - step_to_3.28) / (3500 - 2500), 0, 1)` if val ≤ 3.28 else `0`. Aligns with the ghost-test contract from the pytest cache (`test_grade_clipped_to_zero_one`, `test_baseline_returns_grade_zero`).

**Pros.**
- Reuses everything we already put in `track3_optimization_harness/`.
- Standard Harbor CLI: `harbor run -p tasks/ -m <model> -a bash -n 10 -j 5 --env docker --jobs-dir jobs/`.
- Gets the full Harbor stack: trial re-runs, `jobs/<name>/trials/<task>_trial-N/{result.json,trajectory/,artifacts/}`, `report.html` viewer.
- Long-running-GPU-safe: `timeout_sec` supports 4h+; `/logs/artifacts/checkpoint.*` preserved.
- Oracle/NOP evaluation becomes free: `solution/solve.sh` runs the canonical Muon reference; NOP is a task with `solution/` empty and reward defined as the baseline delta.

**Cons.**
- Doesn't capture the PI *autonomous loop* — this is one variant per task. You'd run Harbor in a `while True: propose variant → run task → read reward → update plan` outer loop that YOU write.
- Doesn't preserve `AGENTS.md` conduct rules automatically. Those live in your outer orchestrator, not in the task.
- Multi-seed pruning has to be modeled as `-n 2` (or scripted).

**Effort estimate** (skeptic's honest number):
- Write Dockerfile + data caching wrapper: **~1 day**
- Write test.sh + grader.py (adapt claim_stats.py): **~4 hours**
- Write task.toml + instruction.md: **~2 hours**
- Test end-to-end on 1×H100 with 1 seed for 100 steps: **~4 hours**
- **Total: ~2 days** to first green trial.

**Pick this if:** you want a clean, submittable, Harbor-native benchmark that any agent (Claude Code, Codex, custom) can be pointed at. Best fit for "publish a benchmark task."

---

### Path 2 — Standalone harness (fork the PI pattern; don't touch Harbor)

**Shape.** Recreate PI's markdown-driven loop verbatim, with our own paths. No Harbor. Runner is `torchrun` + a shell/python outer loop we write.

**Directory:**
```
autonomous_speedrun/
├── AGENTS.md              # our conduct constitution (adapt from v1/claude-code/AGENTS.md)
├── goal.md                # our mission (adapt from v1/claude-code/goal.md)
├── plan.md                # empty starter (agent rewrites)
├── scratchpad/
│   ├── THREAD.md          # empty starter
│   ├── variants/          # empty
│   ├── sbatch-stubs/      # empty
│   ├── runs/              # empty
│   └── runs.jsonl         # empty
├── train_gpt_simple.py    # verbatim from track3_optimization_harness/
├── data/cached_fineweb10B.py
├── requirements.txt       # torch==2.11 huggingface_hub
└── tools/
    ├── ingest_log.py      # runs.jsonl updater
    ├── noise_floor.py     # baseline-Muon 3-seed measurement
    └── prune.py           # 2-seed leave-one-out pruning
```

**Pros.**
- Highest fidelity to what PI actually did (all 10,428 runs used this shape).
- Zero Harbor dependency; runs on any SLURM cluster (or even a bare 8×H100 box with a bash `while true` loop).
- Agent-writable: everything (`plan.md`, `variants/*.py`, `sbatch-stubs/*.sh`, `runs.jsonl`) is edited by the agent itself.
- Cheapest to bootstrap if you already have H100 access.

**Cons.**
- You reimplement Harbor's plumbing: trial concurrency, timeouts, artifact isolation, result viewer, task registry. That's what the PI team spent weeks tuning.
- No `report.html`, no trial dashboard, no `--env docker/e2b/daytona/modal/gke/langsmith/runloop` — you're on bare SLURM only.
- No reward normalization / cross-task comparability.
- The autonomy-failure Claude Code shows (quitting after 22h) is on you to work around; PI added their ~100 human interventions to compensate.

**Effort estimate:**
- Copy + adapt the 5 markdown files: **~1 day** (careful adaptation, not verbatim copy — you have to rewrite paths, budget numbers, cluster names, and the noise-floor estimates for your hardware).
- Write `tools/ingest_log.py`, `noise_floor.py`, `prune.py`: **~2 days** (the log-parse contract exists in claim_stats.py; noise-floor needs baseline runs which is a wall-clock cost, not code cost).
- Cluster setup (SLURM partitions, worktree convention, shared FS): **~1 day** if infra already exists; **~week+** if not.
- **Total: ~1 week** engineering, plus baseline-run wall-clock (~4 hours to nail noise-floor at 3 seeds × ~12 min).

**Pick this if:** you have SLURM + H100 already, want the PI setup faithfully, and don't need cross-benchmark comparability or Harbor's ecosystem.

---

### Path 3 — Hybrid (recommended for our situation)

**Shape.** Path 1's Harbor task is the *reward machine*. Path 2's markdown is the *policy bundle* the outer orchestrator loads. The autonomous loop drives Harbor by calling `harbor run` per candidate.

**Layout:**
```
autonomous_speedrun/
├── policy/                              # from Path 2, stripped of runner details
│   ├── AGENTS.md                        # conduct rules — but the "Benchmark run command"
│   │                                    #   section now says "call harbor_run(variant_path)"
│   ├── goal.md                          # unchanged
│   ├── plan.md                          # mutable
│   └── scratchpad/{THREAD.md, variants/, ideas/, papers/, picklist.md, audits.md}
├── tasks/nanogpt-speedrun/              # from Path 1, unchanged
│   ├── task.toml
│   ├── instruction.md
│   ├── environment/{Dockerfile, train_gpt_simple.py}
│   ├── solution/solve.sh                # oracle/NOP
│   └── tests/{test.sh, grader.py}
├── runner/
│   ├── harness.py                       # thin outer loop
│   ├── mount_variant.py                 # copy scratchpad/variants/<slug>.py into task env override
│   └── ingest_result.py                 # read jobs/<name>/trials/*/result.json → runs.jsonl
└── runs.jsonl                           # aggregated ledger
```

**Outer loop (pseudocode, ~100 LOC):**
```python
while not stop_gate_hit():
    variant_path = orchestrator.propose_next_variant()   # LLM agent writes scratchpad/variants/<slug>.py
    task_dir = mount_variant("tasks/nanogpt-speedrun", variant_path)
    job = harbor.run(task_dir, model=None, agent="oracle",  # or "bash" with an inner LLM
                     n=2, j=2, env="docker", timeout_agent=3600)
    for trial in job.trials():
        ingest_result(trial, ledger="runs.jsonl")
    orchestrator.update_plan(ledger="runs.jsonl")
    orchestrator.apply_gates()   # noise-floor / stuck / pruning — inside the LLM agent
```

**Why this is the right shape:**
- Harbor owns what Harbor is good at: containerized execution, timeouts, trial concurrency, artifact capture, reward extraction, cross-run comparability, `report.html`.
- Markdown policy owns what markdown is good at: conduct constitution, mission framing, mutable state that survives context compaction.
- The two are cleanly decoupled — you can swap Claude Code for Codex without editing tasks; you can swap the trainer without editing policy.
- Oracle/NOP evaluation: `solve.sh` in the task is the oracle path; the agent phase can be `oracle` in Harbor to compare against agent-driven agent phases.

**Pros.**
- Best of both: PI's proven policy pattern, Harbor's proven execution plane.
- Everything except `runner/*.py` (~200 LOC total) is either lifted verbatim (`train_gpt_simple.py`) or narrowly adapted (5 md files, 1 task.toml, 1 grader).
- Directly answers all six of the ghost-test node names from the original audit — the grader is now Harbor's reward machine; the ledger is Harbor's `result.json` aggregated.
- Portable: you can host the Harbor task on any of docker / e2b / daytona / modal / gke / langsmith / runloop.

**Cons.**
- Two moving parts to keep in sync (policy prose ↔ runner API surface). Mitigated by keeping the runner interface tiny (`propose_next_variant`, `update_plan`, `apply_gates`).
- You still write `runner/harness.py` — the LLM loop with tool exposure. Non-trivial but bounded.

**Effort estimate:**
- Path 1's task: **~2 days** (from above).
- Adapt Path 2's 5 md files with new "Benchmark run command" section pointing to `runner/harness.py`: **~1 day**.
- Write `runner/harness.py` + `mount_variant.py` + `ingest_result.py`: **~2 days** (Harbor Python API + subprocess + jsonl append).
- End-to-end test with 1 seed on 1×H100, 100 steps: **~4 hours**.
- **Total: ~1 week** to first autonomous loop that reports Harbor-native rewards.

**Pick this if:** you want the PI capability with a Harbor-native benchmark side-output — i.e. both a running research loop AND a task you can publish/share/evaluate other agents against.

---

## 6. Comparison Matrix

| Dimension | Path 1 (Harbor task) | Path 2 (Standalone) | Path 3 (Hybrid) |
|---|---|---|---|
| Fidelity to PI method | Low (loses autonomy loop) | High (verbatim pattern) | High (verbatim + typed reward) |
| Harbor benefits | Full | None | Full |
| Bootstrap effort | ~2 days | ~1 week | ~1 week |
| Cross-agent portability | High | Low | High |
| Cross-benchmark comparability | High | Low | High |
| Oracle/NOP eval | Free (`solution/`) | Manual | Free |
| Autonomy loop | You write outer wrapper | Native | Native |
| Cluster infra required | Any Harbor-supported env | SLURM only | Any Harbor-supported env |
| Recommended | If benchmark-first | If research-first + already on SLURM | **Default** |

---

## 7. Blockers & UNVERIFIED items

**Verified blockers (real, budget for them):**
1. **CUDA + torch==2.11 + kernels** — the Dockerfile is nontrivial. Track-3 README explicitly warns torch==2.10 has a NaN bug on A100 with torch.compile; use 2.11. Test on your actual GPU class first.
2. **8×H100 assumption baked into `train_gpt_simple.py`**: `assert 8 % world_size == 0`. If you only have 1×H100, you can run at `nproc_per_node=1` but the schedule constant `train_steps=3250` and batch (`8*64*1024`) won't yield the 3.28 target in reasonable time. Path 1's task should declare `gpus=8` and accept it won't run on 1-GPU CI.
3. **CWD-relative paths** in `train_gpt_simple.py` (`Path.cwd().glob("data/fineweb10B/...")`, `logs/{uuid}.txt`). The Docker WORKDIR and volume mounts need to match.
4. **`dist.get_rank()` at import time in `evals/hellaswag.py:15-16`** — irrelevant to Path 1/2/3 baseline because Track-3 baseline doesn't call hellaswag. If you later add optional eval, this needs refactoring.

**UNVERIFIED items (I don't have proof; you may want to verify before committing):**
1. **Harbor version compatibility.** Cited v0.16.1+. If Harbor breaks task.toml between versions, pin explicitly.
2. **Whether `harbor run --agent oracle` supports GPU-heavy solve.sh** or expects fast shell tasks. Skim the harbor examples repo for a long-running-GPU example before committing Path 1/3.
3. **Whether the aggregated `runs_self_contained/manifest.json` fully documents the export tool** — sample only showed field names. If we need bit-exact reproduction of PI's ledger, we'd need their export script (not visible in the repo tree we cataloged).
4. **The 8 ghost test names in `.pytest_cache/v/cache/nodeids`** — they *look* like the right grader contract (`test_grade_clipped_to_zero_one`, `test_grader_uses_median_steps_across_seeds`, etc.). Source file not recoverable from git. Grader design in Paths 1/3 should match this contract on principle, but treat the exact test bodies as UNVERIFIED.
5. **`v1/claude-code/AGENTS.md` § "Never create a worktree inside the repo tree itself"** — verified in file. But whether the worktree convention is *required* for Path 2/3 or just PI's preference is UNVERIFIED. Read: it's cheap to follow either way.

---

## 8. Recommendation

Go **Path 3 (Hybrid)**. Concrete first-week plan:

1. **Day 1** — extend `track3_optimization_harness/` with Docker: write `environment/Dockerfile` (CUDA 12.6, torch==2.11, huggingface_hub, tiktoken, datasets, tqdm). Bake `data/cached_fineweb10B.py` install step. Test build locally.
2. **Day 2** — write `tasks/nanogpt-speedrun/task.toml`, `instruction.md`, `tests/test.sh`, `tests/grader.py` (adapt `claim_stats.py`). Aim for a single-seed reward on a truncated run first (`train_steps=100`).
3. **Day 3** — copy `v1/claude-code/{AGENTS,goal,plan}.md` into `autonomous_speedrun/policy/`, adapt paths + hardware numbers + noise-floor to yours. Rewrite the "Benchmark run command" section to invoke your `runner/harness.py`.
4. **Day 4** — write `runner/harness.py` (thin: propose → harbor.run → parse result.json → append runs.jsonl → hand policy to LLM as context on next turn). ~150 LOC.
5. **Day 5** — first end-to-end autonomous cycle. 1 variant, 2 seeds, verify reward comes out sane. Instrument.
6. **Week 2+** — measure noise-floor properly (baseline 3 seeds), then unleash the outer loop with the stuck-detector + pruning rules from AGENTS.md.

Do NOT start with Path 2. The autonomy failures PI observed (Claude Code quitting, Codex plateauing) will hit you too, and Harbor's trial machinery is worth having under you when you diagnose them.

Do NOT start by writing new md files from scratch. Adapt PI's. Their AGENTS.md is 400 lines of hard-won prose (2 weeks × 2 models × 4 waves × the fixes for the ~100 human interventions). Rewriting it is negative-value work.

---

## 9. TL;DR

- **What they took from modded-nanogpt:** exactly one file (`records/track_3_optimization/train_gpt_simple.py`) verbatim, plus its README rules and cached data pipeline. Nothing else. No trainer fork, no runner wrap.
- **What their "harness" is:** 5 markdown files per agent per wave (`AGENTS.md`, `goal.md`, `plan.md`, `scratchpad/THREAD.md`, `picklist.md` / `audits.md`). No Python. No runner. The intelligence is in the *conduct spec*.
- **What "harness = md, no code" really means:** the harness is the *policy the agent reads at session start*. The Python that ran (`variants/*.py`) was written by the agent per-run.
- **Harbor fit:** clean — the training-run-as-task pattern maps to Harbor's task/verifier/reward model directly. `timeout_sec` supports the ~15 min run length; `gpus=8` + `gpu_types=["H100"]` declarations are first-class; `/logs/artifacts/` + `/logs/verifier/reward.json` are the artifact/reward contract. Multi-seed pruning maps to `-n N` trials.
- **What to build:** Path 3 hybrid — Path 1's Harbor task + Path 2's markdown policy + a ~200-LOC outer runner. ~1 week to first autonomous cycle.

---

## Appendix A — Harbor Task Format Reference (distilled)

**Canonical repo:** `github.com/harbor-framework/harbor` · **Docs:** `harborframework.com/docs/tasks` · **DOI:** 10.5281/zenodo.20953922 · **Version referenced:** v0.16.1+ (Aug 2026). Created by the Terminal-Bench authors; integrated by Prime Intellect into `PrimeIntellect-ai/verifiers`; TRL integration via HarborSpec / GRPOTrainer.

**Directory contract:**
```
<task-id>/
├── task.toml            # REQUIRED — config + metadata (TOML)
├── instruction.md       # REQUIRED — agent-readable instructions
├── environment/
│   ├── Dockerfile       # REQUIRED — container definition
│   └── [docker-compose.yaml]  # optional — for GPU/TPU
├── solution/
│   └── solve.sh         # REQUIRED for oracle — optional otherwise
└── tests/
    ├── test.sh          # REQUIRED — verifier entrypoint (bash)
    └── [test_*.py]      # optional pytest
```

**Multi-step syntax (long training pipelines):**
```toml
[[steps]]
id = "train"
[steps.agent]
timeout_sec = 14400.0      # 4h

[[steps]]
id = "grade"
[steps.verifier]
timeout_sec = 600.0
```

**Verifier reward contract (`tests/test.sh` must):**
1. Execute grading logic.
2. Write reward to ONE of:
   - `/logs/verifier/reward.txt` — single float 0.0–1.0
   - `/logs/verifier/reward.json` — `{"reward": 0.85, "accuracy": 0.92, "time_sec": 1.23}`
3. Exit code: 0 success, nonzero → surfaces as `ProgramError` (not silent zero).

**Runner:**
```bash
harbor run \
  -d "<dataset-id>"          # OR: -p <local-tasks-dir>
  -m "<model>" -a "<agent>"  # e.g. anthropic/claude-opus-4-7 ; bash|oracle|custom
  --env docker               # docker | e2b | daytona | modal | gke | langsmith | runloop
  -n <trials> -j <concurrency>
  --ae KEY=VAL --ve KEY=VAL
  --jobs-dir <path>
```

**Output:**
```
jobs/<job-name>/
├── result.json                        # job-level summary
├── trials/<task-id>_trial-N/
│   ├── result.json                    # reward, status, metrics, timestamps
│   ├── trajectory/                    # captured agent trace
│   └── artifacts/                     # files from /logs/artifacts
└── report.html                        # web viewer
```

**Gap analysis (Harbor vs Track-3 today):**
| Aspect | Track-3 today | Harbor requires | Adapter effort |
|---|---|---|---|
| Config | Hardcoded HPs in .py source | `task.toml` (TOML) | High — generate TOML per variant |
| Output | `logs/{uuid}.txt` unstructured | `/logs/verifier/reward.txt` float 0–1 | High — parse + normalize |
| Grader | `claim_stats.py` (stdlib, log regex) | `tests/test.sh` (bash entrypoint) | Medium — wrap in bash + write reward.txt |
| Env | pip + torchrun on bare host | Dockerfile | High — write Dockerfile, expose GPU |
| Instruction | N/A (training script) | `instruction.md` | Medium — write once |
| Solution/oracle | N/A | `solution/solve.sh` (optional) | Low — skip for RL/training |
| Verifier mode | in-process | shared vs separate container | Medium — recommend shared |
| Timeouts | none | explicit per phase | Low — declare `agent.timeout_sec` ≥ 15–60 min |
| GPU declare | `nproc_per_node=8` in shell | `[environment] gpus=N gpu_types=[...]` | Low — declare in TOML |

---

## Appendix B — Reference Harbor tasks in the wild

- **Terminal-Bench 2 official:** `github.com/harbor-framework/harbor/tree/main/examples/tasks` (~10 shell tasks)
- **Data Agent RL Eval:** `huggingface.co/datasets/AdithyaSK/data_agent_rl_environment_eval` (366 tasks, pandas+SQL, mode-aware grader.py)
- **Hello MCP Harbor:** `github.com/PrimeIntellect-ai/verifiers/tree/main/environments/hello_mcp_harbor/tasks/hello-mcp`
- **CLI-Gym:** PI/CLI-Gym on HuggingFace (~500 rows), converted to Harbor via `PrimeIntellect-ai/verifiers` PR #1665
- **Long-running GPU precedent:** "Endless Terminals" (arXiv:2601.16443, Feb 2026) — uses Harbor for persistent multi-turn Apptainer shells

---

## Appendix C — Prime Intellect blog key numbers (for context)

- Codex (gpt 5.5 xhigh) + Claude Code (opus 4.7 xhigh); ~2 weeks on idle compute
- ~10,000 runs, ~14,000 H200 hours
- Records: Opus 2930 steps, Codex 2950 steps. Human baseline 2990. Canonical Muon reference 3500.
- 4 waves: v1 (from scratch), novelty (novelty-constrained — agents FAILED to invent new methods without upstream records), v2 (continuation), v3 (further continuation)
- ~100 human interventions total
- Only 23.9B tokens total (including cached)
- Best techniques in final stacks: Contra-Muon, Soft-Muon, NorMuon-lite, MLP+V SOAP with warmstart/skip-first, tangent-sphere radial, LR power 1.2, LACV on q/k/mlp_proj, Radial damping 0.45/1.0, u/w floor 0.3825, AdamW betas (0.8, 0.99). Many of these are already present in the record #46 stack shipped in `track3_optimization_harness/`.
