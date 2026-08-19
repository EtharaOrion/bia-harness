# policy/ — the harness

This folder IS the harness. No Python here. Just markdown that an autonomous orchestrator (Claude Code / Codex / any LLM CLI agent) reads at session start.

Scope: this serves the planner-authors-variant path in `legacy/harness2/`, which
scaffolds these files via `scaffold_policy.py` and reads them in
`orchestrator.build_system_prompt`. The current pipeline, `runner/agentloop/loop.py`,
does not read `policy/` — its agent authors code inside the container and receives
prior-iteration history through `extra_instruction_paths`. See `README.md` and
`FLOW.md` at the repo root.

## Files

- **AGENTS.md** — Conduct constitution. 6 always-law rules, subagent doctrine, paper + idea templates, pruning procedure, SLURM conventions, git rules. Read first.
- **goal.md** — Mission context. Current frontier table, priority pivot directive, explicit named candidates (v1..v7), ruled-down list, binding speedrun rules, budget, scope.
- **plan.md** — Mutable state. Current family, stuck-detector counter, active picklist, lessons log. The agent overwrites this file.
- **scratchpad/** — Agent workspace. THREAD.md is the durable mission log; picklist.md is the working shortlist; audits.md is the rule-check + ruled-down log. `variants/`, `ideas/`, `papers/`, `runs/`, `sbatch-stubs/`, `sweeps/` are populated during the run.

## Why markdown, not code

Prime Intellect's Auto-NanoGPT project (blog: https://www.primeintellect.ai/auto-nanogpt) ran 10,428 training runs across 4 waves and found that the harness — the *policy the agent obeys* — is best expressed in prose, not code. Prose is:

- **Editable at run-time** by the agent itself (see AGENTS.md §Scratchpad).
- **Load-bearing exactly where the intelligence sits** (the rules, the doctrine, the frontier).
- **Immune to schema drift** — a new modifier doesn't require a code change.

For the planner path this folder serves, the Python in `legacy/harness2/` is thin plumbing and the intelligence is here. That trade is reversed in `runner/agentloop/`, where the loop is code and the agent's steering is generated per iteration rather than authored in markdown.

## Version lineage

Merged from Prime Intellect's `v1/claude-code/{AGENTS,goal,plan}.md` (Lawful Core rules 1–6), `v2/codex/{AGENTS,goal}.md` (SLURM/preempt operational rules, worktree conventions), and `v3/codex/goal.md` (frontier table format, ruled-down append-only pattern). Novelty gate (v1 rule 7) and rules-compliance gate (v1 rule 8) intentionally dropped — those were the failed-wave experiment.
