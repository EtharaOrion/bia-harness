"""Code-driven agentic refinement loop: the agent authors code INSIDE a container.

Formerly ``runner/track3/``. "track3" named a *benchmark*, not this code. Nothing
in this package is specific to that benchmark: it is a code-driven loop that runs
an agent inside a container and feeds the results back into the next iteration,
and it is task-agnostic. The name now describes what the code does.

Each iteration renders prior results into agent-facing markdown, injects it into a
harbor ``--config`` JSON, launches the job, locates and parses the resulting trial,
classifies the outcome, scores it, and appends one row to the campaign ledger. The
ledger is the loop state.

Run artifacts live under ``runs/agentloop/<slug>/`` (see
``agentloop.harbor_config.resolve_run_root``).

Contrast with ``legacy/harness2/`` (the old ``runner/legacy_planner/``), where a
planner LLM authors the candidate *outside* the container and the harness mounts
it in. Use this package when the container owns the candidate.
"""
