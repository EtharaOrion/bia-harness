# bia/minicalc agent + verifier image. One image serves both, because
# [verifier].environment_mode defaults to 'shared' and grade.py is stdlib-only.
#
# Pinned by immutable digest, not by a floating tag: a tag can be re-pointed and
# would silently change what graded.
FROM python:3.11-slim@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1

# WHY NODE AND CLAUDE-CODE ARE BAKED IN.
# harbor's claude-code agent (harbor/agents/installed/claude_code.py::install) opens
# with `command -v claude` and RETURNS IMMEDIATELY if it finds one; only otherwise
# does it apt-install nodejs/npm and then `npm install -g @anthropic-ai/claude-code`.
# The bia/track3nov image bakes node+npm to survive that setup step inside harbor's
# agent-setup timeout. This image goes one further and bakes the CLI too, so setup is
# a no-op and `agent_setup_timeout_multiplier` can stay at its default 5.0 -- the
# point of this task is a fast loop, and an npm install per iteration is most of the
# startup cost. It also means the run does not depend on npm being reachable through
# the allowlist the [agent] phase drops to.
ENV CC_PKG=/usr/local/lib/node_modules/@anthropic-ai/claude-code

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl procps nodejs npm \
 && npm install -g @anthropic-ai/claude-code \
 && npm cache clean --force \
 && apt-get purge -y --auto-remove \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf "$CC_PKG/node_modules/@anthropic-ai/claude-code-linux-x64-musl" \
 && claude --version
# The musl prune and `apt-get install` MUST share one RUN. npm unpacks both linux
# prebuilts, glibc and musl, at ~310 MB each, and this debian/glibc base can never
# execute the musl one -- but a delete in a LATER layer only whites the files out and
# leaves the image size unchanged (measured: 1.46 GB either way). Same layer: 1.04 GB.
# NOT pruned: bin/claude.exe. The name says windows and `du` bills it at 314 MB, but
# it is a HARDLINK to claude-code-linux-x64/claude (one inode, nlink 2) and is what
# /usr/local/bin/claude resolves to -- deleting it uninstalls the CLI, which is how
# this was found. `claude --version` runs in the same RUN so a future change to that
# layout fails the BUILD rather than the first trial.

# /workspace ships INSIDE the image, exactly as bia/track3nov does. That is what makes
# this task agentloop-only: there is nothing for the legacy planner path to mount a
# --variant into, and runner/harness.py raises VariantDeliveryImpossible if asked.
WORKDIR /workspace
COPY runner/ /workspace/runner/
RUN mkdir -p /workspace/submission/logs \
 && chmod -R a-w /workspace/runner

# No optimizer.py is shipped. An empty submission must fail closed: train_mini.py
# raises SubmissionError and grade.py emits telemetry_absent.
CMD ["/bin/bash"]
