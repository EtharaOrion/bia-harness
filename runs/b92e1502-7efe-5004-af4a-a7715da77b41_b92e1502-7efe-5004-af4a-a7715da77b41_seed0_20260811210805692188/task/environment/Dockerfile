# Agent environment for the track-3 optimizer search.
# Base pinned by immutable digest, never by a floating tag.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRACK3_TELEMETRY_DIR=/telemetry

WORKDIR /workspace

# Sanitization, contract 10h. The agent-facing closure must not carry version-control
# history, remotes, credential stores, or issue and pull-request threads. Nothing is
# copied from a git working tree into this image, and the checks below fail the build if
# any such carrier appears. This raises cost; it is not a secrecy claim, because the
# upstream record lineage is public and network egress is granted by requirements.
COPY runner/ /workspace/runner/
COPY train_gpt_track3.py /workspace/train_gpt_track3.py
COPY submission/ /workspace/submission/
COPY README_TASK.md /workspace/README_TASK.md

# HISTORY ABSENCE and REMOTE ABSENCE enforced at build time over the layer being created.
RUN set -eu; \
    if find /workspace -name '.git' -o -name '*.pack' -o -name 'packed-refs' | grep -q .; then \
      echo 'SANITIZATION FAIL: version-control history present'; exit 1; fi; \
    if find /workspace -name '.gitmodules' -o -name '.git-credentials' | grep -q .; then \
      echo 'SANITIZATION FAIL: remote or credential carrier present'; exit 1; fi; \
    mkdir -p /telemetry

# The telemetry directory is written by the harness runner and is not agent-writable in
# the graded configuration. Harbor mounts it read-only to the agent and read-write to the
# runner process. Ownership is asserted by the verifier rather than trusted here.
VOLUME ["/telemetry"]
