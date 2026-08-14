# Verifier environment. Isolated from the agent environment by Harbor, per contract 2a.
# Base is pinned by immutable digest. A floating tag is forbidden under contract 2b and
# invariant 21, because a tag can be re-pointed and would silently change what graded.
FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1

# CPU torch only. The verifier never trains. It replays recorded logs, replays each corpus
# member against the run's own captured gradients, and asserts invariants over telemetry,
# all of which are CPU work. Keeping the verifier accelerator-free means grading cannot
# silently depend on GPU availability. There is no behavioral probe: it was removed when
# novelty measurement moved inside the graded run, and this comment described it for a
# release after it had gone.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.0 \
 && pip install --no-cache-dir pyyaml==6.0.2

WORKDIR /verifier
