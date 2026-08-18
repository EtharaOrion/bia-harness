"""Legacy "planner authors the variant" loop architecture (harness generation 2).

This is the OLD of the two loop architectures. Here a planner LLM runs *outside*
the container, authors a variant file on the host, and the harness delivers it
into the run by mounting it (``harness.run(..., variant=)`` ->
``mount_variant.mount_task``). The planner never enters the container; it only
sees the ledger rows and attempt summaries that come back out.

It serves ``nanogpt-speedrun``, whose ``/workspace`` is assembled by the harness at
dispatch time, so there is a mount point to deliver a variant to.

It CANNOT serve tasks whose ``/workspace`` ships inside the task image (the
``bia/track3nov`` family): the image already contains the workspace, so harbor
exposes no channel to hand a host-authored variant to the container. The harness
refuses that combination explicitly -- see ``harness.VariantDeliveryImpossible``.
Those tasks use ``runner/agentloop/`` instead, where the agent authors
``/workspace/submission/optimizer.py`` from *inside* the container.

Location note: this lived at ``runner/legacy_planner/`` and now lives at
``legacy/harness2/``, so that ``runner/`` holds only the current pipeline. Its own
tests moved with it, into ``legacy/harness2/tests/``. ``runner/harness.py`` stays
in ``runner/`` because it is shared by both architectures.

Import note: these modules are imported flat (``import orchestrator``,
``from summarize import ...``), not as ``harness2.orchestrator``, and they import
each other flat too. This directory is placed on ``sys.path`` by
``runner/harness.py``, ``tests/conftest.py``, ``legacy/harness2/tests/conftest.py``
and the entrypoints under ``tools/``/``scripts/``, so the flat names keep resolving
from this new home. Keep this file free of re-exports: importing the same module
under two names would load two independent copies of it.
"""
