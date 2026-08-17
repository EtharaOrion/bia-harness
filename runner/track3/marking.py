"""Mark synthetic (fabricated) results so they can never pass as real training.

The `dry` backend in `harness.dispatch_dry` writes a val_loss curve derived from
a hash of the seed -- no model, no data, no gradients. Those numbers have the
same shape as real trainer output and have already been mistaken for a validated
run. Every artifact produced by a synthetic backend therefore carries an explicit
`is_synthetic` flag, and any surface that renders rows can ask for a banner.

Additive only: nothing here changes reward/status/backend semantics, and
`is_synthetic` is deliberately NOT a canonical ledger field (see
`ingest_result.CANONICAL_FIELDS`, fixed at 33).
"""

SYNTHETIC_BACKENDS = frozenset({"dry"})

SYNTHETIC_BANNER = (
    "*** SYNTHETIC RESULTS *** these numbers are FABRICATED by the "
    f"{'/'.join(sorted(SYNTHETIC_BACKENDS))} backend "
    "(a seed hash, no model/data/gradients) and are NOT from real training -- "
    "do not report them as validated results."
)


def is_synthetic_backend(backend: str) -> bool:
    """True when `backend` fabricates its metrics instead of training."""
    return backend in SYNTHETIC_BACKENDS


def mark_reward_payload(payload: dict, backend: str) -> dict:
    """Return a copy of `payload` tagged with `is_synthetic` for `backend`.

    The input dict is never mutated.
    """
    marked = dict(payload)
    marked["is_synthetic"] = is_synthetic_backend(backend)
    return marked


def _row_is_synthetic(row: dict) -> bool:
    if row.get("is_synthetic"):
        return True
    return is_synthetic_backend(row.get("backend"))


def synthetic_banner(rows: list[dict]) -> str | None:
    """One-line loud warning if ANY row is synthetic, else None.

    A row counts as synthetic when it carries a truthy `is_synthetic` flag or
    its `backend` is in `SYNTHETIC_BACKENDS`.
    """
    if any(_row_is_synthetic(row) for row in rows):
        return SYNTHETIC_BANNER
    return None
