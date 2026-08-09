# ENGRAM augmentation backlog

GENERATED SECTION. DO NOT HAND-EDIT.

Source of truth: `memory/capabilities.yaml` in this submodule. This file is emitted from the capability block and the generated `bucket_d_status` block. It is never hand-maintained, and any drift between it and `memory/capabilities.yaml` fails closed.

This is the `harness` submodule of the bia parent project. ENGRAM seeded this harness shell at Phase G step 9 at the default posture. The authoritative ledger, projections and proof store live at the parent root, and nothing here births a difficulty record.

## Declared capabilities not yet implemented

No capability sits at `declared`. Every optional capability sits at `default`, which rests on the default definition rather than on an undefined stronger claim.

## Bucket-D instruments not yet liveness proven

Every required instrument is honestly reported as not built inside this submodule. That is a benign scaffold and never broken theater, because no row claims an implementation it cannot exercise.

| Instrument | Bytes present | Module to build | Fail-closed consequence |
|---|---|---|---|
| ingestor | no | `memory/engram/ingest.py` | no envelope is admitted here, and admission stays at the parent root |
| signature_verifier | no | `memory/engram/verify.py` | no signature is verified here |
| freshener | no | `memory/engram/fresh.py` | no lever state is computed here |
| checkpointer | no | `memory/engram/checkpoint.py` | no tree head is appended here |
| recovery_procedure | no | `memory/engram/recover.py` | no ledger is rebuilt here |
| provenance_gate | no | `memory/engram/provenance.py` | no passing disposition is reachable here |

## Named coverage gaps

| Gap | Statement | Consequence |
|---|---|---|
| `submodule-harness-shell-only` | this submodule carries the capability block and this backlog and no ENGRAM harness modules | the disposition ceiling from this block is STALE, which is the honest scaffold case and never a pass |
