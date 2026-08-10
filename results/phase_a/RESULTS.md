# Phase A — faithful structural mechanics

## Scope

Phase A implements the shared deterministic carrier, boundary-to-block indexing,
fixed-capacity hard packing, exact masks, and inverse unpool. Neural, Torx,
denoising, and refinement modules remain placeholders.

Historical M1–M4.5 experiment directories and measurements are preserved. Those
experiments remain records of Torx substrate and recurrent-proxy behavior; they
are not evidence about the faithful looped-DiT architecture.

## Gate A

Status: PASS (Gate A).

The gate is passed only when the complete test suite and available repository
quality checks pass, including boundary examples, overflow errors, pack/unpool
identity, inactive-site masks, and exact draft/refine connectivity.

## Validation record

- `make format-check`: passed.
- `make lint`: passed.
- `make typecheck`: passed; archival `experiments/` are excluded from active
  Pyright checking because they intentionally reference the superseded source.
- `make test`: 17 passed.
- `make test-slow`: no slow tests are present; 15 tests deselected (pytest exit
  status 5 for an empty selection).
- `make boundaries`: passed.

The scaffold reference dimensions (`C=32`, `h_dim=64`, `L_max=4`,
`M_max=32`, `K=8`) load successfully. The worked cut-after-carrier example,
pack/unpool bijection, inactive-site participation masks, draft block-causal
mask, refine global mask, and explicit overflow behavior all pass.

## Notes

The `CarrierState.predicted` field is additive and optional for compatibility
with the scaffold constructors. When present, its leading batch/carrier shape
is validated separately from committed routing state.
