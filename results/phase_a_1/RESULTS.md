# Phase A.1 — JIT safety and reproducibility hardening

## Motivation

Harden the accepted Phase A structural mechanics for compiled execution and
fresh development environments without changing architecture or semantics.

## Implementation

- `boundaries_to_blocks`, packing metadata construction, value packing, unpool,
  masks, and committed-state checks have pure-JAX execution paths.
- `BlockLayout` and `PackMetadata` are registered JAX pytrees, so compiled
  functions can return them directly.
- `build_pack_metadata_core` returns separate per-example
  `block_count_overflow` and `block_length_overflow` flags. The eager
  `build_pack_metadata` wrapper raises explicit errors from those flags.
- `M_max` and `K` remain static capacity arguments; overflow never truncates or
  silently splits a logical block.
- Pyright is included in the dev dependencies, and the Torx install target now
  matches Torx's normal pinned project dependency.
- README status and scaffold-manifest provenance messaging are current.

## Tests and validation

- `pytest -q`: 22 passed.
- `make format-check`: passed.
- `make lint`: passed.
- `make typecheck`: passed.
- `make test`: 22 passed.
- `make boundaries`: passed.
- `make test-slow`: passed; no slow tests are present, so pytest reports an
  empty selection and the Make target accepts that expected status.
- `python -m pip install -e ".[dev]"`: passed in the active environment.
- JIT tests cover block construction, metadata, pack/unpool, masks, inactive
  holes, state invariant flags, and separate overflow flags.

## Semantics and gate

No Phase A architectural semantics changed. This milestone hardens execution
and reproducibility only.

Decision: **PHASE_A_1_PASS**.

Phase B and all learned/neural modules remain unimplemented.
