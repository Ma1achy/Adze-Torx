# Phase H draft/refinement engineering interface

Status: implementation-only feature branch; not scientifically validated and
not eligible for integration before Phase G passes its gates.

## Scope and invariants

`adze_t.refinement` implements the deterministic outer control sequence:

```text
shared denoiser in draft mode
  -> select
  -> reset selected carrier sites
  -> same denoiser in global-refine mode
  -> controlled structure commit boundary
  -> repeat x R
```

The controller accepts exactly one denoising callback for both modes. It keeps
committed routing fixed through each denoising call and permits structural
change only through an explicit post-trajectory commit callback. Global
refinement may propose changes everywhere, but unselected carrier content,
observations, and structure are pinned after the call.

Reset changes only selected `h` and observed/predicted structure; it never
deletes/reindexes sites and never mutates committed routing. Refine-mode
attention remains the existing global mask in `adze_t.masking`/`adze_t.dit`.

## Public engineering interfaces

- `SelectionPolicy` / `select_regions`: deterministic threshold/top-k regions.
- `reset_selected_carriers`: erase selected state without committing structure.
- `preserve_unselected_state`: exact state pinning after global computation.
- `run_draft_refinement`: shared-callback orchestration and commit boundary.
- `RefinementTrace`: R-step state and structural transition trace.
- `refinement_error_diagnostics` / `selector_diagnostics`: engineering metrics.

## Proposed shared-interface changes

No shared interface is changed on this branch. Phase G should eventually
provide an adapter implementing:

```python
CommitFn = Callable[[before, candidate, selected, refinement_step], CarrierState]
```

The adapter must apply G's frozen commit policy only at this boundary. The
local `STRUCTURE_UNKNOWN = -1` is bookkeeping only; G must freeze its encoded
UNKNOWN representation before live denoising consumes reset observations.

## Unresolved dependencies

1. Phase G commit/hysteresis API and generated-boundary repacking contract.
2. Final encoded UNKNOWN values for boundary and extent inputs.
3. First validated selector feature/calibration policy.
4. Live S-step adapter carrying `mode="refine"` and `r_index` into conditioning.
5. Reset distribution and PRNG namespace if stochastic reset is selected.

## Resource requirements for deferred validation

- Current tests use tiny eager arrays and mock callbacks only.
- After F3, run the full format/lint/type/test/boundary suite.
- After G integration, run JIT/static-shape and deterministic/Torx parity tests.
- Live H evaluation must use one shared checkpoint and paired examples/keys.
- Authoritative H training/evaluation must wait for G's required gates.

## Exact integration prerequisites

1. F3 completes and its decision/provenance are frozen.
2. G is merged and its recovery/deletion/reactivation/UNKNOWN/churn gates pass.
3. H's UNKNOWN bookkeeping is mapped to G's encoded representation.
4. The live adapter proves one parameter tree is used in both modes and operator
   keys distinguish refinement occurrences.
5. Tests prove exact unselected preservation, between-trajectory-only commits,
   and global refine attention.
6. Only then may live intervention or an H decision run. This branch makes no
   claim that live refinement works.
