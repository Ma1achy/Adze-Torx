# Phase H draft/refinement engineering interface v1

Status: remediated branch-local engineering contract. It is not a live G/H
integration, training result, or scientific claim.

## Frozen branch-local ownership

H owns persistent carrier-content orchestration:

```text
shared denoiser in draft mode
  -> select deterministic seeds
  -> form an eligibility-safe closed region
  -> reset selected h and observed structure
  -> same denoiser in global-refine mode
  -> optional structural-only G commit
  -> validate routing and repack in the future live adapter
  -> repeat x R
```

The controller cannot delegate `h`, observed structure, predictions, or carrier
identity to a commit callback. A future G adapter may return only committed
structure and structural diagnostics through `StructuralCommitResult`.

Committed lengths are explicit at every H boundary. Prediction presence is
site-resolved through `PredictedStructure.valid`; selected predictions may
therefore emerge when the preceding state had none without inventing
unselected predictions.

## Public interfaces

- `SelectionPolicy`, `RegionSelection`, and `select_regions` distinguish the
  pre-halo seed budget from the final closed region. Nonfinite scores never
  become eligible. Inactive reactivation is an explicit policy flag.
- `reset_selected_carriers` resets selected `h`, writes valid UNKNOWN codes,
  invalidates selected old predictions, and never changes committed routing.
- `preserve_unselected_state` pins every unselected content, observation, and
  prediction field while retaining newly produced selected predictions.
- `RefinementContext`, `RefinementOccurrence`, `DenoiseEvidence`, and
  `DenoiseResult` form the compile-time boundary for the future full-S F
  adapter and expose diffusion/operator occurrence evidence.
- `StructureCommitMasks`, `StructuralCommitFn`, `StructuralCommitResult`,
  `RegionClosureFn`, and `StructureKeyFn` form the compile-time boundary for a
  future validated G adapter. H contains no edge-ownership, hysteresis,
  sampling, routing, or score-gradient implementation.
- `RefinementTrace` records selection, reset/regeneration/commit state,
  observed/predicted/committed structure, validity/overflow, churn, keys,
  S/operator evidence, support counts, and preservation checks.
- Diagnostics return support and validity. Empty support is represented by
  NaN, never a meaningful-looking zero.
- `selector_calibration` reports support, Brier score, fixed-bin ECE, and
  threshold coverage/error-capture curves independently from rollout metrics.

## Structural UNKNOWN v1

`PHASE_H_STRUCTURE_UNKNOWN_V1` defines:

- observed boundary UNKNOWN: code `2`;
- historical left-edge BOS: left-embedding row `2`, unchanged;
- left-context boundary UNKNOWN: appended left-embedding row `3`;
- observed extent UNKNOWN: code `L_max + 1`.

`migrate_model_structure_unknown_embeddings` returns a new deterministic
candidate parameter tree. Every existing row is preserved bitwise and only
one zero-initialized candidate row is appended to each structural embedding.
Accepted checkpoint files are never edited. Migration occurs before the
ordinary deterministic-to-Torx mapping so the established rho policy remains
the sole owner of stochastic parameter construction.

## Selector mechanics frozen for engineering tests

- Every policy explicitly names budget scope; there is no default.
- `budget_scope="seed"` caps threshold-passing seeds before halo expansion.
- `budget_scope="expanded"` retains the capped seeds and deterministically
  admits the highest-score, lowest-index eligible halo sites until the final
  closed-region cap is reached.
- Equal scores prefer the lower carrier index deterministically.
- The closed region may exceed the seed budget.
- Eligibility is reapplied after halo expansion.
- Inactive sites remain ineligible unless
  `allow_inactive_reactivation=True`; even then nonfinite sites remain
  ineligible.

Traces retain seed/closed masks, counts, scope, and limit. These mechanics are
an engineering protocol, not approved selector thresholds or a scientific H
policy.

## Live dependencies intentionally unresolved

Validated G must supply:

1. topology-safe region closure and separate boundary-edge/extent-site masks;
2. a structural-only result satisfying the H protocol;
3. fail-closed validated routing and overflow evidence;
4. namespaced structure keys bound to example/execution/R/S/commit identity;
5. frozen edge ownership, commit/hysteresis, reactivation, and score-gradient
   semantics;
6. a validated repack result consumed by the next R application.

The full-S F adapter must supply one shared parameter tree, actual-S and
conditioning coordinates, exact diffusion/operator keys, Q/layer/scope
coordinates, model-application counts, draft block-causal behavior, and refine
global behavior.

Until those adapters exist and their real-model tests pass, this branch cannot
claim shared learned weights, global refinement, valid generated repacking,
deterministic/Torx parity, or H scientific utility.

## Integration order and recovery

1. Preserve `120767a05ba9af60c98513b9f3bd63195b7d6c01` as the pre-remediation H
   rollback point.
2. F3 completion at main `2a0cab0c9c647832596013d56037c842ff3844ea` is merged.
3. G must be corrected, integrated, and pass its separately authorized gates.
4. Adapt H to the accepted G types without copying G policy into H.
5. Run real F/G/H model-path, JIT/static-shape, public-Torx parity, and complete
   occurrence tests.
6. Only then may an H experiment be proposed. No training is authorized by
   this engineering record.
