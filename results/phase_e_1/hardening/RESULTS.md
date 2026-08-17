# Phase E.1B framework hardening

Status: **HARDENING_PASS**. This is an engineering result, not a recurrence
decision.

Base commit: `3f9867e` (`Complete Phase E.1A pointer calibration`). The
POINTER_V0 measurements and `POINTER_V0_TOO_HARD` benchmark diagnosis remain
unchanged.

## Correctness fixes

- Calibration, primary, and overfit checkpoints now have distinct benchmark /
  stage / arm / seed identities. Primary initialization cannot consume a
  calibration state.
- Every seed-dependent curve, summary, final metric, localization result,
  intervention result, and probe result includes initialization and stochastic
  training seeds in its path.
- Actual recurrence occurrence index is separate from the conditioning-cycle
  index. Depth-code overrides leave Torx occurrence keys unchanged while
  retaining independent keys across actual Q cycles.
- Stop-gradient is a forward-identical boundary intervention and no longer
  aliases recurrent-delta suppression.
- Cycle probes pool `[batch, positions, d_model]` over positions and assert
  two-dimensional `[batch, features]` inputs for `x_pre`, every `x_q`, and
  `h_hat`.
- Q0 is documented and tested as the zero-physical-block / DiT-shell baseline:
  learned shell projections and structural embeddings remain active.
- The corrected POINTER_V0 audit uses full prompt+target+depth hashes, reports
  within-split duplicates and every split intersection, and writes versioned
  evidence rather than replacing the historical audit.

## Focused contracts

- Synthetic calibration state does not affect primary scratch initialization.
- Seed 0/1 checkpoint and evidence paths do not collide.
- Correct/all-q0/reversed conditioning schedules use identical stochastic
  occurrence keys under fixed-root lambda=1 execution.
- Actual recurrence cycles retain distinct stochastic keys.
- Stop-gradient control has identical forward output and a changed gradient
  path; delta suppression changes the forward computation.
- Perturbing all physical blocks leaves Q0 unchanged; perturbing the DiT shell
  may change Q0.
- Probe matrices preserve the feature dimension.
- Complete-example hashes detect duplicates and distinguish generated splits.

## Validation

- `make format-check`: pass
- `make lint`: pass
- `make typecheck`: pass
- `make test`: 126 passed, 8 deselected
- `make boundaries`: pass
- `make test-slow`: 8 passed, 126 deselected
- private-Torx import scan: clean

No benchmark training, MC evaluation, probes, or recurrence interventions were
run during hardening.
