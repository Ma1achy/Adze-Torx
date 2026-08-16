# Phase E.1 — Controlled Computational Depth × Q

Status: **incomplete / unresolved**. The implementation and generator audits
are complete, but the calibration evaluator did not finish within the available
runtime window. No recurrence conclusion is drawn.

## Implemented

- `POINTER_V0`: `N=10`, `D_max=11`, eight queries, fixed 120-byte prompts,
  eight-byte targets.
- JAX-only deterministic generation with independent per-example mappings,
  direct composition oracle, split seeds `920/921/922`, and chance baselines
  of 0.10 byte accuracy and `1e-8` exact-sequence accuracy.
- Diagnostic-only `Q_exec=0` bypass and computation-localization capture.
- Depth-conditioned paired Q evaluation, final-answer/intermediate frozen probe
  support, recurrent suppression, state shuffling, and depth-bucket reporting.

## Calibration progress

- `P_REF`, seed `(0,0)`, reached step 500 with finite loss `2.35130`.
- `P_Q1`, seed `(0,0)`, reached step 100 with finite loss `2.58916`.
- Evaluation artifacts from the first implementation contained an invalid
  NaN bucket aggregation and are not scientific evidence; masked aggregation
  was corrected before the final calibration rerun.
- No valid Q=0/Q=1 localization table, final MC evaluation, intervention
  table, replication, or VM result is recorded yet.

The resumable working checkpoints remain ignored under `results/runs/phase_e_1`.
