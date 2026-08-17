# Phase E.1 — Controlled Computational Depth × Q

Status: Phase E.1A calibration complete; **POINTER_V0_TOO_HARD**. No
recurrence conclusion is drawn.

## Implemented

- `POINTER_V0`: `N=10`, `D_max=11`, eight queries, fixed 120-byte prompts,
  eight-byte targets.
- JAX-only deterministic generation with independent per-example mappings,
  direct composition oracle, split seeds `920/921/922`, and chance baselines
  of 0.10 byte accuracy and `1e-8` exact-sequence accuracy.
- Diagnostic-only `Q_exec=0` zero-physical-block / DiT-shell baseline and
  physical-block localization capture. Learned shell projections and
  structural embeddings remain active at Q0.
- Depth-conditioned paired Q evaluation, final-answer/intermediate frozen probe
  support, recurrent suppression, state shuffling, and depth-bucket reporting.

## Phase E.1A calibration

The infrastructure commit `c638e95` recorded an interim unresolved status
because its calibration run had not completed. E.1A resumed the durable
checkpoints and preserves that status as historical, non-final context.

- Both `P_Q1` and `P_REF`, seed `(0,0)`, resumed from durable checkpoints and
  completed 5000 steps on the deterministic 16,384-example subset.
- On the balanced 704-example validation set, `P_Q1` Q1 accuracy was 10.80%.
  `P_REF` Q0/Q1/Q2/Q3 accuracy was 10.80%/10.51%/10.78%/10.78%. All exact
  accuracies were zero and all logit nonfinite rates were zero.
- Q2/Q3 NLL approached `ln(10)` without above-chance accuracy. This is
  valid-output support/marginal learning, not pointer-state prediction.
- Fresh P_REF diagnostics memorized 1 example by step 25, 8 examples by step
  250, and a balanced 264-example set by step 1000 (98.96% byte, 93.56%
  exact). The model/path can fit pointer targets, but did not generalize from
  the calibration corpus by 5k.
- The calibration diagnosis is `POINTER_V0_TOO_HARD`. The apparent Q-dependent
  NLL improvement is only DiT output-distribution localization because pointer
  accuracy remains at chance.

Full evidence is in `pointer/CALIBRATION.md`, `pointer/calibration.json`, and
the referenced JSONL/final-evaluation files. No MC, probes, VM, unshared
controls, or expensive interventions were run. The resumable working
checkpoints remain ignored under `results/runs/phase_e_1`.
