# Phase E.1 — Controlled Computational Depth × Q

Status: Phase E.1B calibration complete; **POINTER_V0_TOO_HARD** and
**FIXED_TRANSITION_V0_TOO_HARD** are benchmark diagnoses. No recurrence
conclusion is drawn.

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

## Phase E.1B hardening

The framework was hardened before new science:

- calibration and primary checkpoints/evidence now have disjoint stage- and
  seed-specific identities;
- actual recurrence occurrence indices are separate from depth-conditioning
  indices, preserving stochastic keys under conditioning interventions;
- the stop-gradient identity control preserves forward values while cutting
  the selected backward path;
- cycle probes pool positions rather than features;
- Q0 is documented and tested as the zero-physical-block / DiT-shell baseline;
- dataset provenance uses full split hash sets and current generator metadata.

All focused and full hardening validation passed. Evidence is under
`hardening/`.

## FIXED_STATE_TRANSITION_V0

The replacement benchmark uses periodic 64-bit Rule 30 with the fixed update
`next = left XOR (center OR right)` and depths `1,2,3,4,6,8,12,16`. Each prompt
contains only a task tag, explicit depth, and random eight-byte initial state;
the target is the complete eight-byte final state. No fresh arbitrary program
is supplied through the context bottleneck.

The model-independent audit passed, but both calibration arms failed held-out
learnability through 5k on 8,192 training examples:

- T_Q1 Q1: 49.77% bit, 0.293% byte, 0% exact, NLL 6.097.
- T_REF Q0/Q1/Q2/Q3 bit: 49.97%/49.59%/49.60%/49.48%.
- T_REF Q0/Q1/Q2/Q3 byte: 0.391%/0.317%/0.317%/0.269%.
- All T_REF exact accuracies were zero; NLL was
  6.010/6.070/6.165/6.416; all nonfinite rates were zero.
- No depth bucket or Q execution count showed above-chance prediction or a
  meaningful depth gradient.

Fresh T_REF controls memorized 1 example at step 25, 8 balanced examples at
step 50, and 256 balanced examples at step 1k (99.51% byte, 96.48% exact,
NLL 0.0217). Thus the model path can represent and optimize fixed targets but
did not generalize the Rule-30 transition under the approved calibration.

The benchmark-local diagnosis is `FIXED_TRANSITION_V0_TOO_HARD`. No
authoritative run, interventions, MC confirmation, probes, replication, or
strong controls were run. Full evidence is in `fixed_transition/CALIBRATION.md`
and `fixed_transition/calibration.json`.
