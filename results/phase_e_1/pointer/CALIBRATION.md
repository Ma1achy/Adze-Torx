# Phase E.1A POINTER_V0 calibration

Calibration diagnosis: **POINTER_V0_TOO_HARD**.

This is a benchmark diagnosis, not a Phase E.1 recurrence decision. Neither
`P_Q1` nor `P_REF` learned held-out pointer-state prediction by step 5000.
Fresh `P_REF` models nevertheless memorized fixed pointer sets, so the result
is not a gross generator, representation, or optimization-path failure.

## Setup

- Training subset: first 16,384 deterministic examples from POINTER_V0 train
  seed 920.
- Validation: 704 examples from seed 921, exactly 64 per depth.
- Seeds: initialization 0, stochastic training 0.
- Training: lambda 1, sigma 1e-3; evaluation: lambda 0.
- Chance: 10% per byte and 1e-8 for an exact eight-byte sequence.

## Final held-out metrics by depth

All exact-sequence accuracies were zero. All reported logit nonfinite rates
were zero. Cells are `byte accuracy / byte NLL`.

| d | P_Q1 Q1 | P_REF Q0 | P_REF Q1 | P_REF Q2 | P_REF Q3 |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.74% / 2.2962 | 14.65% / 2.3140 | 12.30% / 2.2974 | 11.52% / 2.2959 | 11.52% / 2.2958 |
| 2 | 8.59% / 2.3081 | 8.20% / 2.3239 | 10.35% / 2.3080 | 7.81% / 2.3065 | 7.81% / 2.3063 |
| 3 | 8.59% / 2.3091 | 10.35% / 2.3229 | 10.35% / 2.3085 | 8.79% / 2.3078 | 8.79% / 2.3077 |
| 4 | 11.33% / 2.3021 | 12.50% / 2.3189 | 11.52% / 2.3025 | 11.91% / 2.3010 | 11.91% / 2.3007 |
| 5 | 10.94% / 2.3050 | 9.96% / 2.3205 | 8.59% / 2.3043 | 10.74% / 2.3030 | 10.74% / 2.3029 |
| 6 | 9.77% / 2.3066 | 10.74% / 2.3203 | 8.20% / 2.3050 | 9.57% / 2.3039 | 9.57% / 2.3038 |
| 7 | 12.50% / 2.3033 | 10.74% / 2.3182 | 10.16% / 2.3032 | 11.72% / 2.3025 | 11.72% / 2.3026 |
| 8 | 13.28% / 2.3027 | 11.13% / 2.3198 | 12.11% / 2.3039 | 13.48% / 2.3025 | 13.48% / 2.3022 |
| 9 | 9.18% / 2.3062 | 9.38% / 2.3236 | 8.98% / 2.3071 | 8.79% / 2.3052 | 8.79% / 2.3047 |
| 10 | 11.52% / 2.3023 | 11.72% / 2.3140 | 12.50% / 2.3001 | 11.91% / 2.2998 | 11.91% / 2.3001 |
| 11 | 12.30% / 2.3065 | 9.38% / 2.3218 | 10.55% / 2.3058 | 12.30% / 2.3042 | 12.30% / 2.3037 |
| **all** | **10.80% / 2.3044** | **10.80% / 2.3198** | **10.51% / 2.3042** | **10.78% / 2.3029** | **10.78% / 2.3028** |

There is no monotone or otherwise meaningful held-out depth gradient. The
small per-bucket deviations are compatible with chance variation over 512
target bytes per bucket.

## Training and gradient curves

Values are the scheduled minibatch metrics. Restored step-100/250/500 entries
come from the durable original training records when the E.1A resume record
has no train payload.

| model | step | loss | train byte acc | raw grad norm | permitted grad norm | applied/clipped norm |
|---|---:|---:|---:|---:|---:|---:|
| P_Q1 | 100 | 2.5892 | 15.23% | 3.6143 | — | — |
| P_Q1 | 250 | 2.3666 | 10.16% | 4.1634 | 4.1608 | 1.0000 |
| P_Q1 | 500 | 2.3459 | 7.42% | 2.5175 | 2.4872 | 1.0000 |
| P_Q1 | 1000 | 2.3395 | 12.11% | 2.2531 | 2.2526 | 1.0000 |
| P_Q1 | 2000 | 2.3279 | 10.16% | 1.2891 | 1.2884 | 1.0000 |
| P_Q1 | 5000 | 2.3138 | 14.06% | 0.4389 | 0.4378 | 0.4378 |
| P_REF | 100 | 2.6405 | 8.20% | 4.4908 | — | — |
| P_REF | 250 | 2.3626 | 10.94% | 2.4060 | — | — |
| P_REF | 500 | 2.3513 | 7.03% | 2.4019 | — | — |
| P_REF | 1000 | 2.3373 | 12.89% | 1.8529 | 1.8510 | 1.0000 |
| P_REF | 2000 | 2.3282 | 9.77% | 1.0993 | 1.0982 | 1.0000 |
| P_REF | 5000 | 2.3092 | 13.67% | 0.3889 | 0.3873 | 0.3873 |

Sequence accuracy was zero at every listed calibration checkpoint. Losses and
gradients were finite; the final deterministic logit nonfinite rate was zero
for every model/Q/depth cell.

## Output support versus pointer learning

`ln(10) = 2.302585...`. By step 5000, P_REF Q2/Q3 NLL was
2.3029/2.3028 while byte accuracy was 10.78%. P_Q1 NLL was 2.3044 with
10.80% accuracy. This is consistent with learning that valid answers lie in
bytes 0..9 and matching their marginal distribution. It is not evidence that
the model predicts the correct composed pointer state.

Q execution did materially improve the output distribution earlier in
training (for example, P_REF step 1000 NLL was 4.1667/2.5660/2.3180/2.3170
for Q0/Q1/Q2/Q3), but accuracy remained at chance. This is **DiT
output-distribution localization**, not pointer-computation localization.

## Fixed-set overfit gate

| case | examples | balanced depth coverage | step | byte acc | exact acc | NLL | nonfinite |
|---|---:|---|---:|---:|---:|---:|---:|
| one | 1 | depth 6 | 25 | 100% | 100% | 0.000175 | 0 |
| few | 8 | depths 1,2,4,6,8,9,10,11 | 250 | 100% | 100% | 0.000021 | 0 |
| small | 264 | 24 per depth | 1000 | 98.96% | 93.56% | 0.03552 | 0 |

The fixed-set results demonstrate strong target memorization through the same
generator/oracle and faithful P_REF model/training path. Therefore the held-out
failure is classified as difficult generalization under this calibration
regime, not as evidence about recurrence and not as a gross inability to fit
pointer targets.

## Gate result

POINTER_V0 fails the E.1A pass criteria because it has no held-out task
learning, no functioning depth gradient, and no pointer-computation
localization. It is therefore **POINTER_V0_TOO_HARD**. No authoritative
pointer run, MC evaluation, probes, interventions, VM run, or unshared control
was started.
