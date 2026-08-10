# M4 RESULTS — recurrent Torx-native core and useful compute depth Q

**Status:** complete, review gate reached

## Environment and frozen setup

```text
Python: 3.11.8
JAX: 0.8.1
Torx distribution: extro-torx 0.0.1
Torx commit: f1fc858ed950ecd41935d15c06d0ec7c5e0674ae
Platform/device: macOS arm64 / CPU
dtype: JAX default float32 for M4 experiment arrays
```

The M3 generator and corruption are unchanged: `C=6`, `d=3`, `L_max=3`,
medium content corruption `h0=0.6*h_clean+0.5*epsilon`, and independent
boundary/length UNKNOWN masking at 0.5. Training uses batch size 64, Adam
learning rate 0.03, 60 steps, and seeds 0/1/2 for the final comparison.
The core is public Torx `AffineGaussianGate`; the structural heads are the
small M3 shared affine readouts. Full-model parameter count is 474 for tied
unconditioned runs, 492 with the shared cycle-bias vector, 834 for Q=2
untied, and 1554 for Q=4 untied.

The predeclared gate was: Q>1 must beat its own same-family Q=1 validation h
MSE across the three-seed aggregate, or show a controlled comparable-quality
parameter/compute advantage. No unrelated data, corruption, loss, optimizer,
or training-budget retuning was allowed.

## Fixed-total-noise contract

The implementation stores Torx `log_var`, where `exp(log_var)` is variance.
For fixed-total mode it initializes each cycle with:

```text
variance_per_step = 0.01 / Q
log_var = log(variance_per_step)
nominal sum of independent variances = Q * variance_per_step = 0.01
```

Unit tests verify this variance contract for Q=1,2,4. The residual `eta` is
not applied to noise. A fixed-per-cycle diagnostic was also run; it is not the
primary comparison.

## Final three-seed comparison

Each row is validation h MSE, boundary UNKNOWN F1, and length UNKNOWN
accuracy after 60 steps. Values are mean +/- sample standard deviation across
seeds 0,1,2.

| core | Q | tied | eta | h MSE | b UNKNOWN F1 | length UNKNOWN acc | runtime notes |
|---|---:|:---:|---:|---:|---:|---:|---|
| current | 1 | yes | — | 0.437087 +/- 0.005377 | 0.741770 +/- 0.020891 | 0.417538 +/- 0.008366 | 0.55–1.94 s |
| current | 2 | yes | — | 0.471882 +/- 0.026257 | 0.737692 +/- 0.001250 | 0.411059 +/- 0.016813 | 0.57–0.65 s |
| current | 4 | yes | — | 0.605217 +/- 0.060534 | 0.731267 +/- 0.019205 | 0.411862 +/- 0.011463 | 0.62–0.70 s |
| residual | 1 | yes | 0.25 | 0.439964 +/- 0.005833 | 0.748103 +/- 0.014061 | 0.406572 +/- 0.006215 | 0.56–0.64 s |
| residual | 2 | yes | 0.25 | 0.436202 +/- 0.002727 | 0.749267 +/- 0.016891 | 0.413620 +/- 0.008860 | 0.53–0.61 s |
| residual | 4 | yes | 0.25 | 0.447274 +/- 0.011118 | 0.737968 +/- 0.010617 | 0.414822 +/- 0.007044 | 0.98–1.47 s |

Residual Q=2 is numerically below residual Q=1 by 0.00376, but the effect is
smaller than the combined seed variation and does not establish a robust
multi-seed improvement. Residual Q=4 is worse than residual Q=1. The
predeclared Q>1 quality gate therefore does not pass.

## Original-core diagnosis

Representative trained seed-0 dynamics on one fixed validation corruption:

| core | Q | stochastic? | terminal h MSE | terminal mean h MSE | update norm | state norm | sampled variance | nominal variance |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| current | 1 | yes | 0.44697 | 0.42680 | 2.75974 | 7.44811 | 0.00351 | 0.01000 |
| current | 2 | yes | 0.22835 | 0.24024 | 1.33272 | 7.11714 | 0.00363 | 0.01000 |
| current | 4 | yes | 0.55494 | 0.57498 | 0.60491 | 6.91584 | 0.00375 | 0.01000 |
| residual | 1 | yes | 0.40350 | 0.39089 | 2.53375 | 7.36080 | 0.00348 | 0.01000 |
| residual | 2 | yes | 0.17355 | 0.18146 | 1.44652 | 7.44678 | 0.00352 | 0.01000 |
| residual | 4 | yes | 0.25264 | 0.27307 | 0.70911 | 7.44262 | 0.00375 | 0.01000 |

The stochastic and analytic mean paths show the same qualitative Q behavior;
sampled variance is small and nominal total variance is held constant. This
rules out stochastic variance accumulation as the main explanation. Current
Q=4's mean path is already worse than its Q=2 path. Residual dynamics improve
the per-cycle trajectory but do not turn extra Q into a reproducible terminal
training benefit.

## Eta, tying, and cycle-conditioning ablations

Seed-0 residual Q=4 eta sweep:

| eta | h MSE | b UNKNOWN F1 | length UNKNOWN acc |
|---:|---:|---:|---:|
| 0.10 | 0.43752 | 0.72936 | 0.40678 |
| 0.25 | 0.44375 | 0.72639 | 0.40678 |
| 0.50 | 0.45708 | 0.72040 | 0.39374 |
| 1.00 | 0.63789 | 0.74232 | 0.40026 |

With eta fixed to the primary value per family, seed-0 tied/untied results:

| core | Q | tied | cycle conditioning | h MSE | b UNKNOWN F1 | length UNKNOWN acc | params |
|---|---:|:---:|:---:|---:|---:|---:|---:|
| current | 2 | yes | no | 0.47819 | 0.73892 | 0.39505 | 474 |
| current | 2 | no | no | 0.45930 | 0.72512 | 0.39765 | 834 |
| current | 4 | yes | no | 0.63789 | 0.74232 | 0.40026 | 474 |
| current | 4 | no | no | 0.52726 | 0.72437 | 0.40808 | 1554 |
| residual | 2 | yes | no | 0.43464 | 0.73227 | 0.41199 | 474 |
| residual | 2 | no | no | 0.43417 | 0.72893 | 0.41330 | 834 |
| residual | 4 | yes | no | 0.44375 | 0.72639 | 0.40678 | 474 |
| residual | 4 | no | no | 0.44520 | 0.72457 | 0.40287 | 1554 |
| residual | 2 | yes | yes | 0.43391 | 0.73563 | 0.40808 | 492 |
| residual | 4 | yes | yes | 0.44556 | 0.72816 | 0.40287 | 492 |

Untying helps current recurrence diagnostically but carries a large parameter
increase and does not rescue the residual family. The shared cycle feature
adds 18 parameters and does not improve residual Q=4.

Fixed-per-cycle-noise seed-0 controls were also worse with Q for both current
and residual families. For residual, h MSE was 0.43372, 0.43704, 0.45075 for
Q=1,2,4; fixed-total values were 0.43372, 0.43464, 0.44375. The negative Q
trend is therefore not an artifact of accidentally using standard-deviation
scaling.

## Per-cycle trajectory

The following is a representative seed-0 Q=4 trajectory. The input is the same
for all cycles and q=0 is the corrupted carrier.

| core | q | h MSE | mean h MSE | update norm | state norm | variance |
|---|---:|---:|---:|---:|---:|---:|
| current | 0 | 0.69055 | 0.69055 | 0.00000 | 4.25041 | 0.00000 |
| current | 1 | 0.65891 | 0.65947 | 1.13019 | 4.34879 | 0.00092 |
| current | 2 | 0.48944 | 0.47912 | 0.74472 | 4.68626 | 0.00128 |
| current | 3 | 0.42707 | 0.40538 | 0.63636 | 5.16114 | 0.00193 |
| current | 4 | 0.36444 | 0.34844 | 0.64305 | 5.67364 | 0.00307 |
| residual | 0 | 0.69055 | 0.69055 | 0.00000 | 4.25041 | 0.00000 |
| residual | 1 | 0.53460 | 0.53146 | 0.50986 | 4.60196 | 0.00074 |
| residual | 2 | 0.39745 | 0.39386 | 0.55534 | 4.97720 | 0.00117 |
| residual | 3 | 0.28329 | 0.28319 | 0.62373 | 5.49483 | 0.00180 |
| residual | 4 | 0.20125 | 0.20740 | 0.65952 | 6.04848 | 0.00303 |

On this example both cores progressively improve, but state norms drift upward
and the aggregate trained endpoint is not consistently improved. The current
Q=4 path has the largest mismatch between trajectory improvement and final
multi-seed quality, consistent with optimization/distribution drift rather
than a public Torx gradient failure.

## Scope and limitations

The score bridge was not used: the M4 core is continuous and uses ordinary JAX
pathwise differentiation through public Torx sampling. No private Torx API,
Temper, GenJAX/ADEV, dynamic shape, structural routing, or new stochastic
gradient mechanism was introduced. No new Torx limitation appeared.

All M1–M3 tests remained unchanged and were run as regressions. M4's new
tests cover public sampling, residual math, variance scaling, mean-path
diagnostics, cycle conditioning, finite gradients, and tiny training.
