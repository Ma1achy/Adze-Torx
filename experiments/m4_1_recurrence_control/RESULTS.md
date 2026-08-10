# M4.1 RESULTS — controlled recurrence re-test

**Status:** complete, review gate reached

## Environment and frozen task

```text
Python: 3.11.8
JAX: 0.8.1
Torx distribution: extro-torx 0.0.1
Torx commit: f1fc858ed950ecd41935d15c06d0ec7c5e0674ae
Platform/device: macOS arm64 / CPU
dtype: JAX default float32
```

M3 data, corruption, structural heads, loss, fixed shape, and routing semantics
were unchanged. M4 files and numerical records were not edited. M4.1 adds only
new recurrent family names and a new experiment directory.

## Initialization and variance contract

`AffineGaussianGate` stores log variance, so M4.1 initializes
`log_var=log(0.01/Q)`. The nominal independent accumulated variance is exactly
`Q*(0.01/Q)=0.01`; eta does not modify it. Unit tests verify this contract.

| core | Q | `||F_Q(h)-h||` | `||A_eff-I||` | spectral radius | largest singular |
|---|---:|---:|---:|---:|---:|
| current | 1 | 1.035829 | 1.697056 | .600000 | .600000 |
| current | 2 | 1.657326 | 2.715290 | .360000 | .360000 |
| current | 4 | 2.253963 | 3.692794 | .129600 | .129600 |
| residual | 1 | .258957 | .424264 | .900000 | .900000 |
| residual | 2 | .492019 | .806102 | .810000 | .810000 |
| residual | 4 | .890554 | 1.459045 | .656100 | .656100 |
| identity_residual | 1 | 0 | 0 | 1 | 1 |
| identity_residual | 2 | 0 | 0 | 1 | 1 |
| identity_residual | 4 | 0 | 0 | 1 | 1 |
| q_normalized_residual | 1 | 0 | 0 | 1 | 1 |
| q_normalized_residual | 2 | 0 | 0 | 1 | 1 |
| q_normalized_residual | 4 | 0 | 0 | 1 | 1 |

The new maps are equal to identity to float32 precision before training.

## Solvable deterministic toy control

The toy uses a stable known generator `G` and target `T=expm(G)`. The exact
tied Q-step solution is `B_Q=expm(G/Q)`, so every tested Q is algebraically
solvable. Values below are final MSE after 60 Adam steps:

| core | Q=1 | Q=2 | Q=4 |
|---|---:|---:|---:|
| current | .0000895 | .0003056 | .0011170 |
| residual | .0000005 | .0000109 | .0000117 |
| identity residual | .0000076 | .0000127 | .0000181 |
| Q-normalized residual | .0000076 | .0000077 | .0000084 |

This confirms that the new parameterizations preserve a known-solvable tied
solution and that Q-normalization reduces Q-dependent toy degradation.

## Optimization diagnostics

At step 0, delta parameters have zero norm, so their update/parameter ratio is
reported as `undefined`; gradient and optimizer-update norms are reported
separately. After the first update the ratio is defined. Representative seed-0
values:

| core | Q | step | loss | total grad | core grad | update | delta update | delta update/parameter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identity residual | 1 | 0 | 3.0135 | .8723 | .3372 | .6531 | .5548 | undefined |
| identity residual | 2 | 0 | 3.0080 | 1.0467 | .6710 | .6531 | .5548 | undefined |
| identity residual | 4 | 0 | 3.0045 | 1.5675 | 1.3442 | .6531 | .5548 | undefined |
| Q-normalized residual | 1 | 0 | 3.0135 | .8723 | .3372 | .6531 | .5548 | undefined |
| Q-normalized residual | 2 | 0 | 3.0080 | .8706 | .3355 | .6531 | .5548 | undefined |
| Q-normalized residual | 4 | 0 | 3.0045 | .8735 | .3361 | .6531 | .5548 | undefined |
| identity residual | 1 | 59 | 1.2820 | .4724 | .0893 | .1303 | .0906 | .0176 |
| identity residual | 2 | 59 | 1.2680 | .4338 | .2334 | .1223 | .0819 | .0278 |
| identity residual | 4 | 59 | 1.3227 | .9461 | .8523 | .1148 | .0675 | .0398 |
| Q-normalized residual | 1 | 59 | 1.2820 | .4724 | .0893 | .1303 | .0906 | .0176 |
| Q-normalized residual | 2 | 59 | 1.2837 | .4962 | .1106 | .1267 | .0876 | .0187 |
| Q-normalized residual | 4 | 59 | 1.2892 | .5150 | .1287 | .1253 | .0841 | .0190 |

Q-normalization substantially reduces core-gradient growth at Q=4. Identity
residual still shows larger Q-dependent core gradients, while Q-normalized
gradients are much closer across Q.

## Fixed 60-step and common 240-step results

The final matrix used identical data, optimizer, LR, and seeds 0/1/2. Values
are validation h MSE. The 60-step column is taken from the common 240-step
run at step 60; the best column is the best validation checkpoint through 240
steps.

| core | Q | 60-step mean +/- std | best-through-240 mean +/- std | final-through-240 mean +/- std |
|---|---:|---:|---:|---:|
| identity residual | 1 | .437714 +/- .006647 | .429693 +/- .003406 | .429693 +/- .003406 |
| identity residual | 2 | .435809 +/- .005411 | .431213 +/- .002111 | .431213 +/- .002111 |
| identity residual | 4 | .458353 +/- .021548 | .443488 +/- .001770 | .443488 +/- .001770 |
| Q-normalized residual | 1 | .437714 +/- .006647 | .429693 +/- .003406 | .429693 +/- .003406 |
| Q-normalized residual | 2 | .439975 +/- .005263 | .431924 +/- .003794 | .431924 +/- .003794 |
| Q-normalized residual | 4 | .442772 +/- .004354 | .432582 +/- .003517 | .432582 +/- .003517 |

Best-step locations ranged from 140 to 220. All Q received the same 240
optimizer steps; deeper Q was not granted extra steps.

The original M4 references remain:

```text
current:  Q1 .437087 +/- .005377, Q2 .471882 +/- .026257, Q4 .605217 +/- .060534
residual: Q1 .439964 +/- .005833, Q2 .436202 +/- .002727, Q4 .447274 +/- .011118
```

The corrected families remove most of the original degradation, especially for
Q=4, but neither corrected Q=2 nor Q=4 beats its own Q=1 in the aggregate
extended comparison.

## LR sensitivity

Seed-0, 60-step validation h MSE under the same LR grid for every Q:

| LR | identity Q1 | identity Q2 | identity Q4 | Q-normalized Q1 | Q-normalized Q2 | Q-normalized Q4 |
|---:|---:|---:|---:|---:|---:|---:|
| .01 | .468197 | .448168 | .441632 | .468197 | .467234 | .466914 |
| .03 | .431026 | .433711 | .458077 | .431026 | .435050 | .438472 |
| .10 | .438456 | .456712 | .438282 | .438456 | .445041 | .451889 |

This is a robustness grid, not per-Q tuning. It shows that a single seed and
one LR can make Q=4 look favorable, but the multi-seed common-budget result
does not support an endpoint Q benefit.

## Trajectory-level refinement

These are seed-0, 240-step models evaluated over 256 validation examples. q=0
is the corrupted carrier. `improve fraction` is computed per example, not from
the aggregate MSE.

| core | Q | q | h MSE | delta MSE | improve fraction | b F1 | length acc | update norm | state norm | variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity residual | 2 | 0 | .97595 | 0 | 0 | .58044 | .30248 | 0 | 5.7035 | 0 |
| identity residual | 2 | 1 | .57460 | -.40134 | .9961 | .68715 | .37940 | 1.5053 | 6.8778 | .000358 |
| identity residual | 2 | 2 | .43198 | -.14263 | .8008 | .72986 | .39505 | 1.7329 | 8.4408 | .000700 |
| identity residual | 4 | 0 | .97595 | 0 | 0 | .57413 | .29857 | 0 | 5.7035 | 0 |
| identity residual | 4 | 1 | .74010 | -.23585 | .9961 | .65089 | .34550 | .7726 | 6.2635 | .000181 |
| identity residual | 4 | 2 | .56181 | -.17830 | .9805 | .68306 | .37027 | .8215 | 6.9359 | .000353 |
| identity residual | 4 | 3 | .45522 | -.10659 | .8945 | .73762 | .39374 | .8900 | 7.7186 | .000530 |
| identity residual | 4 | 4 | .44487 | -.01035 | .5508 | .75111 | .40287 | .9718 | 8.6081 | .000703 |
| Q-normalized residual | 2 | 0 | .97595 | 0 | 0 | .57325 | .30769 | 0 | 5.7035 | 0 |
| Q-normalized residual | 2 | 1 | .58443 | -.39152 | .9922 | .65903 | .37158 | 1.4419 | 6.8518 | .000354 |
| Q-normalized residual | 2 | 2 | .43328 | -.15115 | .8203 | .72993 | .40026 | 1.6924 | 8.3766 | .000702 |
| Q-normalized residual | 4 | 0 | .97595 | 0 | 0 | .57778 | .31160 | 0 | 5.7035 | 0 |
| Q-normalized residual | 4 | 1 | .75837 | -.21758 | .9961 | .62840 | .34420 | .6922 | 6.2357 | .000176 |
| Q-normalized residual | 4 | 2 | .58562 | -.17275 | .9844 | .66477 | .38070 | .7456 | 6.8591 | .000349 |
| Q-normalized residual | 4 | 3 | .47046 | -.11515 | .9219 | .72105 | .40808 | .8111 | 7.5767 | .000530 |
| Q-normalized residual | 4 | 4 | .43323 | -.03723 | .7148 | .72993 | .40156 | .8846 | 8.3866 | .000708 |

Both corrected families show progressive refinement on most examples. This is
evidence that the recurrent computation is meaningful within trajectories,
but the terminal endpoint advantage does not survive the multi-seed aggregate.

## Compute and scope accounting

Each optimizer step applies Q recurrent Torx gates; Q=1/2/4 therefore uses
1/2/4 recurrent applications per forward and backward pass. The 240-step
runs give each Q exactly 240 optimizer steps. Representative steady step times
were approximately `.00030 s` for Q=1, `.00033 s` for Q=2, and `.00038 s` for
Q=4 after compilation. Total wall time included compilation and was generally
1.5–1.8 s per run after the first compiled run; the first run was 4.18 s.
No compute-efficiency claim is made from the same-step comparison.

The score bridge was not used. No structural routing, dynamic shape, S/R
denoising, Temper, GenJAX/ADEV, hidden denoiser, private Torx API, or new Torx
limitation was introduced. M4 tests and records remain intact.
