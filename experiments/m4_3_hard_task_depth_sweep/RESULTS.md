# M4.3 results

## Environment and controls

- Python 3.11.8; JAX 0.8.0; CPU `TFRT_CPU_0`; float32
- Torx `extro-torx`, commit `f1fc858ed950ecd41935d15c06d0ec7c5e0674ae`
- width 64; dynamic state dimension 8; four operators; physical depth L=2
- eta 0.75; total nominal pre-tanh variance 0.04; Adam learning rate 0.03
- batch size 64; seeds 0, 1, 2; Q = 1, 2, 4, 8, 12

The primary metric is final-cycle validation MSE over the first eight dynamic
coordinates. Best-intermediate MSE uses the same metric at the best cycle.
Paired intervals use the fixed paired t procedure in README.md.

## Mechanical Q gate

All Q values passed the pre-run checks. The nonlinear and affine controls had
the same values.

| Q | parameters | initial deterministic error | nominal variance | initial gradient norm | initial Adam update norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 8448 | 0.0 | 0.04 | 0.4264 | 0.9675 |
| 2 | 8448 | 0.0 | 0.04 | 0.4264 | 0.9675 |
| 4 | 8448 | 0.0 | 0.04 | 0.4264 | 0.9675 |
| 8 | 8448 | 0.0 | 0.04 | 0.4264 | 0.9675 |
| 12 | 8448 | 0.0 | 0.04 | 0.4264 | 0.9675 |

Conditioning error was exactly zero after every traced physical block and cycle,
while a coupling test confirmed dynamic state dependence on conditioning slots.
Measured post-stack variance was reported separately because fixed nominal
pre-tanh variance does not imply fixed post-tanh variance.

## 120-step adequacy

Every primary configuration was still improving materially at step 120 under
the predeclared 1%/20-step rule. The 120-step endpoints were therefore not used
for the architectural decision. Every Q, task depth, and core type was rerun
with the same common 240-step budget.

## Extended endpoint MSE

Best validation MSE over 240 steps, mean ± population standard deviation over
three seeds:

| core | k | Q=1 | Q=2 | Q=4 | Q=8 | Q=12 |
|---|---:|---:|---:|---:|---:|---:|
| nonlinear | 4 | 0.01282 ± 0.00051 | 0.01272 ± 0.00050 | 0.01286 ± 0.00054 | 0.01297 ± 0.00055 | 0.01302 ± 0.00056 |
| nonlinear | 8 | 0.01743 ± 0.00047 | 0.01970 ± 0.00015 | 0.02124 ± 0.00008 | 0.02203 ± 0.00013 | 0.02228 ± 0.00016 |
| nonlinear | 12 | 0.02113 ± 0.00049 | 0.02462 ± 0.00020 | 0.02658 ± 0.00022 | 0.02760 ± 0.00021 | 0.02797 ± 0.00022 |
| affine | 4 | 0.01207 ± 0.00070 | 0.01186 ± 0.00063 | 0.01192 ± 0.00047 | 0.01196 ± 0.00047 | 0.01199 ± 0.00046 |
| affine | 8 | 0.01358 ± 0.00059 | 0.01443 ± 0.00070 | 0.01536 ± 0.00042 | 0.01588 ± 0.00037 | 0.01605 ± 0.00035 |
| affine | 12 | 0.01438 ± 0.00042 | 0.01594 ± 0.00088 | 0.01751 ± 0.00038 | 0.01824 ± 0.00037 | 0.01856 ± 0.00031 |

## Paired depth interaction

`delta = Q1_error - Q_error`; positive is an improvement. Values below are
extended-budget paired means with 95% paired t intervals.

| core | Q | delta k=4 | delta k=8 | delta k=12 |
|---|---:|---:|---:|---:|
| nonlinear | 2 | +0.00011 [-0.00001,+0.00020] | -0.00227 [-0.00328,-0.00127] | -0.00349 [-0.00439,-0.00259] |
| nonlinear | 4 | -0.00003 [-0.00017,+0.00010] | -0.00380 [-0.00515,-0.00246] | -0.00545 [-0.00668,-0.00423] |
| nonlinear | 8 | -0.00014 [-0.00028,-0.000002] | -0.00460 [-0.00580,-0.00341] | -0.00647 [-0.00770,-0.00524] |
| nonlinear | 12 | -0.00019 [-0.00034,-0.00005] | -0.00485 [-0.00608,-0.00362] | -0.00684 [-0.00820,-0.00548] |
| affine | 8 | +0.00011 [-0.00058,+0.00081] | -0.00229 [-0.00298,-0.00160] | -0.00386 [-0.00439,-0.00333] |
| affine | 12 | +0.00008 [-0.00065,+0.00082] | -0.00246 [-0.00329,-0.00164] | -0.00419 [-0.00471,-0.00366] |

At k=8, nonlinear Q=8 and Q=12 were approximately 26% and 28% worse than
Q=1. At k=12 they were approximately 31% and 32% worse. All corresponding
paired intervals exclude zero. The effect is stronger at harder depth.

## Trajectory and compute diagnostics

For a representative extended-budget nonlinear k=8 validation batch, Q=8 MSE
was 0.24835, 0.18883, 0.14090, 0.10330, 0.07467, 0.05360, 0.03872,
0.02869, and 0.02233 at cycles 0 through 8. Q=12 ended at 0.02249 after
monotonic improvement from 0.24835. Q=1 ended at 0.01781 on the same diagnostic
batch. Deep Q therefore improved intermediate states without improving the
endpoint, and did not achieve a 5% best-intermediate improvement.

Nonlinear k=8 steady step time was approximately 0.00044s at Q=1, 0.00212s at
Q=8, and 0.00318s at Q=12. Effective recurrent applications were `2Q` per
optimizer step. Full train curves, best step, final step, runtime, and finite
counts are in `raw_results.json`.

No score bridge was used. No private Torx API, routing, S, M5, Temper, or
GenJAX functionality was added. M4, M4.1, and M4.2 records remain unchanged.
