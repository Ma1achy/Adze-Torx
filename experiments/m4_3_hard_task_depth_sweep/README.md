# M4.3 — harder-task recurrent-depth sweep

M4.3 tests whether larger tied Q becomes useful when the task requires more
nonlinear composition. M4, M4.1, and M4.2 are preserved unchanged.

The carrier has width 64. The dynamic state is the first 8 coordinates. The
remaining coordinates contain one-hot IDs for a per-example sequence of four
fixed nonlinear operators. Those conditioning coordinates are read-only: every
physical block and every cycle preserves them exactly, while the dynamic state
can depend on them.

For sequence depth `k ∈ {4, 8, 12}`, the target is the final state of
`s[j+1] = tanh(1.25 * M[op[j]] @ s[j] + c[op[j]])`. The operator sequence is
part of the input. Intermediate states are retained for diagnostics but are not
supervised. Train and validation sequence codes are disjoint by deterministic
residue class; validation uses residue 0 and training uses residue 1. All
examples use explicit JAX PRNG keys.

The recurrent core is M4.2's public-Torx `AffineGaussianGate` plus ordinary JAX
`tanh` residual, with `L=2`, `eta=0.75`, and total nominal pre-tanh variance
0.04. The affine control removes `tanh`. Q is swept over 1, 2, 4, 8, and 12.
Parameters are tied across Q and the unique parameter count is Q-invariant.

## Primary metric and statistics

The primary endpoint error is validation mean squared error over the first eight
dynamic state coordinates at the final recurrent cycle. Intermediate score is
the same MSE after each cycle; best-intermediate error is its minimum over those
cycles. A 5% improvement means `(Q1_error - Q_error) / Q1_error >= 0.05`.

The paired difference is `delta(k,Q) = error(k,Q=1) - error(k,Q)`, so positive
means Q improves over Q=1. Confidence intervals are two-sided paired t
intervals using fixed 95% critical values: 4.303 for three seeds and 2.776 for
five seeds. Ambiguous/wide intervals are not treated as neutrality evidence.

The first budget is 120 optimizer steps with validation every 10 steps. If any
Q is still improving by more than 1% over the last 20-step validation interval,
all Q values for that depth and both core types receive the same 240-step
extension. The final decision uses the extended results in that case.

Compute-normalized improvement is endpoint improvement over Q=1 divided by
`L*Q` effective recurrent applications. Wall-clock quality reports step time,
total time, and effective applications separately; it is not treated as an
efficiency claim without a compute-matched comparison.

Run with:

```bash
python -m experiments.m4_3_hard_task_depth_sweep.run_experiment \
  --steps 120 --seeds 3
```
