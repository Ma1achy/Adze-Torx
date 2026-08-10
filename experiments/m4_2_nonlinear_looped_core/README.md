# M4.2 — nonlinear looped core fidelity test

This experiment tests whether tied recurrent compute becomes useful when the
Torx-native carrier update is genuinely nonlinear and has multiple distinct
physical blocks. It is a follow-up to M4 and M4.1; those records are not
rewritten.

The core is explicitly a **public-Torx stochastic affine transform followed by
an ordinary JAX `tanh` nonlinear residual**. The public Torx inspection found no
documented nonlinear continuous factor suitable for this role. Each physical
block owns an `AffineGaussianGate` with distinct parameters. A cycle applies
all `L` blocks, and the same stack is repeated `Q` times. Parameters are tied
across Q and never cloned as Q changes.

The residual mean update is

```text
x <- x + eta/(L*Q) * tanh(A_l x + b_l)
```

The Gaussian variance is independently set to
`total_variance/(L*Q)`. This fixes nominal pre-tanh variance, not post-tanh
output variance; measured output variance is reported separately. Parameters
start with `A=b=0`, so the deterministic mean map is identity for every Q.

The primary compositional task samples `s0` and a per-example sequence of
operator IDs. Operators select one of three fixed nonlinear maps
`tanh(M_r s + c_r)`. The carrier input contains `s0` and one-hot operator IDs,
but no intermediate states. Validation holds out complete operator
combinations for k≥2; k=1 includes all individual operators so that it does
not accidentally test unseen operators. Only the final state is supervised;
intermediates are diagnostic only.

The secondary task is the frozen M3 structured-carrier reconstruction task.
All stochastic execution uses explicit JAX keys. The score bridge is not used.

Run the matrix with:

```bash
python -m experiments.m4_2_nonlinear_looped_core.run_experiment \
  --steps 60 --seeds 3
```
