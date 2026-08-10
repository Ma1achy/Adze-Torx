# M4.4 — faithful loop semantics and recurrent-state proxy

M4.4 compares fixed-horizon and fixed-eta compute-scaling semantics while
testing a richer public-Torx recurrent-state proxy. M4–M4.3 remain immutable.

The state is one flat width-92 vector, not a site-wise carrier:

```text
task state       8
mutable scratch 24
operator IDs    48  (12 positions × 4 categories)
valid mask      12
total           92
```

For k=4 and k=8, unused operator slots are zero and their mask bits are false.
For k=12 all mask bits are true. Operator IDs are multiplied by the mask before
every Torx gate. Both IDs and masks are read-only. The implementation therefore
claims dense recurrent-state mixing, not true carrier-position mixing.

Each L=2 physical block contains two public Torx `AffineGaussianGate` operations
with JAX `tanh` between them and a residual write to task+scratch only. The same
two block parameter trees are reused across Q cycles.

The two loop semantics are explicit:

```text
fixed_horizon  : x <- x + (eta / Q) * Delta(x, c)
compute_scaling: x <- x + eta * Delta(x, c)
```

Eta and the small nonzero residual initialization are identical across Q in the
compute-scaling family. Noise uses two gates × L blocks × Q cycles:
`var_per_gate = total_variance / (2*L*Q)` under fixed-total nominal noise.

The M4.3 per-example operator-composition task is reused at k=8 and k=12 in the
primary run. The focused controls also use k=8. Intermediate states are not
supervised.
