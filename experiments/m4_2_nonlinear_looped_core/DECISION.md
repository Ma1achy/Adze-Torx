# M4.2 decision

## M4_2_Q_NEGATIVE

The public-Torx stochastic affine plus deterministic-JAX-tanh residual core was
genuinely nonlinear, stable, fixed-shape, and tied across Q. Its Q=4 trajectory
often improved at every loop, but across three seeds and task depths k=1, 2, 4,
Q=2 and Q=4 were consistently worse than the same nonlinear core at Q=1. The
affine control showed the same direction, so the result is not evidence of a
nonlinear recurrence advantage.

The M4/M4.1 records remain intact. M4.2 therefore weakens rather than supports
the original looping hypothesis for this small affine-plus-tanh Torx surrogate.
It does not generalize to a future looped-DiT-like architecture. The next
milestone must not be started automatically; review should decide whether a
different core family is justified before any routing work.
