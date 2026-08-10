# M4 — recurrent Torx-native core and useful compute depth Q

M4 keeps the frozen M3 structured carrier (`C=6`, `d=3`, `L_max=3`), medium
content corruption `(alpha=0.6, sigma=0.5)`, structural masking
`rho_b=rho_length=0.5`, batch size 64, Adam learning rate 0.03, 60 steps, and
the M3 h/b/length loss. Only the recurrent content core changes.

The current core is the M2 public Torx `AffineGaussianGate` recurrence. The
residual core uses the same public gate with transformed parameters:

```text
h_next = h + eta * (A_delta @ h + b_delta) + Normal(0, variance_per_step)
```

`eta` controls only the mean residual step. Noise is independent of eta. With
`noise_mode="fixed_total"`, `variance_per_step = total_variance / Q`, so the
standard deviation is `sqrt(total_variance / Q)`. The diagnostic
`fixed_per_cycle` mode leaves per-cycle variance unchanged.

Cycle conditioning, when enabled, adds one shared learned vector multiplied by
the normalized cycle index. It does not create one parameter tree per cycle.
Predicted boundary and length remain readout targets only and never affect core
routing, topology, or shapes.

The M4 gate was predeclared before the final aggregate: a tied Q>1 candidate
must beat its own same-family Q=1 reference in validation h MSE across the
three-seed aggregate, or provide a measured comparable-quality
parameter/compute advantage. A stable negative result selects `M4_Q_NOT_USEFUL`;
it does not justify adding more recurrence mechanisms.
