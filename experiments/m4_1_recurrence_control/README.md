# M4.1 — controlled recurrence re-test

M4.1 is a follow-up to M4 and leaves `experiments/m4_recurrent_core/`
unchanged. It uses the frozen M3 task: `C=6`, `d=3`, `L_max=3`, medium content
corruption `h0=0.6*h_clean+0.5*epsilon`, structural masking `rho_b=rho_length=.5`,
batch size 64, Adam, learning rate `.03`, and explicit JAX keys.

The two new tied families are:

```text
identity_residual:
    h_next = h + eta * (A_delta @ h + b_delta) + epsilon

q_normalized_residual:
    h_next = h + (eta / Q) * (A_delta @ h + b_delta) + epsilon
```

Both initialize `A_delta=0` and `b_delta=0`, so their deterministic Q-step
mean map is exactly identity for Q=1,2,4. Noise is independent of eta and uses
fixed-total variance `var_step = 0.01 / Q`; Torx receives
`log_var=log(var_step)`. The original M4 `current` and `residual` families are
retained as immutable references.

M4.1 uses equal 60-step and equal 240-step optimizer budgets for every Q. It
also runs the same LR grid `.01,.03,.1` for every Q, records optimizer and
wall-clock diagnostics separately, and uses a guaranteed-solvable toy target
`T=expm(G)` with exact roots `B_Q=expm(G/Q)`.

The predeclared interpretation is narrow: if corrected Q>1 is not a convincing
same-family endpoint improvement, the result is neutral rather than a claim
that recurrence in general is not useful.
