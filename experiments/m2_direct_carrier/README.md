# M2 direct-carrier reconstruction

This experiment tests a fixed-capacity continuous carrier without structure,
bytes, routing, refinement, or language data. Targets are smooth low-rank
synthetic carriers generated from three latent variables and position basis
functions. Corruption is explicitly

```text
h_corrupt = alpha * h_clean + sigma * epsilon,
epsilon ~ Normal(0, I).
```

The iterative core is a public Torx `AffineGaussianGate` over the flattened
fixed carrier. Q transitions use explicit split keys. Tied transitions reuse
one parameter tree; untied transitions use identical initial parameter trees at
each occurrence. Fixed-total-noise mode divides per-step variance by Q so
increasing Q does not automatically multiply injected variance.

The primary experiment uses direct-target identity mode. No score bridge is
used: this core is continuous and its sampled path is differentiated by the
ordinary JAX pathwise route validated in M1.

The predeclared M2 success gate is: at medium corruption, the main tied Q=4
configuration must reduce validation MSE by at least 25% relative to the
no-update baseline on each of three seeds, with finite diagnostics, decreasing
training loss, fixed shapes, and no failed recurrence/contract tests. A weaker
but stable result is reported as `M2_PASS_WITH_LIMITATIONS`; an unstable or
non-useful affine-Gaussian core is `M2_REDESIGN_CORE`.
