# M4 decision

**M4_Q_NOT_USEFUL**

The public Torx `AffineGaussianGate` remains trainable, and the residual
near-identity parameterisation is materially more stable than the original
current recurrence. However, no tied Q>1 configuration produced a convincing
reproducible improvement over its own same-family Q=1 reference under the
frozen M3 task, noise convention, loss, optimizer, and training budget.

The current family degraded from Q=1 h MSE `0.43709 +/- 0.00538` to
`0.47188 +/- 0.02626` at Q=2 and `0.60522 +/- 0.06053` at Q=4. The residual
family was comparatively stable (`0.43996 +/- 0.00583`, `0.43620 +/- 0.00273`,
and `0.44727 +/- 0.01112` for Q=1,2,4), but its Q=2 difference is too small
relative to seed variation to pass the predeclared gate, and Q=4 is worse.

Mean-only recurrence has the same qualitative Q trend as stochastic
recurrence, while sampled variance remains small and fixed-total nominal
variance remains `0.01`. This makes mean-dynamics/optimization drift the main
diagnosis, not missing stochastic-gradient machinery or accidental noise
growth. Untying and a shared cycle-index feature do not provide a controlled
solution.

Carry Q=1 forward as the default core for M5. Reconsider recurrent compute only
if a later milestone creates an independent architectural reason; do not begin
M5 automatically from this result.
