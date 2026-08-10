# M2 decision

**M2_PASS_WITH_LIMITATIONS**

## Evidence

The fixed-capacity direct carrier learns synthetic reconstruction with the
public Torx `AffineGaussianGate` core. At medium corruption, tied Q=4 reduces
validation MSE by 66.1% relative to no-update, and all three predefined seeds
pass the predeclared 25% reduction gate. Training, gradients, shapes,
reproducibility, Q=1 semantics, and public composite contracts are tested.

The result is limited because Q=4 is weaker than Q=1 and the deterministic
affine control on this toy distribution. It also worsens the low-corruption
no-update baseline, although it succeeds at medium and high corruption.
Untied recurrence supplies no reliable benefit and has a parameter-count
increase that is reported explicitly.

## Gradient and Torx scope

The core uses only public Torx APIs and ordinary JAX pathwise differentiation
through continuous Gaussian samples. The M1.5 score bridge is deliberately not
used. No private Torx access, Temper, GenJAX/ADEV, hidden conventional
denoiser, or new stochastic-autodiff mechanism was introduced.

## Recommendation for M3

Proceed to review before M3. Carry Q=1/Q=2 as stronger direct-core controls,
retain Q=4 as a stress-test configuration, and preserve fixed-total-noise.
M3 should add structural channels only after the direct content path remains a
baseline; this result is not evidence that deeper recurrence is beneficial.

Stop here; no M3 implementation is included.
