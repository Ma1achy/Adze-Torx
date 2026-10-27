# G2.1 — independent gradient-oracle adjudication

**G21_GRADIENT_ESTIMATOR_CONFIRMED**, under the prospectively frozen finite-
distribution procedure. Original G23 V0 remains **PHASE_G_BLOCKED**; its failed
component and manifest were not changed, waived or replaced.

## Prospective provenance

Implementation/protocol committed at `5676402` before new results; identity
committed at `bb13056` before execution. Full identities are retained in
`IDENTITY.json` and Git history. Binding:
`9f0ae45d25a0d09c8ae9927f5d209c4bf9a805634f947086347bbffb9e714375`.
Protocol SHA-256:
`27c45031d3cc5aea0e5db206003b5ae10025ffb3b9246c5371e9b1c6e6410f86`.
No model parameter, codec, loss, baseline, sampler, score bridge, task or
scientific effect threshold changed in adjudication. No neural training ran.

## Original failure: reproduced, not erased

Fixture `peaked`, extent logit site0/class2. Temperature1 product law: two
binary nonterminal boundaries and three three-class extents. Terminal fixed1,
excluded from random occurrences/log probability; mechanical capacity cuts
are deterministic downstream routing, not semantic actions.

| Quantity | Original result |
|---|---:|
| Exact gradient (original float32) | 0.000002794152806018 |
| Independent float64 exact gradient | 0.000002794152179278 |
| Original MC mean | -0.001080604037270 |
| Empirical root SE | 0.000243438757025 |
| Error / empirical root SE | 4.450393401 |
| Frozen limit | 4.0 — NONPASS retained |

All 16 original root gradients, not just the selected values, are in `audit.json`
and `roots/original/`. For this coordinate roots 10/13 were approximately
+0.00141390; the other 14 approximately -0.00143696. Two rare actions were seen
versus 8.079 expected. The empirical variance was 23.10% of exact mean variance;
the two roots account for 87.50% of observed squared deviations. Root skew 2.059.
This is a sparse, highly non-Gaussian set of root estimates.

## Independent correctness audit

Independent NumPy float64 enumeration implements all 108 actions, normalized
softmax product probabilities, dynamic capacity routing, global emitted-byte
alignment, missing/extra costs, query CE and analytic score/pathwise derivatives.
Central differences check all 14 parameter coordinates in each fixture. All
per-action production losses and public-factor log probabilities agree within
the prospectively specified float32 tolerances; expected production bridge
gradients agree with the independent oracle in all 42 coordinates.

The byte/query loss semantics are the actual production `aligned_metrics`
semantics, including zero/missing/extra emissions. The declared quadratic
direct-theta fixture is NOT the numerical latent MSE of the full neural model;
it isolates a continuous dependency. Complete-network new-task gradient,
connection and optimizer gates remain mandatory before G2 updates.

All six mutations were detected across the complete family: omitted scores,
duplicated scores, reversed sign, detached continuous descendant, omitted first
boundary and omitted first extent. Action-independent baseline identities pass;
the production baseline remains zero. Native/local double ownership is rejected.
Masked actions have zero selected LP after mask application; forced terminal
never draws/scores. Existing structural regressions cover masks/reactivation/
occurrence coordinates. Exact score covariance rank is 9, not 14 independent
coordinates; categorical logit constraints create strong correlations. The new
bound conservatively retains all 42 comparisons rather than reducing the family.

## Old gate calibration under a known-correct oracle

2048 independent simulated full-family experiments, preserving within-action
component correlations, with the original16 × 4,096 design:

- False-rejection frequency: **140/2048 =6.83594%**.
- 95% Wilson interval: **5.82182–8.01168%**.
- Identical frequency against original float32 and independent float64 targets.
- A four-empirical-SE cutoff is not a Gaussian tail probability when root
  means are sparse/non-normal; moreover42 comparisons are made together.
  The old protocol specified no nominal family alpha. It would be incorrect
  to invent one retrospectively or assume independent scalar coordinates.

The root sample-variance formula itself is not arithmetically wrong: under iid
sampling it is unbiased in expectation. Its realized underestimation here and
the uncalibrated studentized/multiple-comparison gate explain why a known-
correct estimator can be rejected. This is evidence consistent with an expected
statistical false rejection (C), not demonstrated stochastic gradient bias (A)
or a coding defect in the variance formula (B). It does not identify a unique
causal decomposition of multiplicity versus non-Gaussian studentization.

## Independent confirmation: complete affected family

Unused namespace `g21_confirmation_v1:<fixture>`, roots 0..63;4096 draws/root
for EACH of uniform/asymmetric/peaked:786432 total production draws. All 42
coordinates pass the registered family-wise Bernstein acceptance region, using
exact finite-action variances/ranges and alpha=.01. No root exclusion, optional
stopping, baseline change or sample escalation. Maximum error/allowed-radius
ratios: uniform 0.36425, asymmetric 0.49975, peaked 0.30008.

| Selected coordinate | Independent confirmation |
|---|---:|
| Exact gradient | 0.000002794152179278 |
| MC mean | 0.000211189297261 |
| Absolute error | 0.000208395145081 |
| Empirical root SE | 0.000267915130850 |
| Exact-variance SE | 0.000253247846461 |
| Error / exact-variance SE | approximately 0.823 |
| Simultaneous statistical radius | 0.001238568445136 |
| Interval, including numerical tolerance | [-0.00104738,0.00146976] |

Relative error is 74.58 times the tiny exact gradient; this is NOT high relative
precision. The interval spans zero, so sign agreement is not meaningful proof
of sign. Reported conformance is at the fixed absolute uncertainty/resolution,
not a claim of zero bias for every distribution/model.

Every confirmation root gradient reconstructs from its action histogram and
independent analytic per-action contributions within numerical tolerance. All
4,915,200 original-plus-confirmation occurrence keys are unique/disjoint, with
their digest and explicit root words saved. Distinct PRNG keys are an engineering
check, not a mathematical proof of independent randomness.

New-rule known-correct calibration: **0/2048 family rejections**,95% upper
Wilson bound 0.18722%, below the prospective 2% sanity limit. The analytic union
bound supplies the declared <=1% family guarantee conditional on the finite
distribution and independent draws; simulations are an additional check, not
post-hoc tuning. The new procedure replaces only the affected 42-coordinate
query-loss prerequisite. The two inherited 21-fixture oracle families remain
separate unchanged passing gates; no joint630-coordinate calibration is claimed.

## Validation, decision and continuation

Focused new/structural tests:30 passed. Formatting/lint and targeted typecheck:
pass, zero type errors. Full raw root values, exact means/covariances, calibration
indicators, simultaneous intervals, LP/reconstruction errors and key identities
are retained in `audit.json`, `calibration.json`, `confirmation.json`, `roots/`
and the immutable `decision.json`.

No estimator fix is warranted by these results. A separately versioned G23 V1
may now replace ONLY the new query-oracle statistical prerequisite with this
validated complete-family procedure, while keeping all scientific thresholds,
losses, model, task, optimizer, initialization, data, controls, selection and
budgets unchanged. V0 blocked history remains permanent. Outstanding real-model
preflight must pass before any counted updates; confirmation alone is not
generated-structure trainability or scientific Phase G completion.

No H/I/Thermo change, S4, main merge or push occurred during adjudication.
