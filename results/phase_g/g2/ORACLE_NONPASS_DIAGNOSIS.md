# G2 V0 preflight — finite-MC nonpass, not a trainability result

The frozen execution identity is
`1d8ee3e970c9b8c98de1f027baa1adc4d3f03a941526238d91dbc827ad55a784`.
Prospective source commit `6e68e0f688261ec954ba718f9c7e8da0fa59ef47`;
identity/preservation commit `10aec599e4ce08244e297d752c19ee5a770179fd`.
No frozen execution source, protocol, optimizer or loss was changed after this
gate result. The separate diagnostic was added afterward as identified below.

## Frozen result

All three new query-length exact 108-action checks pass, and duplicated-score
mutations are rejected. Uniform and asymmetric 16-root × 4,096 MC fixtures pass.
Peaked fixture: one extent-logit component, site 0/class 2, fails the required
componentwise four EMPIRICAL ROOT standard-error criterion.

| Quantity | Value |
|---|---:|
| Exact gradient | 0.000002794152806 |
| MC mean | -0.001080604037270 |
| Absolute error | 0.001083398237824 |
| Estimated root SE | 0.000243438757025 |
| Error / estimated root SE | 4.450393 |
| Frozen maximum | 4.0 |

This remains a NONPASS. No component is discarded, no seed/root rerolled, no
extra samples/roots added and no threshold changed. The previous accepted
21-fixture global-loss and capacity-pushforward evidence remains unchanged;
this is a new, additional query-loss preflight, not a rewrite of old passes.

## Same-draw diagnosis (not a replacement gate)

Diagnostic source was committed at
`08b1299d5e75ebcdc788e420ba93b348f70e087c` before executing it. A later type-only
cast fixes a static checker annotation, not its numerical algorithm/outcome.
See `ORACLE_NONPASS_DIAGNOSIS.json` for complete component moments and counts.

- Public-Torx selected-action log probabilities checked over all 108 actions;
  maximum difference from independent summed LP 1.9073486328125e-6 (float32
  summation order). Production bridge exact-gradient errors are 0 in every
  parameter block. No omitted/doubled score ownership was found.
- All 327680 boundary/extent stochastic keys across the ORIGINAL 16 × 4,096 draws
  are unique. Terminal and mechanical capacity cuts consume no stochastic key.
- The offending class has probability 0.000123282171070. Expected count over
  65536 samples: 8.07942036; observed: 2. Binomial lower-tail probability for
  count<=2 is 0.01292254. No new draw identities were introduced.
- Independent finite-action analytic variance is 0.01681246749 per draw;
  known-variance SE for 65536 draws is 0.000506495701. The saved MC error is 2.13901
  of those diagnostic SE, whereas the estimated root SE is smaller. This is
  consistent with a rare-event undercount depressing the finite empirical SE.
- Reconstructing the SAME sampled-action frequencies and analytic score
  contributions matches the saved MC gradients within 1.3741e-6. No evidence
  of a wrong gradient sign, duplicate keys or hidden native estimator emerged.

These diagnostics do NOT establish that the predeclared acceptance rule passes.
Exact agreement does not erase the separate MC prerequisite. A statistical
fluctuation is not proof that the estimator is biased; conversely a plausible
variance explanation does not authorize replacing the frozen statistic after
seeing it. There is no identified implementation repair that would make this
unchanged draw set satisfy the unchanged rule.

## Consequence and remaining work

Stop dependent G2 counted training and G3 scientific evaluation under V0.
No A/B/C update, full-size parent-model gradient/trial update, G2 checkpoint
selection, new generated-policy performance result or protected-test exposure
occurred. Random small-model regression updates are engineering tests only.
The accepted teacher100 checkpoint and all historical artifacts are untouched.

A future separately versioned gate could prospectively calibrate finite-oracle
MC uncertainty using known categorical moments and rare-event coverage, or
prespecify sufficient draws. That would need independent validation and must
retain this failed V0 result. No such amended confirmation/training was executed
here; changing the failed denominator/threshold or drawing until a pass would
violate the frozen V0 protocol. This is a procedural/statistical preflight
blocker, NOT evidence that generated structure or Adze cannot train.
