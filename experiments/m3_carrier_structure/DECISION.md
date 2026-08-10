# M3 decision

**M3_PASS_WITH_LIMITATIONS**

## Evidence

The fixed-capacity carrier jointly represents and reconstructs content,
boundary, and expansion-length state using the public Torx Q=1
`AffineGaussianGate` plus small deterministic readout heads.

Across three seeds at medium structural corruption:

- boundary UNKNOWN accuracy was `0.78946 +/- 0.00092` and UNKNOWN F1 was
  `0.73236 +/- 0.00945`;
- length UNKNOWN accuracy was `0.45566 +/- 0.02205`;
- UNKNOWN length-zero accuracy was `0.67310 +/- 0.02661`;
- h MSE was `0.43269 +/- 0.00572`, exactly matching the h-only control per
  seed.

At full UNKNOWN corruption, all three seeds exceeded the predeclared thresholds
of boundary accuracy `.73828`, boundary F1 `.30`, and length accuracy `.39167`.
The clean and full-UNKNOWN corruption limits, explicit length-zero semantics,
fixed shapes, shuffled-target control, and finite-gradient checks passed.

## Limitations

The structural task is deliberately synthetic and uses an explicit local
boundary/type marker in h to establish recoverability at full UNKNOWN. The
structural heads are readouts only; structural predictions do not affect
computation, so M3 demonstrates representation/recovery rather than routing
utility. Observed-position metrics are near-perfect when those positions are
not masked and are not the primary success measure.

Length recovery at full UNKNOWN is meaningfully above the class baseline but
remains less accurate than boundary recovery. No Q>1 redesign was attempted.

## Scope and gradient route

Only public Torx APIs and ordinary JAX pathwise differentiation through the
continuous AffineGaussianGate are used. The M1.5 score bridge is not used and
was not modified. No private Torx access, Temper, GenJAX/ADEV, stochastic
compiler, dynamic shape, structural routing, or M4 functionality was added.

## Recommendation

Proceed to review before M4. Carry the Q=1 direct carrier as the structural
baseline, retain UNKNOWN-only metrics as the primary structural evaluation,
and treat the synthetic local-marker assumption as an explicit limitation.
M4 must not infer that structural predictions already improve routing; that
question remains outside M3.

Stop here; M4 was not started.
