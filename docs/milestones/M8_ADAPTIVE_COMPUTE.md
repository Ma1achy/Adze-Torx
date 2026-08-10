# M8 — Adaptive selection and stopping

Only after fixed-Q/S/R behaviour is trustworthy.

## Inner S

Use convergence diagnostics over content/boundary/length state.

## Selection

Candidate signals:

- snapshot confidence;
- predictive entropy;
- random;
- causal-vs-global disagreement;
- calibrated learned selector.

## Outer R

Continue only when expected utility is positive.

A post-hoc survival predictor may batch examples by likely R but is not a hard cap.

## Gate

Compute-vs-quality curves are monotonic enough to support an explicit operating point and adaptive logic does not reduce correctness through premature stopping.
