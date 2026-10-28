# Production structural-gradient validation

Execution source:`f571ab40659da872777f4deccbfcd05531c4c576`.
Evidence:[production_oracle.json](f571ab40659d/production_oracle.json),
[focused_tests.json](f571ab40659d/focused_tests.json),
[full_model.json](f571ab40659d/full_model.json).

The validated local score bridge was not redesigned. The amended run repeats
the production full-site semantic-sampling/dynamic-capacity pushforward oracle
even though codec integration changes neither the action distribution nor
estimator ownership.

All 21 fixture combinations passed: uniform/asymmetric/peaked logits ×
inactive/reactivation/zero/missing/extra/direct-continuous-dependency/baseline
loss fixtures. Each independently enumerates 108 actions (two binary edges,
fixed terminal, three three-class extents), their categorical probabilities,
the expected actual generated-length-aware loss and its exact gradient.
An independent host partition oracle supplies mechanical cuts.

Monte Carlo uses 16 roots × 4096 samples per root, seed 975, the frozen componentwise
four-standard-error comparison across 16 root-level estimates (zero-SE absolute
tolerance 1e-6).
`production_oracle.json` retains every component's exact gradient, mean, SD,
SE and bias; a nonzero gradient alone is not the acceptance criterion.
These are finite statistical checks, not a universal estimator proof.

Focused behavioral tests retain independent exact enumeration, omitted/duplicated
occurrence mutations, selected-action/log-probability accounting, capacity-hidden
action ownership, forced-terminal exclusion, public sampler/log-probability
parity, and prevention of local/native estimator double ownership. All88
focused tests passed. No straight-through estimator was introduced.

The full B32 actual Adze graph also compares identical-key connected versus
detached selected-action log probabilities through generated packing, full
attention/SwiGLU, unpool/residual and decoder loss:

| Diagnostic | Value |
| --- | ---: |
| Sampled full raw gradient norm |622.448|
| Connected–detached difference norm |558.719|
| Boundary-head score difference norm |64.9313|
| Extent-head score difference norm |72.3953|
| Teacher full raw gradient norm |255.847|
| Prospective masked raw norm |255.837|
| Prospective clipped norm |1|

Ordinary continuous derivatives remain pathwise. The once-only local score
correction owns only the descendant loss, not structural supervised CE.
Teacher/argmax commitments are stopped discrete actions. The prospective
update was discarded and is not teacher update1.

Four-root fixed-input initial sampled-gradient variance is 128432.711 in summed
squared-norm units; raw root norms are 554.305, 537.344, 422.890, 626.237.
This diagnostic demonstrates substantial finite estimator variance and is
not a comparison of trained arms. No sampled training occurred because the
separate structural-necessity gate failed.
