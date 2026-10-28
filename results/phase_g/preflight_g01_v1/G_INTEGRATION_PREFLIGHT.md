# Actual-model integration preflight

Decision:`G_INTEGRATION_BLOCKED`.

Correctness gates pass, but the required post-teacher semantic-boundary
necessity gate fails. No G merge/push to main, H/I merge or full campaign is
permitted by this result.

The faithful frontend/context/target paths, SSM proposal, persistent h/b/ell,
generated hard packing, full attention and SwiGLU L4×Q3, unpool/carrier residual,
S1 and Mamba byte decoder are preserved. Both applications share the exact
parameter tree and original explicit carrier input. No S2 or R execution is
substituted for structural replay.

`full_model.json` proves:

- B32 real sampled score-gradient connectivity and finite teacher gradient.
- Eager/JIT teacher-loss absolute difference 9.536743e-7 (within frozen tolerance).
- Actual-model deterministic/public-Torx predictions and logits: maximum error 0.
- Sanitized placeholder mutation:bitwise invariant prediction/logits.
- Pack/unpool identity for all 32 persistent sites and exact carrier residual.
- All 12 attention/SwiGLU occurrences execute, including inactive query sites.
- A finite prospective AdamW update advances0→1, changes intended means and
  preserves frozen target encoder/rho and frozen moments bitwise. It is discarded.

The 88 focused tests additionally cover compiled invalid-batch rollback,
params/m/v/step invariance, no invalid completed checkpoint, both overflow
modes, all-nonemitting output, reactivation, inactive holes, multi-batch
identity, eager/JIT and supported scan/vmap routing.

Post-teacher same-input/same-key controls show real structural consumption:

| Intervention vs correct | Packing changes | Carrier-ID attention changes | Emission changes | Byte-logit RMS change |
| --- | --- | --- | --- | ---: |
| Shifted cuts |yes|yes|no|0.331362|
| Permuted extents |no|yes|yes|1.33444|
| Joint |yes|yes|yes|1.36819|
| Fixed neutral |yes|yes|yes|0.904986|

These paired trace diagnostics use two retained examples, not all validation
examples. The separate performance gate uses all 256. Output/routing changes
alone are not proof of useful semantic boundaries; the shifted-cut performance
failure is decisive under the frozen gate.

Whole-project validation passed: 258 regular and 30 slow tests, zero skips;
format/lint/type/boundaries/private-Torx checks pass. Exact results are in
[VALIDATION.json](VALIDATION.json). Accepted
historical artifacts passed the before/after727-file hash check. The previously
missing COPY/REVERSE fixtures were restored by exact hash-verified copies, not
new training. Main/H/I source and accepted experiment files remain untouched.
