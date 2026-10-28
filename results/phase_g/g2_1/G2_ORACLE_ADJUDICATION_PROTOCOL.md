# G2.1 oracle adjudication V1 — prospective confirmation

Base: `340a0ab4c1d7c13b74bc7f2c0fe08eaae3456810`, isolated G branch.
The original G23 V0 `PHASE_G_BLOCKED` result, protocols, roots, raw values and
manifest are immutable. This is not a retry of that gate. No neural training
is authorized by a partial adjudication pass.

## Family, estimator and independent audit

The complete affected preregistered query-loss family is three parameter
fixtures (uniform, asymmetric, peaked), each with 4 boundary-logit, 9
extent-logit and 1 direct-theta coordinate: **42 scalar comparisons**. Do not
drop redundant/oppositely signed coordinates or select only the failed one.
The inherited two 21-fixture oracle families remain unchanged separate gates;
their passing evidence and source hashes must also be verified, not relabelled
as newly confirmed under a family-wide guarantee.

Enumerate all 108 actions independently using Python integer products. Compute
float64 softmax product probabilities, global emitted-byte ordering, missing/
extra penalties, query-byte loss and analytic per-action score/pathwise vectors
using NumPy algebra, not JAX differentiation or the production loss helper.
Check analytic expected gradients against float64 central differences (step
1e-4, tolerance 2e-8 + 1e-4*abs(gradient)). Compare every action's loss and LP,
and exact expected production bridge gradient, with the independent oracle.
Production comparisons allow 2e-5 + 2e-5*abs(reference) for float32 arithmetic;
probability sums and analytic categorical score identities allow 1e-12.

The fixture's length-aware byte/query terms are EXACTLY the production
`aligned_metrics`/generated-byte semantics. Its declared quadratic direct-theta
term is a controlled continuous-dependency fixture, not a claim to reproduce
the numerical latent MSE of the complete neural network. Real-network gradient
connection and new-task first-gradient/update tests remain separate gates.

Retain the production sampler, zero baseline, positive loss*score sign, local
once-only bridge, all five stochastic decisions, forced terminal exclusion,
and mechanical capacity cuts distinct from semantic actions. Audit masks,
reactivation, owner rejection, sample LP equality, and omitted/doubled/reversed
score/pathwise mutations. Baseline checks are diagnostic identities with
action-independent constants; they do not change production baseline zero.

Reconstruct ORIGINAL 16-root draws only to recover root gradients, frequencies,
outliers, skew, variance concentration, component correlations and the exact
old nonpass. Do not overwrite any original evidence or count reconstruction
as independent confirmation. Check original means against their saved values.

## Independent confirmation (fixed, no optional stopping)

For EACH of the three fixtures, run **64 roots, 4096 draws/root** (262144 per
fixture; 786432 total) through the unchanged production sampler and autodiff
bridge. Root r=0..63:
`named_key("g21_confirmation_v1:" + fixture, 0, r)` using existing master12000.
These namespaces have never been executed in V0. Record actual key words,
global IDs0..4095, occurrence/S/R/commit0, kind/site coordinates. Check key
uniqueness within confirmation and disjointness from original sample keys.
Save every root's gradient vector, action histogram and selected LP errors.
Reconstruct each root gradient from its action frequencies and independently
calculated per-action vectors within the float32 tolerance above.

Primary statistic: absolute pooled MC mean minus independent exact mean, for
ALL42 coordinates. Let v be the exact finite-action per-draw variance and
M=max_a |g(a)-E[g]|. Let n=262144, m=42, alpha=.01, t=log(2*m/alpha), and

    radius = M*t/(3*n) + sqrt((M*t/(3*n))**2 + 2*v*t/n)

Accept each coordinate iff error <= radius + float32 numerical tolerance.
For v=M=0 the radius is zero: only numerical tolerance is allowed. This is the
positive solution of the two-sided Bernstein exponent n*r^2/(2*v+2*M*r/3)=t;
union bound gives family-wise rejection probability <=.01 under the declared
finite distribution and independent draws. It does not assume Gaussian root
means or infer variance from missing rare observations. Dependence BETWEEN
gradient coordinates does not invalidate the union bound. PRNG distinctness
is an implementation check, not a mathematical proof of random independence.

Report exact variance and empirical root SD/SE separately, absolute/relative
error (relative unstable near zero), simultaneous intervals, sign agreement,
and signal resolution. Sign agreement is NOT required when the interval spans
zero. A wide conservative interval is not evidence of high relative precision.
No extra roots, sample escalation, root exclusion, baseline tuning or gate
revision after results. A nonpass stops dependent continuation.

## Calibration simulation and old gate assessment

Use 2048 independent simulated experiments for the COMPLETE42-coordinate family
under the known-correct finite-action distribution. Categorical multinomial
counts retain shared-action component correlations. Draw counts via explicit
JAX binomial keys, not Python/NumPy RNG. Process chunks256, sequential fixtures.

Original design: 16 roots x4096, original empirical-root-SE four-SE and zero-SE
1e-6 rules, comparing with original float32 exact targets as well as independent
float64 targets. Confirmation design: pooled262144, finite-action radius above.
Use namespaces `g21_calibration_v1:old:<fixture>` and
`g21_calibration_v1:new:<fixture>`, root=chunk index; disjoint from confirmation.
Save per-coordinate rejection counts, joint-family indicators, max statistics,
95% Wilson intervals, empirical covariance/rank and rare-count distributions.
No simulated experiment is a new production confirmation root.

Calibration sanity gate: new-rule empirical family rejection <=.01 AND its
95% Wilson upper bound <=.02. This is an additional diagnostic check, not a
replacement for the analytic bound. Report uncertainty if simulations lack
power; do not call a rejection “bias” in a known-correct oracle. The original
four-SE rule had NO preregistered family alpha, and Gaussian/Student-t nominal
tail rates are not guaranteed for sparse non-normal categorical gradients.
Estimate its actual false-rejection frequency rather than inventing a nominal
historical guarantee. Keep family42 distinct from the larger inherited gates.

## Decision, budget and stop

`G21_GRADIENT_ESTIMATOR_CONFIRMED`: all independent exact/LP/ownership/mutation/
key checks, all42 confirmation comparisons and calibration sanity pass.
This confirms conformance at tested fixtures and reported precision, not proof
of zero bias in every model/distribution. A single old extreme studentized draw
or an unbiased variance estimator's low realization is not automatically an
uncertainty implementation defect.

`G21_GRADIENT_ESTIMATOR_BIASED`: independently demonstrated expectation/sign/
probability/ownership defect (not merely one finite statistical rejection).
`G21_UNCERTAINTY_ESTIMATOR_DEFECT`: demonstrated variance/SE implementation
error or unsupported claimed nominal coverage, with estimator otherwise correct.
`G21_UNRESOLVED`: any unresolved audit, confirmation or calibration nonpass.
Stop affected work on nonfinite values, invalid probabilities, key collision,
historical mutation, LP/loss discrepancy or failed confirmation. No outcome-
dependent rerun. One sequential CPU job, no simultaneous training; budget one
execution per stage, at most two hours for adjudication (durable partial records
and unresolved result if resource failure prevents completion).

Commit protocol AND implementation, then freeze their SHA-256 identity in a
separate commit BEFORE executing any new confirmation/calibration samples.
Preserve partial/error outputs and all original hashes. Protect test remains
closed; do not touch H/I/Thermo, S4, architecture, optimizer or task.

## Conditional continuation

Only after CONFIRMED and all adjudication gates pass, create separate G23 V1
protocols/source binding. Preserve ALL V0 scientific configurations, losses,
thresholds, controls, budgets, selection, extension and protected-test rules.
Change ONLY the new query-oracle prerequisite to this validated complete-family
procedure; reuse unchanged inherited oracle evidence and retain V0 blocked
history. Run outstanding real-model G2 preflight before counted updates. V1
uses separate evidence/checkpoint paths, fresh Adam, same immutable teacher100
warm-start and codec. Continue authorized G2/G3 sequentially only if those gates
pass. No main merge before full correctness/integration/scientific gates.
