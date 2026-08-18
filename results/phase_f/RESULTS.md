# Phase F — faithful denoising/resampling depth S

Current load-bearing status: **PHASE_F_2_SAME_MODEL_S_DEGRADATION**.

Phase F.2 evaluates repeated application of the accepted F1 one-step checkpoint
without training or parameter mutation. The earlier F0/F1 sections below are
retained as historical provenance.

## F0 corruption audit

The repository does not contain a usable content-corruption contract:

- `src/adze_t/corruption.py` is an intentional stub with no executable
  corruption symbols.
- The architecture document gives only the abstract equation
  `h_nu = alpha(nu) h_0 + sigma(nu) epsilon`; neither function is implemented.
- No accepted domain or endpoint convention for `nu` exists.
- No initial continuous-carrier corruption or inter-step re-corruption kernel
  exists.
- No `eta_diff` semantics or diffusion-noise PRNG namespace exists.
- Phase B/D/E did not sample content-corruption levels.

The current `apply_model()` path is explicitly an S=1/R=0 graph. It encodes the
clean target for supervision and fixed teacher structure, while the heavy model
carrier begins from the context-derived proposal. It does not receive a
corrupted continuous carrier and does not produce a self-generated S
trajectory.

DiT exposes `noise` and `denoise_step` conditioning coordinates, but
`apply_model()` does not supply them; their defaults are `0.0` and `0`.
Constant conditioning defaults are not a corruption schedule and do not define
`alpha`, `sigma`, corruption sampling, or re-corruption.

Phase D/E Torx noise is learned-operator stochasticity. It must not be
retroactively described as diffusion/content stochasticity.

The machine-readable evidence is in `corruption_audit.json`.

## Required scientific contract

Phase F can resume only after `PHASE_F_CORRUPTION_CONTRACT_V0` explicitly
freezes:

1. the forward corruption family relating `alpha(nu)` and `sigma(nu)`;
2. the domain and meaning of `nu`;
3. the one-step training distribution over corruption levels;
4. the master multi-S schedule `nu_0 > nu_1 > ...`;
5. model-prediction re-corruption semantics;
6. `eta_diff` and diffusion PRNG semantics;
7. endpoint meanings for `nu=0` and the noisiest allowed value.

F0 deliberately selects none of these. No `denoise_step`, DENOISE_V0 dataset,
multi-S trajectory, diffusion keying, eta law, or conventional diffusion
schedule was implemented.

## Validation

- `make format-check`: 117 files already formatted.
- `make lint`: passed.
- `make typecheck`: 0 errors and 0 warnings.
- `make test`: 147 passed, 9 deselected.
- `make boundaries`: passed.
- `make test-slow`: 9 passed, 147 deselected.
- Explicit private-Torx import scan: zero matches.

## Phase F.0.1 contract resolution

The explicit `PHASE_F_CORRUPTION_CONTRACT_V0` design decision resolves the F0
blocker without claiming recovered original-Adze semantics. Its standalone JAX
substrate implements the frozen trigonometric forward kernel, eta 0/1
re-corruption primitive, dedicated diffusion-key namespace, reproducible
sampling, and prefix-compatible S=4 schedule.

No accepted model/training path changed and no S experiment started. The
focused contract and numerical evidence are recorded in
`corruption_contract_v0.md` and `corruption_contract_v0.json`.

## Phase F.1 — faithful one-step corrupted carrier

The explicit current-carrier interface is now architecture-faithful and
backward compatible. Legacy calls still use `proposal_h`; explicit calls use
`carrier_h_input` at both the packed heavy-core input and post-unpool residual
base. The qualified parity, leakage, structural-invariance, carrier-localization,
and paired-diffusion-key regressions pass.

DENOISE_V0 disables the impossible constant-prompt proposal auxiliary objective
with `PHASE_F_1_DENOISE_V0_PROPOSAL_AUX_DISABLED`; all other x0 objective
weights remain one.

The mandatory codec suitability gate blocked training. The accepted frozen
codec was trained on byte values `1..32`. It retains its historical control
performance of 99.4141% byte and 95.3125% exact accuracy there, but on required
uniform `0..255` DENOISE_V0 targets it achieves only 12.4512% byte and 0% exact
accuracy. Uniform-target h0 states are unique but highly clustered
(mean coordinate variance 0.00285248; mean pairwise cosine 0.993610).

No first gradient, overfit, calibration, lambda sanity evaluation, or S>1
experiment was started after this gate failed. The authoritative F1 status is
`PHASE_F_1_ONE_STEP_BLOCKED`; it is not an S-benefit, neutrality, or negative
result.

Phase F.1 validation: format, lint, type checking, dependency boundaries, and
the private-Torx import scan passed; the test suites reported 155 regular tests
and 9 slow tests passing.

## Phase F.1 continuation — DENOISE_V1

The DENOISE_V0 record above is preserved as a benchmark/domain failure. The
domain-aligned DENOISE_V1 changes only the target distribution to eight iid
bytes in `1..32`, exactly matching the accepted frozen codec's historical
domain. The decoder remains an unmasked 256-way head. Chance is 3.125% per byte
and `(1/32)^8` per exact sequence.

The fresh 4,096-example V1 codec audit passed at 99.4568% byte and 95.7275%
exact reconstruction. The stochastic first-gradient gate passed with all
required connected families finite and nonzero, proposal gradient correctly
zero under the benchmark-local disabled proposal auxiliary, rho frozen, and
the target teacher bitwise unchanged. Fixed-corruption overfit reached 100%
exact accuracy for one and eight examples; the 256-example run reached 97.3145%
byte accuracy and NLL 0.112626.

F_ONE_STEP_V1 then trained for 5,000 steps. Its final lambda-zero byte accuracy
was 99.7070%, 78.4668%, 27.5879%, 9.4971%, and 5.0293% at `nu` 0.10, 0.25,
0.50, 0.75, and 0.90 respectively, versus 3.125% byte chance. This establishes
a learnable, non-ceiling, corruption-dependent range. Four finite
`lambda_op=1` roots closely matched lambda zero with no nonfinite values.

The benchmark-local status is `DENOISE_V1_CALIBRATION_PASS`; the milestone
status is `PHASE_F_1_ONE_STEP_PASS`. Detailed evidence is under
`f1/denoise_v1/`. No S>1 code or experiment was run, so no `PHASE_F_S_*`
scientific decision is issued.

Completion validation passed with 120 files formatted, lint and type checking
clean, dependency boundaries intact, 156 regular tests, 9 slow tests, 9 focused
F1 tests, and zero private-Torx imports.

## Phase F.2 — same-model denoising depth

The accepted step-5000 F_ONE_STEP checkpoint and codec matched their required
SHA-256 digests exactly. The primary `F2_STEP0_CONDITIONING` experiment used the
actual rollout indices `0,1,2,3` for diffusion and Torx occurrence identity but
held the learned denoise conditioning coordinate at `0` for every application.
Thus S=1 exactly reproduces F1 and later steps introduce no untrained
conditioning codes. `L=4`, `Q=3`, and `R=0` remained fixed.

On all 4,096 frozen DENOISE_V1 test examples at lambda-op zero and eta zero,
S4 versus S1 byte accuracy changed by -74.4995, -23.2483, and -5.6000
percentage points at nu 0.25, 0.50, and 0.75. The corresponding paired
bootstrap 95% intervals were [-74.9664, -74.0171], [-23.7610, -22.7325], and
[-5.9540, -5.2429] percentage points. NLL worsened by 15.6146, 10.0101, and
5.4611. Carrier MSE was mixed: +0.00891, -0.00465, and -0.01703. This latent
movement does not offset the large, systematic degradation in decoded task
metrics.

The full eta-one test sweep showed the same result. The fixed 512-example,
16-root Torx confirmation at lambda-op one and sigma-op 1e-3 also reproduced
the eta-zero S4-S1 changes (-74.5651, -22.8333, and -5.6274 percentage points
in the informative region), with narrow root-level Student-t intervals and no
nonfinite values. No 1,024-example escalation was needed because there was no
high-variance conflict or stochastic reversal.

All parameter, rho, and teacher trees were bitwise equal before and after
evaluation. The trajectory never accepts clean h0, sanitized target content
placeholders are inference-inert, predicted structure never controls packing,
and diffusion and operator namespaces remain distinct. The F1 nested audit's
stale DENOISE_V0 metadata label is corrected only in new F2 provenance; the
historical F1 evidence is unchanged. Literal native-S conditioning remains an
unrun diagnostic and does not affect the decision.

Detailed aggregate, per-example, bootstrap, Q0-shell, stochastic-root, and
provenance evidence is under `f2/`. No optimizer or rollout training ran.

Phase F.2 completion validation passed with 122 files formatted, lint and type
checking clean, dependency boundaries intact, 162 regular tests, 9 slow tests,
6 focused F2 tests, and zero private-Torx imports.
