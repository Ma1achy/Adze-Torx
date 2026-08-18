# Phase F — faithful denoising/resampling depth S

Current load-bearing status: **PHASE_F_1_ONE_STEP_BLOCKED**.

No denoising-depth experiment has started. There is currently no evidence for
or against additional denoising depth S.

For tooling that requires a top-level Phase-F field, the administrative status
is `PHASE_F_S_UNRESOLVED`: **administrative/contract status only — Phase F
experiments not started because the corruption law is undefined**.

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
