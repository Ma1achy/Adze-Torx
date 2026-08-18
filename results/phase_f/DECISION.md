# Phase F decision

## F0_CORRUPTION_CONTRACT_BLOCKED

The corruption-contract gate did not pass. The repository defines the intended
typed-corruption shape abstractly but contains no implemented continuous
content-corruption law, accepted noise coordinate, training distribution,
rollout schedule, re-corruption rule, diffusion stochasticity parameter, or
diffusion PRNG namespace.

No Phase F S experiment has started, so this record is not a scientific result
about whether additional denoising steps help, hurt, or are neutral.

If a top-level status is administratively required, it is
`PHASE_F_S_UNRESOLVED`: **administrative/contract status only — Phase F
experiments not started because the corruption law is undefined**. The
authoritative milestone status is `F0_CORRUPTION_CONTRACT_BLOCKED`.

The next scientific-design action is to explicitly choose and freeze
`PHASE_F_CORRUPTION_CONTRACT_V0`. Implementation may resume only after that
contract defines the seven choices listed in `RESULTS.md` and
`corruption_audit.json`.

## Phase F.0.1 — PHASE_F_CORRUPTION_CONTRACT_V0_PASS

The previously missing choices are now explicitly frozen in
`corruption_contract_v0.md` and implemented as a standalone, tested corruption
substrate. This is a new Phase-F design decision, not a rewrite of the F0 audit
or a claim about original-Adze behaviour.

No denoiser wiring, DENOISE_V0 calibration, rollout, or S-dependent experiment
was performed. Therefore no `PHASE_F_S_*` scientific decision is issued.

## Phase F.1 — PHASE_F_1_ONE_STEP_BLOCKED

The explicit one-step corrupted-carrier substrate passed qualified parity,
leakage, fixed-structure invariance, localization, and PRNG-pairing tests.
However, the frozen target codec failed the mandatory DENOISE_V0 suitability
gate on uniform random byte targets: 12.4512% byte reconstruction and 0% exact
reconstruction, versus its preserved 99.4141%/95.3125% accepted-domain control
on byte values `1..32`.

Per the predeclared gate order, first-gradient, overfit, calibration, and
lambda-op sanity runs were not started. No `PHASE_F_S_*` decision is issued and
no S>1 implementation exists.

## Phase F.1 continuation — PHASE_F_1_ONE_STEP_PASS

The historical DENOISE_V0 blocker remains valid and preserved. DENOISE_V1
changes only its target distribution to the frozen codec's accepted `1..32`
domain. The new codec audit, stochastic first-gradient gate, fixed-corruption
overfit gates, and 5,000-step one-step calibration all passed.

DENOISE_V1 spans a useful corruption range: low-noise reconstruction is near
ceiling, intermediate levels are learned but imperfect, and high-noise levels
remain difficult. Finite Torx operator noise preserved one-step function over
four sanity roots, with no nonfinite values. Qualified parity, leakage,
fixed-structure, and paired-diffusion-key guarantees from the blocked milestone
remain in force.

The benchmark status is `DENOISE_V1_CALIBRATION_PASS` and the authoritative F1
milestone status is `PHASE_F_1_ONE_STEP_PASS`. This supersedes the top-level F1
blocked status without deleting or reinterpreting DENOISE_V0. No S>1 execution
has occurred, so no benefit, neutrality, or negative decision about denoising
depth S is issued.
