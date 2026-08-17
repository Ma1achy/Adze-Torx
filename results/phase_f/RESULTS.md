# Phase F — faithful denoising/resampling depth S

Load-bearing status: **F0_CORRUPTION_CONTRACT_BLOCKED**.

Phase F experiments have not started. There is currently no evidence for or
against denoising depth S.

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

- `make format-check`: 116 files already formatted.
- `make lint`: passed.
- `make typecheck`: 0 errors and 0 warnings.
- `make test`: 132 passed, 8 deselected.
- `make boundaries`: passed.
- `make test-slow`: 8 passed, 132 deselected.
- Explicit private-Torx import scan: zero matches.
