# PHASE_F_CORRUPTION_CONTRACT_V0

Status: **PHASE_F_CORRUPTION_CONTRACT_V0_FROZEN**.

Milestone decision: **PHASE_F_CORRUPTION_CONTRACT_V0_PASS**.

This contract is an explicit Phase-F design choice motivated by the prior
`F0_CORRUPTION_CONTRACT_BLOCKED` audit. It is not recovered original-Adze
behaviour and is not claimed to reproduce DDPM, DDIM, VP-SDE, VE-SDE, or any
other named diffusion method exactly.

## Forward corruption

The noise coordinate is `nu in [0, 1]`:

```text
alpha(nu) = cos(pi * nu / 2)
sigma(nu) = sin(pi * nu / 2)
h_nu = alpha(nu) * h_0 + sigma(nu) * epsilon
epsilon ~ Normal(0, I)
```

Thus `alpha(nu)^2 + sigma(nu)^2 = 1`. At `nu=0`, `h_nu=h_0`.
At `nu=1`, `h_nu=epsilon`. The latter is a semantic pure-noise endpoint and
is not automatically an appropriate recoverability target for random
DENOISE_V0 examples.

Eager calls reject non-finite or out-of-domain `nu`; JIT-traced calls assume
the declared domain. The pure JAX implementation is differentiable and does
not clamp or apply hidden Q/S/dimension scaling.

## Frozen future experiment coordinates

- One-step training distribution: `nu ~ Uniform(0.025, 0.9)`.
- Evaluation grid: `[0.10, 0.25, 0.50, 0.75, 0.90]`.
- S=4 master schedule: `nu_0 * [1.0, 0.75, 0.50, 0.25]`.
- Every shorter S execution is a strict prefix of that master schedule.

The training distribution is recorded only. No denoiser training or
DENOISE_V0 calibration occurred in this milestone.

## Re-corruption and randomness

Future inter-step re-corruption is:

```text
h_(s+1) = alpha(nu_(s+1)) * h_hat_(0,s)
          + eta_diff * sigma(nu_(s+1)) * epsilon_(diff,s+1)
```

V0 authorizes only `eta_diff=0` (deterministic mean re-corruption) and
`eta_diff=1` (full fresh Gaussian re-noising). Initial corruption always uses
the full forward kernel and is not disabled by `eta_diff=0`. V0 contains no
inferred-epsilon transport or DDIM-style rule.

Diffusion keys use a dedicated fixed namespace followed by explicit markers
for global example identity, diffusion stage, and denoise step. Stages are
`INITIAL_CORRUPTION` and `RECORRUPTION`. Keys never depend on requested total
S, so extending a trajectory appends randomness without changing its prefix.
This key path is separate from Torx operator occurrence keys.

## Structure and later rollout loss

Scope is `PHASE_F_CONTINUOUS_S_FIXED_STRUCTURE_V0`. Only continuous `h` may
participate in a future self-generated S trajectory. Boundaries, extents,
activity, packing, masks, emission, and decoder routing remain fixed under the
accepted Phase D/E structure policy. The discrete score bridge is not used.

The recorded but unimplemented future objective is
`PHASE_F_ROLLOUT_LOSS_V0`: the mean ordinary x0 loss across S predictions.
Clean targets are loss oracles only; no true `h_0` enters intermediate rollout
state, gradients propagate through continuous re-corruption, and V0 specifies
no hidden stop-gradient boundary.

## Implemented substrate

`src/adze_t/corruption.py` provides typed diffusion stages/eta modes/key
contexts, `alpha`, `sigma`, `corrupt_h`, `recorrupt_h`, reproducible sampling
helpers, and `phase_f_schedule`. No existing model, training, DiT, structure,
or Torx operator path was changed.

## Numerical validation

Using deterministic seed 75 and 8,192 samples at `nu=0.5`:

- initial mean absolute error: `0.00427625`;
- initial unbiased-variance absolute error: `0.00535014`;
- eta=1 re-corruption mean absolute error: `0.00427625`;
- eta=1 unbiased-variance absolute error: `0.00535014`;
- eta=0 unbiased empirical variance: `1.07483e-13`, with every sampled value
  bitwise identical;
- maximum unit-energy error on the required grid: `5.96046e-8`;
- maximum fixed-epsilon gradient error for `h_0`, `nu`, and both eta modes:
  `0.0` in float32.

The statistical tolerance was `0.03` for mean and variance absolute error and
`1e-12` for the numerically evaluated eta=0 unbiased variance.

Repository-wide validation passed: 117 files formatted, lint clean, 0 Pyright
errors/warnings, public boundaries clean, private-Torx scan empty, 147 regular
tests passed, and 9 slow tests passed.
