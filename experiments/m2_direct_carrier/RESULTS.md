# M2 RESULTS — direct-carrier reconstruction

**Status:** complete, review gate reached

## Environment

```text
Python: 3.11.8
JAX: 0.8.1
Torx distribution: extro-torx 0.0.1
Torx commit: f1fc858ed950ecd41935d15c06d0ec7c5e0674ae
Platform/device: macOS arm64 / CPU
Numerical dtype: float64 (JAX_ENABLE_X64=1)
```

## Toy data and corruption

Each target is a fixed `C=4`, `d=3` carrier flattened to width 12. Three
Gaussian latent variables are projected through constant, linear, and
quadratic position bases and small keyed Gaussian observation noise of 0.02 is
added. Train and validation arrays use independent explicit JAX keys.

Every run uses:

```text
h_corrupt = alpha * h_clean + sigma * epsilon,
epsilon ~ Normal(0, I)
```

Levels are low `(0.9, 0.1)`, medium `(0.6, 0.5)`, and high `(0.3, 0.9)` for
`(alpha, sigma)`. The clean target is used directly; there is no learned
encoder or decoder. The core is a public Torx `AffineGaussianGate` over all 12
continuous carrier values. Q transitions are manually unrolled with explicit
keys; tied runs reuse one parameter tree. Fixed-total-noise mode divides each
transition variance by Q.

Parameter count is 168 for tied runs (`A`, `b`, `log_var`) and 168×Q for
untied runs. Untied runs use identical initial trees and the same parameterization.

The score bridge was not used. This core is continuous and uses the ordinary
JAX pathwise route validated in M1.

## Predeclared gate

Before the final comparison, the main gate was set to: medium-corruption tied
Q=4 must reduce validation MSE by at least 25% relative to no-update on each of
three seeds, with finite diagnostics, decreasing training loss, fixed shapes,
and no recurrence/public-contract failures. A stable but limited result is
`M2_PASS_WITH_LIMITATIONS`.

## Main comparison

Values are validation MSE after 80 Adam steps, three seeds `(0,1,2)`, batch
size 64, learning rate 0.03. The no-update baseline uses validation seed 700.

| corruption | no-update | Q=1 tied | Q=2 tied | Q=4 tied | Q=4 untied | deterministic Q=1 |
|---|---:|---:|---:|---:|---:|---:|
| low | 0.02958 | 0.00762 | 0.00757 | 0.06130 | 0.01441 | 0.01551 |
| medium | 0.55763 | 0.16151 | 0.16703 | 0.18920 | 0.19105 | 0.17059 |
| high | 1.74926 | 0.97369 | 1.00342 | 1.04548 | 1.01937 | 0.99613 |

Per-seed medium tied Q=4 validation losses were `[0.19830, 0.18933,
0.17999]`; every seed passes the gate. Mean reduction is 66.1% versus
no-update.

Per-seed medium controls:

```text
Q=1 tied:       [0.16965, 0.16344, 0.15145]
Q=2 tied:       [0.17504, 0.16927, 0.15680]
Q=4 tied:       [0.19830, 0.18933, 0.17999]
Q=4 untied:     [0.20301, 0.19302, 0.17712]
deterministic:  [0.17984, 0.17200, 0.15993]
```

## Diagnostics and controls

Representative medium seed 0:

| model | train loss | validation loss | grad norm | parameter norm | output variance | effective log variance |
|---|---:|---:|---:|---:|---:|---:|
| Q=1 tied | 0.16139 | 0.16965 | 0.11319 | 21.15 | 1.73986 | -6.06118 |
| Q=2 tied | 0.16480 | 0.17504 | 0.20499 | 20.83 | 1.82398 | -6.66952 |
| Q=4 tied | 0.20981 | 0.19830 | 1.30444 | 20.96 | 1.59657 | -7.39676 |
| Q=4 untied | 0.21182 | 0.20301 | 0.74395 | 39.06 | 1.65381 | -6.97133 |

All 80-step runs had finite losses, gradients, and activations. Fixed-total
noise prevents Q from mechanically multiplying injected variance; learned
effective log variance decreased as Q increased. Q=4 remains trainable but
has higher gradient norms and weaker final loss. Compile-inclusive runtime was
approximately 0.4–1.1 seconds per run on CPU; steady-state step time was
approximately 0.005–0.014 seconds.

## Required semantic checks

```text
fixed carrier shapes: PASS
explicit-key data/corruption determinism: PASS
no-update equals corrupted input: PASS
zero-corruption identity: PASS
Q=1 equals one base AffineGaussianGate transition: PASS
manual recurrence/public ChainFactor finite semantics: PASS
fixed-seed reproducibility and split-key independence: PASS
finite full forward and gradient step: PASS
tiny training learns: PASS
score_bridge used: NO (continuous pathwise core)
```

The public `ChainFactor` comparison checks fixed shape and finite forward
semantics. Exact sampled-array equality is not asserted because the public
composite owns its own key handling; manual unrolling is the controlled Q path.

## Limitations and surprises

Q=1 and Q=2 tied transitions were consistently better than Q=4 tied on this
toy task. Q=4 clears the medium gate but is worse than the deterministic affine
control and has larger gradient norms. At low corruption, tied Q=4 worsens the
no-update baseline. Untied Q=4 supplies no consistent advantage and has 4× the
transition parameter count.

No new Torx limitation or gradient failure was found. The core is continuous,
so M1.5's discrete score correction remains untouched.
