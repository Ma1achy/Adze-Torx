# M3 RESULTS — carrier structure and structural corruption

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

## Data generator

The frozen generator uses `C=6`, `d=3`, and `L_max=3`. Site zero has
`b=1`; each later site starts a new chunk with probability 0.35. New chunks
receive independent three-dimensional Gaussian latents, while non-boundary
sites receive a small latent drift. Content is:

```text
h_i = chunk_latent_i
      + 0.8*length_i*[1,-0.75,0.5]
      + 1.8*b_i*[1,-0.8056,0.5556]
      + position_effect_i
      + Normal(0, 0.025^2 I)
```

The displayed boundary/type vectors are the exact implementation vectors
`[1.8,-1.45,1.0]` and `[0.8,-0.6,0.4]`. Length is the chunk type sampled
uniformly from `0..3`, so length zero is a normal target class. The target
state is fixed-shape; no site is removed for length zero.

The validation distribution has:

```text
non-forced boundary majority baseline: 0.63828
non-forced boundary prevalence:        0.36172
length majority baseline:              0.29167
length=0 fraction:                     0.25521
```

The local feature vector is `[h_i, h_{i-1}, h_i-h_{i-1}, ||h_i-h_{i-1}||]`
plus one-hot observed b and length. The norm is a single deterministic local
feature that makes jump magnitude linearly readable without routing or a
sequence model. A construction test verifies boundary jumps exceed non-boundary
jumps.

## Corruption and model

Content uses M2's medium law:

```text
h_corrupt = 0.6*h_clean + 0.5*epsilon
```

Boundary and length independently become explicit UNKNOWN states with
probabilities `rho_b` and `rho_length`. `rho=0` exactly preserves clean state;
`rho=1` produces UNKNOWN everywhere. UNKNOWN is distinct from boundary values
0/1 and length zero.

The core is the public Torx Q=1 `AffineGaussianGate` over flattened h. Small
shared per-site affine heads predict boundary and length. Predicted structure
is never passed to the core and never changes connectivity, routing, topology,
or tensor shapes. Parameter counts are 168 for the core plus 58 boundary-head
parameters and 72 length-head parameters in the full model.

The score bridge was not used. Structural targets use ordinary cross-entropy;
no sampled categorical state enters downstream computation.

## Predeclared full-UNKNOWN gate

Before final runs, the thresholds were fixed from the frozen class baselines:

```text
boundary non-forced UNKNOWN accuracy >= 0.63828 + 0.10 = 0.73828
boundary UNKNOWN F1 >= 0.30
length UNKNOWN accuracy >= 0.29167 + 0.10 = 0.39167
length UNKNOWN length=0 accuracy >= 0.25 when enough examples exist
```

## Main ablation, rho_b=rho_length=0.5

All values are validation metrics after 60 Adam steps, batch size 64, learning
rate 0.03, and seeds 0/1/2. Structural metrics include overall accuracy,
UNKNOWN-only recovery, observed-position accuracy, and cross-entropy. Boundary
non-forced accuracy excludes site zero.

| model | seed | h MSE | b overall | b non-forced | b UNKNOWN acc | b UNKNOWN F1 | b observed acc | b CE | length overall | length UNKNOWN acc | length=0 UNKNOWN acc | length observed acc | length CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h | 0 | .42898 | — | — | — | — | — | — | — | — | — | — | — |
| h | 1 | .43873 | — | — | — | — | — | — | — | — | — | — | — |
| h | 2 | .42835 | — | — | — | — | — | — | — | — | — | — | — |
| h+b | 0 | .42898 | .8978 | .8969 | .7902 | .7226 | .9955 | .2469 | — | — | — | — | — |
| h+b | 1 | .43873 | .8991 | .8937 | .7896 | .7331 | .9969 | .2560 | — | — | — | — | — |
| h+b | 2 | .42835 | .8932 | .8906 | .7885 | .7414 | .9937 | .2589 | — | — | — | — | — |
| h+length | 0 | .42898 | — | — | — | — | — | — | .7135 | .4308 | .6575 | 1.0000 | .6583 |
| h+length | 1 | .43873 | — | — | — | — | — | — | .7305 | .4738 | .6570 | .9960 | .6627 |
| h+length | 2 | .42835 | — | — | — | — | — | — | .7292 | .4624 | .7049 | .9987 | .6508 |
| h+b+length | 0 | .42898 | .8978 | .8969 | .7902 | .7226 | .9955 | .2469 | .7135 | .4308 | .6575 | 1.0000 | .6583 |
| h+b+length | 1 | .43873 | .8991 | .8937 | .7896 | .7331 | .9969 | .2560 | .7305 | .4738 | .6570 | .9960 | .6627 |
| h+b+length | 2 | .42835 | .8932 | .8906 | .7885 | .7414 | .9937 | .2589 | .7292 | .4624 | .7049 | .9987 | .6508 |

Full-model aggregate means/stds:

```text
h MSE:                  0.43269 +/- 0.00572
b UNKNOWN accuracy:    0.78946 +/- 0.00092
b UNKNOWN F1:           0.73236 +/- 0.00945
length UNKNOWN accuracy:0.45566 +/- 0.02205
length=0 UNKNOWN acc:   0.67310 +/- 0.02661
```

The full model's h MSE is exactly the same as h-only for each seed because
structural heads cannot affect the Torx content core. Relative to corrupted h,
the full model improves content MSE by approximately 57% on average.

## Structural corruption sweep, seed 0 full model

| rho_b | rho_length | h MSE | b UNKNOWN acc | b UNKNOWN F1 | length UNKNOWN acc | length=0 UNKNOWN acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | .42898 | — | — | — | — |
| 0.25 | 0.25 | .42898 | .8431 | .7736 | .4660 | .6737 |
| 0.50 | 0.50 | .42898 | .7902 | .7226 | .4308 | .6575 |
| 0.75 | 0.75 | .42898 | .7876 | .7179 | .4246 | .6424 |
| 1.00 | 1.00 | .42898 | .7820 | .7144 | .4362 | .6224 |

At rho=0, clean observed structure is recovered exactly; UNKNOWN-only metrics
are undefined. Full-UNKNOWN multi-seed results were:

```text
seed 0: b acc .7820, b F1 .7144, length acc .4362, length=0 acc .6224
seed 1: b acc .7719, b F1 .7056, length acc .4531, length=0 acc .6367
seed 2: b acc .7688, b F1 .7028, length acc .4492, length=0 acc .6378
```

Every seed passes the predeclared full-UNKNOWN thresholds.

Asymmetric checks:

```text
rho_b=1.0, rho_length=0.25: b acc .8109, b F1 .7490, length acc .4207
rho_b=0.25, rho_length=1.0: b acc .9578, b F1 .7593, length acc .4727
```

## Controls and regression checks

The shuffled-target control used fully UNKNOWN observed structure and produced
boundary non-forced accuracy `.5508`, boundary F1 `.2303`, and length accuracy
`.3314`, versus the learned full-UNKNOWN values above. It therefore breaks the
content/structure relationship without exposing original labels.

The exact pre-existing M2 regression path remains covered by the M2 tests; its
raw MSE is not compared numerically with this new C=6,d=3 distribution. Within
M3, h-only is the content baseline and adding b/length leaves h unchanged.

All fixed-shape, rho-limit, UNKNOWN/zero, topology-isolation, reproducibility,
finite-gradient, and tiny-training tests passed. No new Torx limitation was
found. Runtime was approximately 0.3–1.6 seconds per 60-step run including
compilation on CPU, with no non-finite losses, gradients, or activations.
