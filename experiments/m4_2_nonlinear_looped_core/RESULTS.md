# M4.2 results

## Environment and frozen setup

- Python 3.11.13
- JAX 0.8.0
- Torx `extro-torx`, pinned public commit `f1fc858ed950ecd41935d15c06d0ec7c5e0674ae`
- CPU device (`TFRT_CPU_0`)
- float32
- M3 structure: capacity 6, latent dimension 3, max length 3
- nonlinear core width 18, physical depth `L=2`, `eta=0.25`, total nominal variance 0.01
- Q values 1, 2, 4; tied across Q; common Adam learning rate 0.03 and 60 steps
- three seeds: 0, 1, 2
- fixed carrier shape throughout; no routing authority for b/length

The primary composition task uses three fixed nonlinear operators
`T_r(s)=tanh(M_r s+c_r)`, sampled independently per example. The input contains
`s0` and one-hot IDs for `o1...ok`; intermediate states are not inputs or loss
terms. For k≥2, validation holds out complete operator combinations using the
code split `code % 5 == 0`; all individual operators remain represented in
training. k=1 uses all three operators. The structured reconstruction task is
the M3 generator with its fixed medium corruption.

## Core and initialization checks

Each physical block is a public `AffineGaussianGate`; `tanh` is ordinary JAX.
The residual delta is zero-initialized, so the deterministic mean map is exactly
identity at initialization for all Q. The measured initialized end-to-end state
error was 0.0 in float32 for Q=1,2,4. The nominal pre-tanh variance was 0.01 in
every case. Measured post-stack sample variance was 0.000146 (the tanh output
variance is not asserted to equal the nominal Gaussian variance).

| block | L | Q | unique params | effective applications | nonlinear |
|---|---:|---:|---:|---:|---|
| public-Torx stochastic affine + tanh residual | 2 | 1 | 720 | 2 | yes |
| public-Torx stochastic affine + tanh residual | 2 | 2 | 720 | 4 | yes |
| public-Torx stochastic affine + tanh residual | 2 | 4 | 720 | 8 | yes |
| affine control | 2 | 1/2/4 | 720 | 2/4/8 | no |

After training, the nonlinear Q=1 and Q=4 maps had affine defects 0.0163 and
0.0147, with Jacobian state-dependence 0.1655 and 0.1476 respectively. This is
an explicit non-affinity check, not an inference from the use of `tanh`.

## Reconstruction task

Values are best validation h MSE over the common 60-step run, mean ± population
standard deviation across three seeds. Structural heads were retained but the
Q comparison is driven by h MSE.

| block | L | Q | h MSE | b F1 | length accuracy | steady step seconds |
|---|---:|---:|---:|---:|---:|---:|
| nonlinear | 2 | 1 | 0.72730 ± 0.00177 | 0.881 | 0.719 | 0.00031 |
| nonlinear | 2 | 2 | 0.72777 ± 0.00172 | 0.881 | 0.718 | 0.00028 |
| nonlinear | 2 | 4 | 0.72803 ± 0.00178 | 0.881 | 0.718 | 0.00027 |

The M3 reconstruction regression remained finite and retained the expected
structural-head metrics in this short budget; this does not affect the
composition decision.

## Compositional task: fixed L=2

Best held-out final-state MSE, mean ± standard deviation across seeds:

| block | k | Q=1 | Q=2 | Q=4 |
|---|---:|---:|---:|---:|
| nonlinear | 1 | 0.01447 ± 0.00008 | 0.01509 ± 0.00007 | 0.01541 ± 0.00007 |
| nonlinear | 2 | 0.04363 ± 0.00021 | 0.04495 ± 0.00017 | 0.04560 ± 0.00018 |
| nonlinear | 4 | 0.08744 ± 0.00015 | 0.08950 ± 0.00017 | 0.09048 ± 0.00016 |
| affine control | 1 | 0.00824 ± 0.00011 | 0.00923 ± 0.00009 | 0.00973 ± 0.00008 |
| affine control | 2 | 0.02932 ± 0.00031 | 0.03203 ± 0.00023 | 0.03326 ± 0.00020 |
| affine control | 4 | 0.06448 ± 0.00039 | 0.06887 ± 0.00033 | 0.07083 ± 0.00033 |

The affine control was also run at Q=2 and Q=4 for the direct comparison; the
first affine row is the k=1 row. Q>1 did not improve the nonlinear core at any
tested task depth. The anti-Q trend was already visible at k=1 and did not
become a nonlinear-only advantage at k=4.

## Trajectory and parameter-reuse diagnostics

For a representative k=4 validation batch, the nonlinear Q=4 trajectory was
monotonically improving in the supervised final-target distance:

| loop q | target MSE | update norm | fraction improved over prior loop |
|---:|---:|---:|---:|
| 0 | 0.18114 | 0.00000 | — |
| 1 | 0.15347 | 0.05826 | 0.998 |
| 2 | 0.12929 | 0.05540 | 1.000 |
| 3 | 0.10836 | 0.05236 | 0.998 |
| 4 | 0.09035 | 0.04947 | 0.996 |

This is evidence of useful within-trajectory refinement, but its terminal
quality did not beat the independently trained Q=1 model. Internal states were
not required to equal the true intermediate states; they were not used as
training targets.

At approximately matched effective applications, one seed gave:

| L | Q | unique params | effective applications | k=4 loss | steady step seconds |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1440 | 4 | 0.08939 | 0.00029 |
| 2 | 2 | 720 | 4 | 0.08939 | 0.00026 |
| 1 | 4 | 360 | 4 | 0.08938 | 0.00027 |

This small diagnostic suggests parameter reuse can match the endpoint in this
easy setup, but does not establish a quality or efficiency win. Wall-clock
times include neither a claim of compute efficiency nor a separate compilation
amortization claim.

## Controls, failures, and interpretation

- No score bridge was used: all stochastic operations in this core are
  continuous and use the already validated ordinary JAX/Torx path.
- Nominal pre-tanh Gaussian variance is fixed per complete L×Q execution;
  measured post-tanh variance is reported separately.
- Q does not clone parameters, alter carrier shape, or receive structural
  routing information.
- The public Torx limitation encountered was the absence of a documented
  nonlinear continuous factor suitable for this experiment; ordinary JAX tanh
  was therefore kept outside the public Torx stochastic affine operation.
- No M4 or M4.1 record was modified.

The strongest predeclared pattern—neutral Q at k=1 and a growing Q advantage at
larger k, present only for the nonlinear core—did not occur. Q=4 showed
progressive internal refinement but worse terminal held-out error. The result is
negative for this nonlinear Torx-native surrogate, not a universal statement
about all looped architectures.
