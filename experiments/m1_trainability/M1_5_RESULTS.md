# M1.5 RESULTS — mixed-gradient bridge

**Status:** complete, review gate reached

## Scope and estimator

The local helper implements:

```python
score_zero = log_prob_sum - stop_gradient(log_prob_sum)
corrected_loss = loss + stop_gradient(loss) * score_zero
```

The helper requires exactly matching shapes. Scalar loss/log-probability pairs are
accepted; per-trajectory arrays are accepted only when their shapes are identical.
Ambiguous scalar/vector and mismatched batch shapes are rejected.

The mixed runner accumulates the public `PNOT.log_probability` for every
discrete recurrence occurrence, applies the correction per sampled trajectory,
then averages corrected losses across the batch. Continuous parameters remain on
the ordinary JAX pathwise route through the public Torx `MixtureGaussianGate`
sample operation.

Native Torx `BranchingSimulator` estimates are not wrapped. The helper has no
estimator registry or route planner; callers must supply log probabilities only
for stochastic operations whose gradients are not already native.

## Environment

Same pinned environment as M1:

```text
Python: 3.11.8
JAX: 0.8.1
Torx: extro-torx 0.0.1
Torx commit: f1fc858ed950ecd41935d15c06d0ec7c5e0674ae
Platform/device: macOS arm64 / CPU
Numerical runs: JAX_ENABLE_X64=1
Samples per seed block: 4096
Mixed recurrence seed blocks: 8 for depth sweep, 16 for before/after comparison
Acceptance: absolute error <= 4 measured standard errors per parameter
```

## Mixed gradient before/after bridge

Parameters were `[theta, alpha, beta, log_var]=[0.2,0.8,0.4,-1]`,
depth 4, and the exact conditional-moment oracle was used.

| route | theta mean | alpha mean | beta mean | log_var mean |
|---|---:|---:|---:|---:|
| exact oracle | 0.00362619 | 2.98460569 | 2.20750560 | 0.68035447 |
| before bridge | 0.00000000 | 3.00777408 | 2.20842838 | 0.68618423 |
| after bridge | -0.00740517 | 3.00777408 | 2.20842838 | 0.68618423 |

Before/after used 16 seed blocks of 4096 trajectories. Before-bridge
standard errors were `[0, 0.01906473, 0.00722986, 0.00436534]`; after-bridge
standard errors were `[0.00724347, 0.01906473, 0.00722986, 0.00436534]`.

Before bridge, the discrete theta gradient was identically zero. After bridge,
the error in standard-error units was:

```text
theta   -1.52
alpha    1.22
beta     0.13
log_var  1.34
```

The continuous gradients were bitwise unchanged between the two routes for
the same seed blocks.

## Mixed recurrence depth sweep

Each row used 8 seed blocks of 4096 trajectories. Columns are vectors in
parameter order `[theta, alpha, beta, log_var]`.

| depth | exact gradient | estimate mean | std | stderr | error/stderr |
|---:|---|---|---|---|---|
| 1 | [0.160391, 0, 0.516844, 0.294304] | [0.159924, 0, 0.519084, 0.291834] | [0.009256, 0, 0, 0.006718] | [0.003273, 0, 0.005724, 0.002375] | [-0.14, exact-zero, 0.39, -1.04] |
| 2 | [0.007014, 0.712836, 1.075960, 0.482658] | [0.007541, 0.712251, 1.074348, 0.486558] | [0.010336, 0.012567, 0.035706, 0.013735] | [0.003654, 0.004443, 0.012624, 0.004856] | [0.14, -0.13, -0.13, 0.80] |
| 4 | [0.003626, 2.984606, 2.207506, 0.680354] | [0.012455, 2.996667, 2.206629, 0.686132] | [0.019721, 0.062811, 0.026706, 0.015599] | [0.006972, 0.022207, 0.009442, 0.005515] | [1.27, 0.54, -0.09, 1.05] |
| 8 | [-0.031156, 7.464715, 3.791186, 0.794499] | [-0.020065, 7.544927, 3.789991, 0.818458] | [0.095259, 0.111183, 0.055550, 0.019770] | [0.033679, 0.039309, 0.019640, 0.006990] | [0.33, 2.04, -0.06, 3.43] |
| 16 | [-0.055212, 11.919296, 4.890724, 0.816862] | [-0.077593, 11.926735, 4.879317, 0.822271] | [0.104907, 0.254357, 0.113337, 0.025292] | [0.037090, 0.089929, 0.040071, 0.008942] | [-0.60, 0.08, -0.28, 0.60] |
| 32 | [-0.060243, 13.382248, 5.122027, 0.817509] | [-0.087544, 13.445222, 5.142386, 0.813951] | [0.181378, 0.348630, 0.110267, 0.036622] | [0.064127, 0.123259, 0.038985, 0.012948] | [-0.43, 0.51, 0.52, -0.27] |

All nonzero-gradient parameters were within 4 standard errors. Exact-zero
components were checked for exact zero estimates where the route structurally
produces zero.

## Discrete-only consistency

The manual public-Torx score bridge and native `BranchingSimulator`
`param_shift_filter` route were both compared with the exact Markov oracle at
depth 4 using 16 seed blocks of 4096 samples.

```text
manual bridge: passes 4-standard-error oracle criterion
native route:  passes 4-standard-error oracle criterion
bridge/native comparison: passes measured combined-uncertainty criterion
```

The native route remains separate and is not score-corrected.

## Required edge cases

```text
forward identity, scalar and per-trajectory: PASS
ambiguous shape rejection: PASS
per-trajectory batching/cross-coupling regression: PASS
constant loss: score contribution is zero only in expectation, not per sample
disconnected/zero downstream dependence: PASS through constant-zero case
all recurrence log probabilities accumulated: PASS
test-only omitted occurrence mutation fails oracle comparison: PASS
sample/log_probability consistency: PASS
near-saturation finite values: PASS for supported float64 range
native-route double-counting exclusion: explicit helper contract + separate test
```

The mixed oracle mismatch found during implementation was corrected before
estimator judgment: the prior M1 conditional-moment recursion failed to weight
additive moment terms correctly by the current-state probability. Raw Torx
forward Monte Carlo then matched the corrected oracle.

## Known limitations

The bridge is intentionally local and explicit. It does not infer which Torx
route is active and cannot prevent a caller from supplying log probabilities
for a native estimator. The supported contract therefore requires route
selection outside the helper and excludes native Torx estimators by policy.

