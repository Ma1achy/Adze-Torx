# M1 RESULTS

**Status:** complete, review gate reached

## Environment

```text
Python: 3.11.8
JAX: 0.8.1
Torx distribution: extro-torx 0.0.1
Torx commit: f1fc858ed950ecd41935d15c06d0ec7c5e0674ae
Platform: macOS-26.2-arm64-arm-64bit
Device: CpuDevice(id=0)
Default precision: jax_enable_x64=False
Experiment precision: JAX_ENABLE_X64=1 for numerical sweep/oracles
```

## Public API inventory

```text
factor interfaces:
  torx.AbstractFactor, AbstractReferenceFactor, AbstractMatrixFactor
probability/log-probability capabilities:
  AbstractHasLogProbability, AbstractHasExplicitOutputDistribution,
  AbstractEnumerableOutputFactor, AbstractFiniteStateSpaceFactor
DFG/Site:
  torx.DFG, torx.Site, DFG.distribute_params, DFG.gather_param_grads
parameter sharing:
  Site.param_key; DFG initialises once per distinct key and scatters publicly
composites:
  torx.ChainFactor, torx.TiledFactor; weight_tied=True/False
PSC circuits:
  torx.psc.DiscretePCircuit, HybridPCircuit, PNOT,
  AffineGaussianGate, MixtureGaussianGate
simulators:
  StateVectorSimulator, BranchingSimulator, AffineGaussianSimulator
gradient routes actually used:
  discrete exact: ordinary JAX through StateVectorSimulator;
  discrete estimate: BranchingSimulator param_shift_filter;
  continuous: ordinary JAX through public AffineGaussianSimulator moments;
  mixed: continuous pathwise JAX only; no unified public mixed route.
```

All listed public symbols were present. No private Torx imports or fields were used.

## P1 discrete recurrence

Model: public `PNOT(0)` repeated with `DiscretePCircuit(reps=T)`. Independent law:
`K(theta)=[[1-p,p],[p,1-p]]`, `p=sigmoid(theta)`, `pi0=[1,0]`, cost `[0,1]`.
`theta=0.35`, 16 seed blocks, 4096 samples/block, `param_shift_filter`, float64 enabled.

| depth | exact objective | estimated objective | exact grad | grad mean | std | stderr | error/stderr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.586617579 | 0.598144531 | 0.242497395 | 0.242717320 | 0.004729927 | 0.001182482 | 0.19 |
| 2 | 0.484994790 | 0.471191406 | -0.084018149 | -0.084341636 | 0.005730067 | 0.001432517 | 0.23 |
| 4 | 0.499549687 | 0.498291016 | -0.005042840 | -0.005254275 | 0.008113252 | 0.002028313 | 0.10 |
| 8 | 0.499999594 | 0.507324219 | -0.000009083 | 0.005211463 | 0.011432455 | 0.002858113 | 1.83 |
| 16 | 0.500000000 | 0.499511719 | -0.000000000 | -0.006938836 | 0.011631896 | 0.002907975 | 2.39 |
| 32 | 0.500000000 | 0.497802734 | 0.000000000 | 0.003735790 | 0.017795630 | 0.004448907 | 0.84 |

All depth results were within the predeclared 4-standard-error criterion. Difficult
logits `theta=+-30` remained finite in the exact oracle.

## P2 continuous

Model: one public `AffineGaussianGate`, exact public `AffineGaussianSimulator`,
`H'=0.8H+0.25+N(0,exp(-1))`, objective `E[(H-0.7)^2]`.

```text
objective: 0.5703794411714422
analytic gradient [A,b,log_var]: [0.0, -0.9, 0.36787944]
Torx/JAX gradient:                  [0.0, -0.9, 0.36787944]
route: exact Gaussian moment propagation differentiated by ordinary JAX
```

This validates a differentiable exact-moment route, not sampled pathwise differentiation.

## P3 mixed

Forward model: public `PNOT` followed by public `MixtureGaussianGate`, repeated
in `HybridPCircuit`. Independent oracle propagates `P(X_t)`, `E[H_t|X_t]`, and
`E[H_t^2|X_t]` for `H'=alpha H+beta X'+sigma epsilon`.

```text
depth=4, params=[theta=.2, alpha=.8, beta=.4, log_var=-1]
exact objective: 0.87217536
exact gradient:  [-0.00395272, 0.77031668, 0.97391773, 0.42836197]
forward sample: finite and correctly shaped
ordinary JAX sample gradient: discrete theta component exactly 0.0
```

`BranchingSimulator` accepts only `DiscretePCircuit`; `AffineGaussianSimulator`
supports only affine-Gaussian gates and rejects the controlled mixture gate.
Therefore the pinned public API has no unified mixed stochastic-gradient route.

## P4 parameter sharing

For the exact binary recurrence at depth 4 and equal occurrence parameters:

```text
untied occurrence grads: [-0.00126071, -0.00126071, -0.00126071, -0.00126071]
sum untied:              -0.00504284
tied grad:               -0.00504284
difference:              < 1e-12 (float64 oracle)
```

The public `Site.param_key`, `DFG.distribute_params`, and composite
`weight_tied` semantics were contract-tested. Fixed-key sampling was exactly
reproducible; split keys differed.

## P5 ChainFactor

Public `ChainFactor` was tested against manual deterministic recurrence:

```text
depth=1 agrees with base factor: PASS
depth=4 manual recurrence agrees: PASS
weight_tied=True uses one parameter tree: PASS by public distribution contract
```

PSC stochastic recurrence uses public `DiscretePCircuit(reps=T)` and its public
`BranchingSimulator`; generic `ChainFactor` has no independent native gradient driver.

## P6 TiledFactor

```text
n_tiles=1 agrees with base factor: PASS
manual independent tile values agree: PASS
weight_tied=True public parameter behavior: PASS
```

## Pre-existing/scaffold failures

- Initial `pytest` failed collection because the editable package was not
  installed; `PYTHONPATH=src pytest` passed the original 3 tests.
- Initial format/lint checks failed on existing formatting/import ordering and
  one long comment. These were fixed as trivial scaffold issues.
- Initial Torx installation failed because Hatch direct references were not
  enabled; adding `tool.hatch.metadata.allow-direct-references=true` was a
  packaging-only fix.
- Pip reported unrelated global-environment conflicts from upgrading NumPy;
  they were not used by M1.

## M1 failures and limitation

The mixed forward computation is expressible through public Torx, but the
required mixed stochastic gradient is not. Ordinary JAX gives zero gradient
through the sampled discrete transition, while the two documented Torx
simulators cover disjoint discrete and affine-continuous fragments.

With float64 enabled, the pinned `HybridPCircuit` exposes an int32 top-level
discrete input spec while its public `PNOT` gate expects int64 input states.
The experiment uses the gate's documented public `input_ports` dtype and records
this compatibility quirk rather than relying on private layout.

No private API, copied implementation, RNG workaround, or tolerance relaxation
was used.
