# M4.4 results

## Environment

- Python 3.11.8
- JAX 0.8.0
- Torx `extro-torx`, commit `f1fc858ed950ecd41935d15c06d0ec7c5e0674ae`
- CPU `TFRT_CPU_0`, float32
- L=2, eta=0.15, residual/output scale=0.10
- total nominal variance=0.04; Adam 0.03; batch size 32
- three matched seeds; initial common budget 40 steps, extended common budget 80 steps

## Public Torx primitives

The public surface inspected included `AffineGaussianGate`,
`GaussianNoiseGate`, `MixtureGaussianGate`, `DeterministicFactor`, `DFG`,
`ChainFactor`, and `TiledFactor`. The selected recurrent stochastic operations
are public `AffineGaussianGate` instances. `tanh`, masking, and residual math are
ordinary JAX. No private Torx API was used. The richer block is manually
unrolled because deterministic nonlinearities occur between its two stochastic
gates.

## Layout and initialization

| region | width |
|---|---:|
| task | 8 |
| scratch | 24 |
| operator IDs | 48 |
| valid mask | 12 |
| total | 92 |

Parameter count was 34,592 for the faithful two-gate core and 17,296 for the
minimal one-gate control; both counts were invariant across Q within a family.
The linearized faithful control retained the same 34,592 parameters.

Initial deterministic displacement was approximately 2.5–5.7e-6 for Q=1–12.
All four gate-specific gradients were finite and nonzero. For faithful
compute-scaling Q=1 and Q=12 respectively:

| Q | displacement | gate gradient norm | optimizer update norm |
|---:|---:|---:|---:|
| 1 | 2.84e-6 | 3.21e-5 | 0.5051 |
| 12 | 3.41e-5 | 3.86e-4 | 0.5088 |

Initial residual/output scale and eta were identical across Q. Fixed-total gate
variance used the full `2*L*Q` denominator.

## 2×2 semantics/fidelity matrix

Extended 80-step best validation MSE at k=8, mean across three seeds:

| core | fixed horizon Q=1 | fixed horizon Q=8 | compute scaling Q=1 | compute scaling Q=8 |
|---|---:|---:|---:|---:|
| minimal | 0.24346 | 0.24350 | 0.24346 | 0.11792 |
| faithful | 0.23986 | 0.24187 | 0.23986 | 0.10604 |

The fixed-horizon family was approximately neutral. Fixed-eta compute scaling
produced a large apparent improvement as Q increased. At k=8, faithful
compute-scaling results were:

| Q | best MSE |
|---:|---:|
| 1 | 0.23986 |
| 2 | 0.21250 |
| 4 | 0.17309 |
| 8 | 0.10604 |
| 12 | 0.06339 |

At k=12, faithful compute scaling was 0.27270, 0.26592, 0.24188, 0.12104,
and 0.07412 for Q=1,2,4,8,12 respectively. The same directional effect was
present in the minimal core, though weaker.

## Progress pilot

The no-progress Q=8 pilot had mean MSE 0.10649; the shared q/Q pilot had mean
MSE 0.10572, an improvement below the predeclared 5% promotion threshold.
Progress was not promoted to a primary family. If used in future, test-time
progress must be `q/Q_test`; this changes both recurrent count and progress
schedule, so it cannot be sole evidence of compute scaling.

## Runtime and self-normalization

The faithful compute-scaling trajectory showed increasing effective computation
and decreasing endpoint error, but the current runner did not produce a fully
separate per-loop update table for every primary Q. The saved trajectories and
curves contain cycle state/update data for Q=1,4,8. This limits the strength of
the self-normalization conclusion and is recorded rather than inferred away.

## Critical conditioning control

At k=8/Q=8 with a trained faithful model, focused MSE was:

| conditioning | MSE |
|---|---:|
| correct | 0.24443 |
| zeroed | 0.24381 |
| shuffled | 0.24460 |

The dynamic output changed under a valid-ID mutation in the deterministic unit
test, and padded operator slots were correctly masked. However, training did not
make endpoint performance depend materially on the operator conditioning. This
means the apparent compute-scaling improvement is consistent with learning a
generic contraction/shortcut rather than composing the per-example program.

Scratch width 0 at k=8/Q=8 produced 0.24495 in the focused 40-step control. The
linearized faithful controls produced 0.02100 at Q=1 and 0.01545 at Q=8 in the
focused 40-step run, but these low values are not directly comparable to the
primary stochastic nonlinear endpoint table because they use a short focused
run and require the conditioning caveat above.

No score bridge was used. No routing, S, R, M5, Temper, or GenJAX functionality
was added. M4–M4.3 records remain unchanged.
