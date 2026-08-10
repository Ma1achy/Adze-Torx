# M1 decision

**TORX_GAP_LOCAL**

## Evidence

Torx + JAX correctly handled:

- discrete recurrent stochastic gradients through depth 32 using the public
  `BranchingSimulator` parameter-shift/filter route;
- exact affine-Gaussian objectives and gradients through the public
  `AffineGaussianSimulator` route;
- public parameter sharing, `ChainFactor`, and `TiledFactor` semantics.

The mixed public forward graph also works, but no single documented public Torx
gradient mechanism covers a `HybridPCircuit` containing both a discrete
stochastic transition and a controlled continuous Gaussian transition.
Ordinary JAX gives the continuous pathwise derivative and zero discrete
derivative. `BranchingSimulator` rejects hybrid circuits, and
`AffineGaussianSimulator` rejects the controlled mixture gate.

## Exact gap

M1 needs a small Adze-T-local mixed estimator that combines a discrete score/
parameter-shift contribution with the continuous pathwise or moment contribution
for the mixed recurrence. This is a local integration helper, not evidence yet
for a generic stochastic-gradient compiler: Torx already supplies validated
routes for each constituent fragment and public probability/factor interfaces.

The helper was intentionally not implemented in M1; the spike stops at the gate
with the gap isolated and oracle-backed.

## Next milestone recommendation

Proceed to M2 only after reviewing and approving the small local mixed-gradient
helper contract. Do not add Temper, GenJAX/ADEV, or another compiler based on
this result alone.
