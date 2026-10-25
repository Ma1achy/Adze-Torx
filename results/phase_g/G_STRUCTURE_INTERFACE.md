# Phase G structure interface

The normative implementation record is
`docs/design/phase_g_generated_structure.md`.

Public types are `SampledStructure`, `StructureCommitResult`,
`GeneratedRoutingResult`, `StructureHysteresisState`, and
`StructureEstimatorOwner`. Explicit policies are prediction-only, stopped
argmax, sampled with temperature, and hysteresis with complete parameters.

The generated-routing adapter accepts explicit committed lengths and returns a
whole-batch `executed` gate plus per-example error codes. Invalid routing cannot
reach generated scatter or optimizer mutation. `ell=0` is non-emitting and a
later positive extent reactivates the same carrier identity.

For the bounded G task, mandatory K=8 cuts and the terminal sentinel are fixed
scaffold, while all optional nonmandatory edges and all extent sites are commit
owned. Sample log probability includes only those owned stochastic actions;
there is no validity-conditioned resampling.

Phase H integration prerequisite: consume only committed structure through this
adapter after `G_INTEGRATION_READY`; do not route from predicted logits.
