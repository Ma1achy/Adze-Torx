# Phase G scientific protocol proposal

This branch implements only the approved bounded engineering study. It preserves
the exact F1 parent means and frozen teacher/rho, initializes fresh Adam moments
at step zero, uses seed 703 for any new leaves, and leaves the architecture and
optimizer hyperparameters unchanged.

Task data are iid bytes 1..32 with lengths 8..64. Structural truth is
deterministic from codec content: mandatory K=8 cuts plus eligible content cuts;
extents are codec extents. Training uses 1024 examples, validation uses 256,
seeds 950/951/952, lambda-op zero, and `nu ~ Uniform(.025,.25)`.

The two applications are predictor plus same-input committed replay, both S1
condition zero and without re-corruption. Arms are teacher, full-site stopped
argmax, and temperature-1 local-score sampling, each independently initialized
from F1 and run sequentially for at most 100 updates.

Runtime gates may block the study. Passing a bounded smoke is not an
authoritative generated-structure result and does not authorize Phase H live
integration or Phase I training.

## Unresolved ownership discrepancy

The approved proposal above says `full-site` stopped argmax and sampling. The
current implementation through
`a5786fdaf542939439bf4ebb97973f870e063da4` instead preserves mandatory K-spaced
safety cuts and the terminal edge, and commits/samples only optional
nonmandatory boundaries; all extent sites remain owned. That implementation
choice prevents arbitrary sampled boundaries from destroying the packable
scaffold, but it has not been approved as a replacement scientific protocol.

Accordingly, this document records a mismatch rather than claiming
implementation conformance. The codec gate already blocks execution, and even
after that block is resolved, no smoke or full training may begin until an
explicit decision freezes full-site ownership or approves the mandatory-edge
scaffold variant.

JSON evidence is bound to implementation
`1a96675cab9bdc2d7c9c377b4f732d03c3124f1d`. The later dtype/test correction at
`a5786fdaf542939439bf4ebb97973f870e063da4` was validated non-scientifically but
did not regenerate scientific evidence.
