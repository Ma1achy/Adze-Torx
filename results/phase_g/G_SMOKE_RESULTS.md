# Phase G bounded smoke results

Status: not run; blocked by `G_CODEC_TASK_COMPATIBILITY_BLOCKED`.

No teacher, argmax, or sampled Phase-G smoke arm has started. The runner is
resume-safe and records checkpoints at updates 0, 10, 25, 50, and 100 with exact
protocol and checkpoint hashes. Sampled execution is gated behind durable codec,
engineering, oracle, and teacher-necessity evidence.

No authoritative Phase-G training and no scientific Phase-G claim are present.

In addition to the codec block, smoke is not protocol-ready: the proposal says
full-site argmax/sample boundary ownership, while implementation
`a5786fdaf542939439bf4ebb97973f870e063da4` fixes mandatory K-spaced and terminal
edges. This unresolved scientifically consequential mismatch requires approval
before any future smoke execution. The final non-scientific suites passed 233
regular tests (1 skipped, 30 deselected) and 30 slow tests (234 deselected), but
those results do not constitute protocol approval or a smoke result.
