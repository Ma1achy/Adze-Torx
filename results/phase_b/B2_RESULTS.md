# B2 — deterministic surrounding topology

Status: **B2_MODEL_PASS**.

Implemented the deterministic shared byte frontend, context encoder, target
analysis encoder, SSM-style proposal, fixed-slot byte decoder, clean-state
heads, and pack → DiT → unpool forward path. The complete reference forward
passes eagerly and under JIT for `adze_reference_small_v0`, with fixed/teacher
committed structure and `S=1`, `R=0`.

No adaptive structure, stochastic backend, denoising rollout, refinement, or Q
science experiment was added.
