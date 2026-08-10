# B1 — real deterministic looped DiT

Status: **B1_DIT_PASS**.

The implementation contains explicit Q/K/V/O attention, persistent-coordinate
carrier embeddings, draft/refine masks, residual branches, SwiGLU up/gate/down
projections, and AdaLN-style conditioning. Four physical parameter sets are
reused across three recurrence cycles; no Q-cycle parameter copies or `1/Q`
residual scaling are used.

Validated behaviors:

- eager and JIT shape preservation;
- no later-block leakage in draft mode;
- global later-block influence in refine mode;
- inactive KV exclusion;
- finite, non-zero gradients through Q/K/V/O and all FFN projections.

The AdaLN/positional micro-details use the v3 documented provisional reference
defaults and are labeled as implementation defaults, not recovered endpoint
facts.
