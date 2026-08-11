# B1 re-audit — looped DiT

Decision: **B1_DIT_PASS**.

Corrections relative to the first attempt:

- effective depth is now consumed at every application as `q*L+ell`; reference
  execution observes `0..11` for `L=4,Q=3`;
- one shared conditioning trunk feeds a distinct modulation head for each
  physical block, with each head tied across Q;
- deterministic Q/K RoPE uses persistent carrier IDs;
- packed inputs include the v3 carrier/block/within, observed extent, and
  left/right observed-boundary embeddings;
- the provisional residual-gate default is frozen at `g0=0.1`, applied equally
  to attention and FFN gates through modulation-head bias initialization;
- exactly four physical parameter sets are reused for twelve applications;
  no per-Q parameters and no `1/Q` residual scaling exist.

Tests cover distinct physical blocks, Q-independent parameter count, manual
unroll equivalence, the exact effective-depth sequence, block-specific
modulation, persistent-coordinate RoPE, draft leakage, refine influence,
inactive K/V exclusion, JIT, and finite non-zero gradients for every physical
block's modulation/Q/K/V/O/up/gate/down paths plus packed input, conditioning,
carrier output, and h/b/l heads.

The known Phase-B K-bucket teacher produces `M=4` blocks. The compiled teacher
path removes the remaining all-padding M_max capacity blocks before attention;
`M_max=32` and its overflow contract are unchanged. Runtime/custom structure
retains full capacity. Tests establish numerical equivalence to the padded
execution.
