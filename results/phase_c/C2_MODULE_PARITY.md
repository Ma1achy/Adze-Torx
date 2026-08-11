# C2 — module parity

External decision: **C2_MODULE_PARITY_PASS**.

## C2A_MAMBA_PARITY

Decision: **C2A_MAMBA_PARITY_PASS**.

One-block and configured multi-layer Mamba execution preserve the frozen
input/split, causal depthwise convolution, input-dependent delta/B/C, stable
diagonal selective scan, gate, output projection, residual, and masked-state
no-op semantics. Tests cover all-valid batches, prefix/tail masks, leading and
internal holes, consecutive holes, and JIT. Worst absolute error: `0`.

## C2B_DIT_PARITY

Decision: **C2B_DIT_PARITY_PASS**.

The four physical blocks remain distinct and tied across three Q cycles. The
Torx parameter topology contains no Q-indexed mean copies and is independent of
Q for fixed L. Draft/refine execution, all twelve `q/b` states, effective-depth
conditioning, inactive holes, all-K/V-empty attention, padding, persistent
RoPE, packing, and JIT match Adze-D. Worst absolute error: `0`.

Machine evidence: `parity/mamba.json` and `parity/dit.json`.
