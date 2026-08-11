# C1 — primitive operator parity

Decision: **C1_OPERATOR_PARITY_PASS**.

Adze-T uses public Torx factors for affine/categorical projections, embeddings,
causal depthwise convolution, and direct learned SSM means. Continuous noisy
operators use one rho per output channel with `sigma_min=1e-6`,
`sigma_max=0.25`, and initial sigma `1e-3`.

At `lambda_op=0`, all tested primitive outputs equal Adze-D exactly. Changing
root keys or replacing rho with extreme finite values leaves outputs unchanged.
Raw mean gradients equal deterministic gradients exactly, and rho gradients are
exactly zero. The same contracts pass under JIT while tracing the stochastic
branch.

Full-model instrumentation recorded 237 factor occurrences and 145 unique
learned operation paths at zero noise. It includes the shared frontend,
context/target/proposal Mamba maps, combined delta/B/C projections, every
DiT Q/K/V/O and SwiGLU map, conditioning/modulation, clean heads, and decoder.
The frozen combined `dbc_proj` tensor is neither split nor reordered; delta, B,
and C parity follows from exact parity of its output slices.

Machine evidence: `parity/primitives.json` and
`parity/parameter_mapping.json`. Worst recorded primitive absolute error: `0`.
