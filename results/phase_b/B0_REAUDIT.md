# B0 re-audit — backend boundary

Base commit: `c723f423d66aaed693d62fdf2fe4c451782e4e22`.

Decision: **B0_INTERFACE_PASS**.

The first-attempt B0 claim was incomplete: `DeterministicOps` existed, but major
learned transforms bypassed it. The corrected repository has one `apply_model`
graph parameterized by `LearnedOps`. Embeddings, affine/categorical maps,
depthwise-convolution parameters, selective-SSM learned parameters, all DiT
projections/modulation, carrier projections and heads, and decoder transforms
cross that boundary. Reshape, normalization, RoPE, attention algebra,
selective-scan algebra, masking, nonlinearities, and residual additions remain
explicit deterministic JAX.

`TorxOps` remains an intentional Phase-C `NotImplementedError`; no stochastic
backend or second model graph was added. A counting-backend integration test
executes the full graph and asserts calls in every major family.
