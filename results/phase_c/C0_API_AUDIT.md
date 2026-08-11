# C0 — pinned Torx public-API audit

Base deterministic reference: `03d4677dca646b89719284775b035113f6fca6e8`.

Pinned dependency: `extro-torx` distribution version `0.0.1`, Git revision
`f1fc858ed950ecd41935d15c06d0ec7c5e0674ae`.

Public API used:

- `torx.AbstractReferenceFactor` for the project-local affine, embedding,
  depthwise-convolution, and direct-mean factors;
- the inherited public `sample(key, inputs, params, info, ...)` execution
  contract;
- ordinary JAX transformation of those public factor calls.

Each factor is small and local. Attention, SwiGLU, selective-scan recurrence,
Mamba discretisation, packing, masks, unpool, residuals, and Q orchestration
remain in the existing shared JAX architecture. A Torx-only DFG/Chain model was
not built because that would duplicate the frozen shared graph. Direct factor
sampling is sufficient for this zero-noise backend parity milestone.

The factor's `sample()` method owns the exact mean branch. The architecture
never substitutes `DeterministicOps` when `lambda_op=0`. Both branches of the
factor's `jax.lax.cond` trace successfully under JIT.

Module occurrence IDs use a frozen BLAKE2s 32-bit digest, not Python `hash()`.
Tests verify the same path identity in a separate Python process.

No `torx._*` import, private object layout, vendoring, monkey-patching, Torx
upgrade, Equinox import, or new dependency was introduced.

Decision: **C0_TORX_API_PASS**.
