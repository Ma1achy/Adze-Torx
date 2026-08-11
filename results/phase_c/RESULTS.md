# Phase C — Adze-T zero-noise Torx parity

Base commit: `03d4677dca646b89719284775b035113f6fca6e8`.

Final decision: **TORX_PARITY_PASS**.

## Implementation

The existing Adze graph now accepts either `DeterministicOps` or `TorxOps`.
TorxOps invokes small project-local factors through public
`torx.AbstractReferenceFactor.sample()`. It never calls DeterministicOps and
does not wrap a complete Mamba, Transformer, or model inside a factor.

Occurrence keys fold a deterministic BLAKE2s module-path ID together with
evaluation/optimizer identity, `s`, `r`, `q`, and physical layer. Mean
parameters remain tied while repeated Q occurrences can receive distinct keys.

The semantic-path parameter map reports:

- deterministic parameters: `2,268,245`;
- Torx mean parameters: `2,268,245`;
- Torx stochastic-only rho parameters: `18,901`;
- total Torx parameters: `2,287,146`.

Every deterministic leaf maps exactly once to a Torx mean leaf. Stochastic-only
rho leaves intentionally have no deterministic counterpart. Round-trip mean
extraction is exact.

Direct Mamba coefficients (`a_log`, `d_skip`, `delta_bias`, `layer_scale`) are
mapped and accessed through a public mean factor but receive no speculative
noise law in Phase C:

`PHASE_D_DIRECT_PARAMETER_NOISE_POLICY_TBD`

## Parity summary

All recorded maximum absolute, relative, and RMS differences are zero for:

- primitive learned operators;
- C2A Mamba modules;
- C2B DiT blocks and all twelve recurrent applications;
- full initialized model forward and JIT forward;
- available trained target-codec/COPY/REVERSE checkpoints;
- Phase-B loss components;
- all mapped raw mean gradients.

Rho gradients are exactly zero. Root-key and rho perturbations are inert at
zero noise. Zero-noise instrumentation observes 237 public Torx factor
occurrences in the complete model.

## Validation

- `python scripts/run_phase_c_parity.py`: passed; regenerated all five parity
  records plus the semantic mapping record;
- targeted non-slow Phase-C tests: 23 passed;
- `make format-check`: passed, 94 files already formatted;
- `make lint`: passed;
- `make typecheck`: 0 errors, 0 warnings;
- `make test`: 73 passed, 1 deselected in 389.00 seconds;
- `make boundaries`: public dependency-boundary check passed;
- `make test-slow`: 1 passed, 73 deselected in 20.21 seconds.

No Phase B/B.1 history was modified. No architecture, Torx pin, Q/S/R semantics,
or deterministic computation changed. No finite-noise experiment was run.
