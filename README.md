# Adze-T faithful scaffold

This repository scaffold is for the **faithful Adze architecture reset**.

The canonical architecture is:

- **Adze-D** — deterministic reference backend
- **Adze-T** — Torx/stochastic backend

Both must implement the same architecture. Torx changes the computational substrate, not the model topology.

## Source of truth

Read, in order:

1. `docs/architecture/adze-architecture-v3.md`
2. `docs/decisions/0001-faithful-adze-reset.md`
3. `AGENTS.md`

The architecture document is authoritative. If code and prose disagree, stop and resolve the discrepancy rather than silently changing the architecture.

## Initial implementation milestone

The scaffold is intentionally incomplete.

Implement **Phase A only** first:

- carrier/state types;
- observed/predicted/committed structure;
- boundary -> logical-block construction;
- hard-pack metadata and maps;
- draft/refine masks;
- unpool;
- tests for all of the above.

Do **not** implement Mamba, DiT, training, Torx learned operators, denoising `S`, or refinement `R` before the Phase A gate is green.

## Intended progression

```text
A  state / blocks / pack / masks / unpool
B  deterministic Adze-D
C  Adze-T zero-noise parity
D  finite stochasticity
E  real looped-Transformer Q study
F  denoising S
G  generated/mutable b,l
H  draft/select/erase/global-refine R
I  natural byte generation
```

## Torx pin

The scaffold pins Torx to:

```text
f1fc858ed950ecd41935d15c06d0ec7c5e0674ae
```

Use the public `torx` API only. No `torx._...` imports.

## Test status

The Phase A tests are intentionally written as executable contracts. The initial scaffold contains `NotImplementedError` placeholders, so these tests are expected to fail until Phase A is implemented.
