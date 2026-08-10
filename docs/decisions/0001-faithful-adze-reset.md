# ADR 0001 — Faithful Adze architecture reset

**Status:** accepted  
**Date:** 2026-08-10

## Decision

The project now treats the original non-Torx Adze architecture as the architectural source of truth.

There is one architecture and two computational backends:

- **Adze-D:** deterministic;
- **Adze-T:** stochastic/Torx.

Torx changes how learned computation is executed. It does not justify replacing the model topology.

## Consequences

The faithful baseline retains:

```text
bytes
 -> Mamba byte frontend
 -> SSM proposal
 -> persistent h/b/l carrier
 -> generated hard pack
 -> transient M x K block stream
 -> looped attention + MLP DiT
 -> unpool / carrier residual
 -> h/b/l prediction
 -> S denoising
 -> draft/select/erase/global-refine x R
 -> Mamba byte decoder
```

Hardware-near alternatives such as local stochastic couplers, removing hard packing, or replacing full attention are future ablations only.

## Prior M1–M4.5 record

Do not delete or rewrite prior results.

M1–M4.5 remain valid as **Torx substrate and simplified recurrent-proxy experiments**.

They must not be interpreted as evidence for or against recurrence in the faithful original Adze looped-DiT architecture.

The recurrence question is reopened pending the real looped Transformer implementation.

## Build policy

Fixed-compute correctness and deterministic/Torx parity come before scientific claims.

The first milestone in this scaffold is Phase A only: state, block construction, packing, masks, unpool, tests.
