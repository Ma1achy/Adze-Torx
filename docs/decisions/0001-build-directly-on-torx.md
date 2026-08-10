# ADR 0001 — Build Adze-T directly on Torx before Temper

**Status:** accepted  
**Date:** 2026-08-09

## Context

A separate project, Temper, was considered as a generic stochastic-gradient compiler around Torx.

Further inspection of Torx showed that Torx is itself designed as a stochastic differentiable programming framework and already exposes public factor-graph, parameter-sharing, composite, probability, simulator, and training-related machinery.

The question "can Adze-T be trained?" should therefore be answered directly against Torx before creating another abstraction layer.

## Decision

Adze-T will initially depend directly on Torx.

Temper is deferred and is **not** a prerequisite.

M1 will test the actual difficult motifs:

- discrete stochastic gradients;
- continuous stochastic gradients;
- mixed computation;
- tied parameters;
- recurrence;
- tiling;
- exact/oracle agreement.

## When to revisit Temper

Only after a concrete failure produces a reusable requirement.

A `TEMPER_CANDIDATE` decision must include a minimal failing Torx program and independent oracle demonstrating the gap.

## Consequences

Positive:

- less infrastructure;
- faster route to the actual model;
- avoids duplicating Torx functionality;
- turns hypothetical requirements into empirical ones.

Risk:

- Torx is new, so public APIs/estimators may change or have rough edges.

Mitigation:

- pin Torx;
- contract-test its public surface;
- maintain exact stochastic-gradient regression tests;
- keep all Torx integration behind a clear public boundary.
