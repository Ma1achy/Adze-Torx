# Torx integration contract

Torx is the stochastic execution substrate for Adze-T.

The integration boundary is deliberately strict because Torx is new and its public surface may evolve.

## Allowed relationship

```text
Adze-T model/compiler code
          │
          ▼
documented public Torx API
          │
          ▼
Torx runtime / simulators / eventual hardware
```

Adze-T may:

- instantiate public factor types;
- implement documented public factor interfaces;
- compose public `DFG`, `Site`, `ChainFactor`, `TiledFactor`, and related capabilities;
- call documented sampling/log-probability/simulator methods;
- use ordinary JAX transformations around documented Torx functions;
- inspect public metadata such as parameter-sharing keys when documented.

Adze-T may not:

- import private Torx modules;
- depend on internal class layout;
- modify Torx runtime behaviour;
- copy private simulator code;
- assume undocumented PRNG splitting;
- infer sharing when Torx already exposes an authoritative public identity.

## Training policy

Do not assume Adze-T needs an external stochastic-autodiff compiler.

M1 must inventory the public Torx/JAX gradient route actually supported for:

1. discrete stochastic transitions;
2. continuous stochastic transitions;
3. mixed graphs;
4. tied/repeated parameters;
5. recurrent composites;
6. tiled composites.

Where Torx supplies a native estimator, validate it against an independent oracle before using it at scale.

Where a route is missing, distinguish:

```text
model-local helper
public Torx feature request
reusable Temper-like compiler requirement
```

Do not jump directly to the third category.

## Public API drift

Tests under `tests/contracts/` own compatibility with the pinned Torx version.

When bumping Torx:

1. change the pin;
2. run contract tests;
3. run M1 oracle/regression tests;
4. inspect changed public semantics;
5. record the bump in `docs/DEPENDENCIES.md`;
6. only then merge.

## Composite-factor semantics

`ChainFactor` and `TiledFactor` are load-bearing for the intended Adze-T design.

The repo must maintain metamorphic tests comparing public composites against manually constructed equivalent programs for small cases.

This protects against changed recurrence semantics, incorrect parameter sharing, aux-axis mistakes, accidental RNG correlation, and graph/compiler regressions.
