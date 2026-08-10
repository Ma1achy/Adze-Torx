# M4 — Torx-native recurrent core and Q

Implement the intended stochastic transition family.

## Hard constraint

Do not hide a conventional DiT/Transformer/Mamba inside `Factor.sample()`.

## Experiments

Keep effective core depth approximately controlled:

```text
L=12,Q=1
L=6,Q=2
L=4,Q=3
L=3,Q=4
```

where feasible.

Compare:

```text
tied vs untied
cycle conditioning off vs on
local vs multiscale coupling
```

## Gate

- recurrence remains trainable;
- sharing invariants pass;
- Q changes compute without accidentally changing unrelated semantics;
- compile/runtime scaling is measured;
- no unexplained degradation from recurrent depth.
