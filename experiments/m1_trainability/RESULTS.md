# M1 RESULTS

**Status:** not started

## Environment

```text
Python:
JAX:
Torx distribution:
Torx version:
Torx commit:
Platform:
Precision:
```

## Public API inventory

```text
factor interfaces:
probability/log-probability capabilities:
continuous capabilities:
DFG:
Site:
ChainFactor:
TiledFactor:
gradient/simulator routes:
```

## P1 discrete recurrence

| depth | params | samples | exact grad | est mean | std | stderr | error/stderr | result |
|---:|---|---:|---|---|---|---|---|---|
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 4 | | | | | | | | |
| 8 | | | | | | | | |
| 16 | | | | | | | | |
| 32 | | | | | | | | |

## P2 continuous

Record analytic objective/gradient and observed Torx/JAX gradient route.

## P3 mixed

Record exact conditional-moment objective/gradient and estimate.

## P4 parameter sharing

```text
untied occurrence grads:
sum untied:
tied grad:
difference:
```

## P5 ChainFactor

```text
forward equivalence:
gradient equivalence:
weight sharing:
staging/compile notes:
```

## P6 TiledFactor

```text
forward equivalence:
gradient equivalence:
weight sharing:
```

## Known failures / surprises

None recorded yet.

## Recommendation

Pending.
