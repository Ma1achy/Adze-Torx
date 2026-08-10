# M4.5 results

## Environment

- Python 3.11.8
- JAX 0.8.1
- Torx `extro-torx` 0.0.1, commit
  `f1fc858ed950ecd41935d15c06d0ec7c5e0674ae`
- CPU `TFRT_CPU_0`, float32
- Public Torx primitive: `AffineGaussianGate`
- L=2, fixed eta 0.15, residual scale 0.10, total nominal variance 0.04
- Adam learning rate 0.03, batch size 16, validation size 64
- No score bridge; no discrete stochastic outputs were fed downstream.

M4–M4.4 records are unchanged.

## Arithmetic

The oracle uses 12 operand digits and a complete 13-digit result. Generated
carry depths were exactly 1, 2, 4, 8, and 12 in unit tests.

Five-Q all-at-once smoke results at carry depth 8, five optimizer steps, three
matched seeds, are shown as `exact accuracy / digit accuracy`:

| Q | seed 0 | seed 1 | seed 2 | mean digit accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.000 / 0.663 | 0.000 / 0.689 | 0.000 / 0.689 | 0.680 |
| 2 | 0.000 / 0.680 | 0.000 / 0.689 | 0.000 / 0.689 | 0.686 |
| 4 | 0.000 / 0.680 | 0.000 / 0.651 | 0.000 / 0.689 | 0.673 |
| 8 | 0.000 / 0.663 | 0.000 / 0.672 | 0.000 / 0.689 | 0.671 |
| 12 | 0.000 / 0.680 | 0.000 / 0.672 | 0.000 / 0.689 | 0.680 |

These are not a Q-benefit result: complete-answer accuracy was zero for every
Q and seed.

The fixed-width shallow-carry solvability check used Q=12 and 20 steps. For
all-at-once arithmetic, exact accuracy was `0.000, 0.000, 0.000` and digit
accuracy was `0.210, 0.213, 0.204` across the three seeds. The cursor version
was `0.000, 0.000, 0.000` exact and `0.210, 0.207, 0.195` digit accuracy.
Both failed the predeclared solvability gate.

The cursor convention was verified independently: Q=12 exposes all 12 digit
pairs, Q<12 exposes an incomplete prefix, and Q>12 exposes DONE only. The
13th final-carry digit remains in the target/output shape.

## Register programs

All 11 instructions were verified bijective over all 256 `Z_16^2` states.
For a fixed initial state and 4096 varying length-12 programs, all 256 final
register states occurred; mean per-register variance was 19.11. Thus the task
does not have the earlier conditioning-collapse problem at the oracle level.

At program depth 8, five-step all-at-once smoke results were:

| Q | mean exact final-state accuracy | mean per-register accuracy |
|---:|---:|---:|
| 1 | 0.0107 | 0.073 |
| 8 | 0.0107 | 0.068 |
| 12 | 0.0103 | 0.057 |

For reference, the random exact-state baseline is approximately 1/256 and the
random per-register baseline is 1/16. No positive Q effect is established.

The k=4 oracle-cursor solvability check at Q=4 and 20 steps produced exact
accuracy `0.000, 0.000, 0.000` and per-register accuracy `0.094, 0.047,
0.031`. It failed the 25% register / 10% exact gate.

## Diagnostics

All focused runs had zero non-finite failures. The recurrent state and output
shapes remained fixed, and parameter counts were Q-invariant within a task
layout. The Q=12 gradient integration test passed.

The solvability gates failed before a recurrence claim could be accepted.
Consequently no paired exact-error Q-benefit statistic, cursor execution
success claim, or post-completion stability claim is reported as a milestone
result. The smoke curves show decreasing arithmetic cross-entropy at some Q
values but not learnable exact execution; this is an optimization/core
diagnostic, not evidence of algorithmic reasoning.
