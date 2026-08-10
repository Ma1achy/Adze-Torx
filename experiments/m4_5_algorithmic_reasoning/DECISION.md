# M4.5 decision

## M4_5_CORE_REDESIGN

The exact arithmetic and register-machine generators are valid: arithmetic
uses a complete 12-digit/13-digit representation with exact carry-depth
buckets, and the register instruction family is bijective with broad fixed-
initial-state target diversity. Public-Torx Q=1/Q>1 forward and gradient paths
are finite and fixed-shape.

However, the predeclared fixed-width solvability gates failed. At 20 common
optimizer steps, shallow carry arithmetic at Q=12 achieved zero exact-answer
accuracy and approximately 20% digit accuracy in both all-at-once and cursor
modes. The k=4 oracle-cursor program achieved zero exact final-state accuracy
and at most 9.4% per-register accuracy, below the 25% gate. The short all-at-
once program Q smoke matrix was likewise near the random component baseline.

Because the easy exact tasks are not learned, the Q/depth results cannot be
interpreted as evidence for or against recurrent algorithmic reasoning. The
next work would need a focused core/task trainability redesign before another
Q sweep. M4.6 and M5 are not started. M4–M4.4 remain immutable.

Allowed decisions are `M4_5_ALGORITHMIC_Q_BENEFIT`,
`M4_5_ORACLE_CURSOR_BENEFIT`, `M4_5_ARITHMETIC_ONLY`,
`M4_5_PROGRAM_ONLY`, `M4_5_Q_NEUTRAL`, `M4_5_Q_NEGATIVE`,
`M4_5_CORE_REDESIGN`, `M4_5_OPTIMIZATION_UNRESOLVED`, and `BLOCKED`.
