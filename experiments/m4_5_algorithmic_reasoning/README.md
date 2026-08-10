# M4.5 — algorithmic reasoning depth × recurrent compute

M4.5 tests the validated public-Torx recurrent substrate on exact arithmetic
and modular register-machine execution. M4–M4.4 remain immutable.

## Fixed semantics

Arithmetic uses 12 operand digits and a complete 13-digit result. Cursor cycles
`q=0..11` expose exactly one digit pair; `Q=12` is complete execution. `Q<12`
is incomplete exposure, and `Q>12` exposes only DONE/NOOP.

Programs use bijective operations over `Z_16`, explicit padding masks, and
lengths 4, 8, and 12. All-at-once conditioning is the matched-information Q
comparison. Oracle-cursor runs are execution tests trained with `Q>=k`; their
`Q<k` results are incomplete-information diagnostics, never Q-benefit claims.

The recurrent core uses public Torx `AffineGaussianGate`, two distinct physical
blocks, fixed eta compute scaling, tied parameters across Q, and deterministic
output heads. No categorical output is sampled into downstream computation, so
the M1.5 score bridge is not used.

## Decision metric

Human-readable results report exact accuracy. For matched all-at-once runs:

```text
exact_error = 1 - exact_accuracy
relative error reduction = (error_Q1 - error_Q) / error_Q1
```

A material improvement requires at least 5% relative error reduction and a
paired 95% interval excluding zero. When Q=1 error is near zero, absolute
accuracy differences are reported and relative ratios are treated as unstable.

Oracle-cursor success is judged against solvability and intermediate-state
thresholds at Q>=k. Post-completion Q>k is compared with Q=k. Q<k is reported
only as intentionally incomplete execution.

## Run protocol and solvability gate

The CPU run used a common 20-step convergence check for the focused solvability
cases, with validation every 10 steps, batch size 16, validation size 64,
Adam learning rate 0.03, and three matched seeds. A shallow fixed-width
arithmetic case was required to exceed 80% digit accuracy and 10% exact
13-digit accuracy; a k=4 cursor program was required to exceed 25% register
accuracy and 10% exact final-state accuracy. These thresholds are deliberately
above the 10% digit and 6.25% register component baselines.

The broader Q smoke matrix used the same initialization and optimizer for five
steps. It is descriptive only because the solvability gate failed at the
longer common check.

## Gates

The fixed-width shallow-carry arithmetic task and fixed-width k=4 cursor
program must be learnable before the full matrix is interpreted. Conditioning
destruction must substantially reduce performance, and program target diversity
must be nontrivial.
