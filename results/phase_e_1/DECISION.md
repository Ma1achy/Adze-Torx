# Phase E.1 decision

Phase E.1A benchmark diagnosis: **POINTER_V0_TOO_HARD**.

No final Phase E.1 scientific decision is issued by this calibration stage.
The earlier `PHASE_E_1_DIFFICULTY_Q_UNRESOLVED` entry was an interim runtime
status, not a completed milestone result.

Historical record from infrastructure commit `c638e95`: calibration had not
completed within that session's runtime window, so the working status was
`PHASE_E_1_DIFFICULTY_Q_UNRESOLVED`. Phase E.1A resumed those durable
checkpoints; it did not amend, squash, or discard that commit.

Both calibration models remained at chance through 5000 steps, while fresh
P_REF diagnostics strongly overfit fixed sets of 1, 8, and 264 examples.
POINTER_V0 therefore failed the held-out learnability gate under the approved
calibration regime. This must not be interpreted as Q neutrality, support, or
negativity. An authoritative Phase E.1 run was not started.

Phase E.1B then hardened the experiment framework and evaluated the predeclared
fixed-transition replacement. Its benchmark-local diagnosis is
**FIXED_TRANSITION_V0_TOO_HARD**: T_Q1 and T_REF remained at chance through
5k at every transition depth and Q execution count, while fresh T_REF models
strongly memorized fixed sets up to 256 balanced examples.

This second calibration failure is also not evidence for, against, or neutral
about recurrence. Task-useful Q0-versus-Q1 localization was not established,
so the authoritative same-model Q experiment remained gated off. No final
`PHASE_E_1_DIFFICULTY_Q_*` recurrence label is issued. Per the approved stop
rule, another benchmark will not be invented automatically; a scientific
design decision is required before Phase E.1 can continue.
