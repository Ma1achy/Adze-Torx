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
