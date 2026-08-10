# M4.4 decision

## M4_4_CORE_REDESIGN

Fixed-eta compute scaling produced a large apparent Q benefit, while eta/Q
fixed-horizon recurrence remained approximately neutral. The richer two-gate
state proxy also outperformed the minimal control under compute scaling.

However, the required conditioning control failed: correct, zeroed, and
shuffled operator conditioning produced nearly identical trained endpoint MSE.
The model therefore did not demonstrate reliable per-example operator
composition. The Q improvement is compatible with a generic contraction or
shortcut and cannot be accepted as evidence of faithful looped computation.

The higher-fidelity core requires redesign before drawing an architectural
recurrence conclusion. This is not a claim against recurrent Transformers in
general. M5 is not started.
