# Phase E decision

Decision: **PHASE_E_RECURRENCE_NEUTRAL**.

The primary seed-0 matrices are numerically stable and all configurations
train to high COPY and REVERSE accuracy under lambda=1. E_REF is competitive,
but has no consistent endpoint advantage over the exactly parameter-matched
E_Q1 control or the depth-matched E_UNSHARED12 control.

Cycle truncation is flat or peaks at Q_exec=2 rather than improving monotonically
through Q_exec=3, and suppression of the q=1 recurrent delta produces no
material degradation. Effective-depth code changes are likewise small. The
trained recurrent core is therefore not demonstrated to provide a controlled
iterative-computation advantage in this experiment.

E5 repeats and E_Q1_COMPUTEMATCH were not run: the primary effect is neutral,
the causal tests do not identify later-cycle benefit, and their additional
runtime would not resolve the observed flat endpoint/causal pattern. This is
not a negative recurrence result and does not reopen Phase D.
