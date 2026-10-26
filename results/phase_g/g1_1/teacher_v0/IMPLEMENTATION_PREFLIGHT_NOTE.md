# Pre-counted-update engineering revision

The ed957f7 source passed the new-task gradient, prospective update, persistence,
rollback and flow gates with ZERO counted updates. Those evidence bytes are
retained unchanged in implementation_preflight_v0/; discarded trial states are
retained separately under implementation_preflight_v0_discarded_trial/.

Before official training, harden reporting for zero-hit negative controls:
relative accuracy against zero is undefined (null), never infinity or an invented
finite improvement. Absolute paired accuracy and all primary criteria remain
unchanged. Add explicit evaluation checkpoint/protocol binding and verify frozen
Adam moments at every update. No task, parameter, loss, optimizer, statistical
seed/resample rule or scientific gate changes. Re-run the required new-task
preflight under the revised, committed source; initialize official training only
from fresh exact means and fresh Adam0, never either discarded trial.
