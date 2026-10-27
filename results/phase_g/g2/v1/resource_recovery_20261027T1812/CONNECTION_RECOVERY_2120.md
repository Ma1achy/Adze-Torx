# Monitoring connection recovery — not a training restart

The agent server restarted during campaign monitoring. On 2026-10-27 at about
21:20 UTC the existing process was verified alive: PID78055, nice10, command
`python scripts/run_phase_g23_v1.py --stage campaign`. Its original shell/PID78022
also remained alive. No duplicate official runner was found or launched, and no
process was signaled. No model, optimizer, checkpoint, root, hyperparameter,
protocol or scientific threshold was changed.

Durable stdout records seed1/C update250 completed at21:18:34 and its scheduled
validation. Matching evaluation shards continued to be persisted at21:19/21:20.
Five previous arms are complete at1000 each:5250 counted updates at this point.
The working branch was `009572925fc50f817c00afabbaea6c1f86595f33`; existing
uncommitted seed1/C evidence was preserved. G3 selection remained absent.

The new tool server no longer recognized old interactive session32859. Monitoring
was restored through the existing project log, without reopening/restarting the
runner. Do not invent a recovered shell exit status if that session cannot later
be polled. Campaign completion must instead be established from all nine cap
states, contiguous verified receipts, matching immutable snapshots, final
validation and frozen selection, plus absence of an active duplicate before any
new heavy stage. If durable evidence is incomplete, absence of a process alone
does NOT establish completion. Preserve any failure record and all logs.

Current monitoring log is unchanged:
`/private/tmp/g23_v1_campaign_resume_20261027T1812.txt`.
No G21 adjudication was repeated. H/I/Thermo and historical experiments were
untouched. This was a control/monitoring interruption, not a numerical failure
or an optimizer resume from an earlier update.
