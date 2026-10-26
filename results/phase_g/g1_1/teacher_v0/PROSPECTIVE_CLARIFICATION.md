# Authorized G.1.1 teacher100 — prospective clarification

Recorded before any new-task gradient, trial optimizer update or counted update.
User authorizes ONE100-update teacher experiment. Old protocol/evidence files
remain immutable. Parent:51c2f9b214c75b28c7ab27b7fdb3aefc8cece1d6.
Frozen bounded protocol SHA256:
35cbd1fed3b0490b73b83693c7246b01d5d41057295046b95d4022e514d8e29c.

## Only scientific-policy clarification

Both nonquery guards are mandatory, against the same fixed update0 A baseline:

    accuracy(A100) - accuracy(A0) >= -0.02
    AND NLL(A100) / NLL(A0) - 1 <= 0.05.

Neither condition may compensate for failure of the other. Endpoints inclusive.
All other frozen criteria remain unchanged, including exact1/28+.10 minimum A
query accuracy (13.5714286%, rounded13.5714% in the authorization).

## Prospective measurement/persistence implementation

Nonquery means valid emitted bytes OUTSIDE the query carrier. Query payload
metrics score its slots1..3; fixed query marker is reported separately. Also
report the complementary definition (all bytes except those3 payload bytes)
as a diagnostic, never select a guard definition after results.
Guards use equal-example means of each example's valid nonquery-byte accuracy
and NLL, matching the frozen validation convention; valid-byte-weighted values
are also reported separately. No support or weighting is selected afterward.

Evaluate A/B on all912 fixed validation examples, same inputs/weights/roots.
Full-validation C/D/E query diagnostics use seed1145 over the full912 rows;
D-outside reports query metrics only because nuisance observation bytes change.
The previously frozen64-example flow selection, roots and two28-symbol contexts
are reused unchanged for exact negative invariance and routing checks at init/end.
No protected-test corpus is constructed, inspected or evaluated by this runner.

Strata are fixed metadata, not performance selections:gap1..5; all57 lengths;
any/full outside emitting competitor; fixed-neutral K8 reachability; packets
with/without a final partial extent. Queries are always full extent; a partial-
query stratum has zero support, not an invented quality score. Retain local
full/partial nonquery-byte metrics too.

One new immutable journal receipt per completed update binds full params/m/v,
step, protocol/source/config/data/codec identities and parent-state hash. A
receipt commits the update only after payload persistence/hash verification;
latest pointer and JSONL are recoverable views. Interrupted payloads without a
receipt are retained as uncommitted attempts, never counted or overwritten.
Recover only a verified contiguous chain; reconcile a stale pointer/view to it.
Retain every payload for this bounded run, no checkpoint deletion. Scheduled
0/10/25/50/100 checkpoints reference immutable completed payloads.

Preflight persists and reloads a separate prospective step1, labelled NOT COUNTED.
Official training initializes again from exact means/fresh zero moments at0.
Do not initialize from this discarded trial or old teacher100/F1/F3 moments.

Endpoint100 is fixed, not validation-selected. No extension, task/loss/optimizer
tuning, sampled/argmax training, main merge/push, H/I work or S4.
