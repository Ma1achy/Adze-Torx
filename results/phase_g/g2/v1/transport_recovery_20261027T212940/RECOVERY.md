# Terminal transport failure and exact continuation

The server restart did not initially stop the original worker. It later exited
at21:29:40 with `BrokenPipeError(32, 'Broken pipe')` from the unchanged logger's
`print(..., flush=True)` at runner line99, called AFTER journal append at270.
`failure_001.json` preserves that exception/trace. The stdout tee lost its
terminal transport; the model/checkpoint operation did not fail. A later
read-only process check found no remaining official worker or its tee. Do not
infer success from that absence: the durable endpoint is seed1/C270, NOT1000.
Five earlier arms remain complete1000; total counted updates5270, next271.

Preserve the old stdout, failure and all receipt/snapshot evidence. Verify the
complete chain and actual loaded full means/moments; archive the exact270 full
state in a new immutable project-owned recovery path before continuation.
Do not repeat adjudication, gradients or training already completed. The
137 recovery backup and original disk-stop record remain unchanged.

## Operational fix, no scientific source change

`scripts/supervise_phase_g23.py` is a stdlib-only process supervisor. It verifies
the frozen source/protocol binding, uses the exact existing runner/stage and
nice10, and redirects worker stdout/stderr to a regular PROJECT file rather
than a terminal pipe. Both the supervisor launch and its worker use independent
OS sessions; their file handles do not depend on the agent connection. It
records immutable LAUNCH/EXIT metadata, exact command, source hashes, child PID,
log checksum and signed child return code. A separate supervisor lock prevents
overlapping supervised jobs; the unchanged worker campaign lock remains the
authoritative numerical-run lock. Attempt names are unique and existing logs
are never overwritten. There are no automatic numerical retries, budget flags,
optimizer changes, extra roots or altered scientific decisions.

The supervisor itself imports no model/JAX and only waits on its child.
Verification tests use a tiny Python child, not a model: regular-file stdout,
new process session, durable exit/checksum, unsupported-stage/budget rejection,
path-escape rejection, bound-source mismatch and attempt-reuse rejection.
Seven tests pass. One initial test run had6 pass/1 fail because the sandbox's
`nice` printed a permission warning into the fixture's expected stdout; the
I/O fixture now uses isolated Python directly, while a separate command test
continues to require production nice10. Actual worker priority is checked on
launch outside that fixture. This is not a model failure or relaxed numerical
gate. Formatting/lint/targeted typechecking must pass before launch.

All scientific computation stays in the unchanged5cb5e736... bound source and
approved datasets/root namespaces. The canonical worker performs its normal
full source/runtime/dataset/state checks before the next counted update. Its
cap remains1000 per arm; the rejected extension is not activated. No protected
test is opened during recovery. No historical or unrelated file is deleted.
