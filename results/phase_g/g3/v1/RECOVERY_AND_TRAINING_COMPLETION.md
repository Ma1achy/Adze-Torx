# G23 V1 — resource recovery and completed training

This record closes the operational recovery, not the pending scientific
evaluation. The frozen campaign completed **9,000 counted updates**: teacher,
stopped-argmax and sampled arms, each at 1,000 updates, for seeds 0/1/2.
The rejected extension was not activated. No seed or optimizer was restarted.

## Project-only storage recovery

See `results/phase_g/g2/v1/resource_recovery_20261027T1812/STORAGE_AUDIT.md`
and `RESUME_AUDIT.json`. The audit distinguished accepted evidence/checkpoints,
live full optimizer states and negligible project-owned disposable caches.
**Nothing was deleted, moved or deduplicated.** No unrelated user files were
scanned or removed. Available volume space recovered independently of this work.
The existing 2 GiB guard was not lowered or bypassed. The whole remaining budget
was 7 GiB, increased prospectively to 8 GiB for the frozen selected-validation
sixteen-root completeness evidence (see `EVALUATION_COMPLETENESS_NOTE.md`).
Approximately 79 GiB was free when the completed campaign was audited.

## Exact resumptions and retained failures

1. The original disk guard stopped before seed 1 B update 138, after 4,137 total
   updates. The immutable B137 backup is
   `results/runs/phase_g/g23/v1/resource_stops/20261027T170605/seed1_B_update00137.pkl`,
   SHA-256 `f82269e87457c71f92117102ed6a7d0eb76a827476b9532f18f9dc32dda9af0d`.
   The prospective loaded-state audit verified complete finite Adam moments,
   matching source/protocol/data/key identities and next-batch identity.
   The actual B138 receipt links directly to that B137 payload.
2. An agent terminal transport failure later caused `BrokenPipeError` in the
   stdout logger, **after** seed 1 C update 270 was durably recorded. This was
   not a numerical failure. Its immutable backup is
   `results/runs/phase_g/g23/v1/resource_stops/20261027T212940/seed1_C_update00270.pkl`,
   SHA-256 `51a912b43b0a8a13d35b4b85ad07034a41322b8a1920a88ae2cd0b4dfcabb2d8`.
   The actual C271 receipt links directly to it; full Adam state was retained.
   See `results/phase_g/g2/v1/transport_recovery_20261027T212940/`.

Both original failure records and failed-attempt output remain preserved.
The replacement supervisor writes directly to a regular project log, uses an
independent OS session and runs the single heavy worker at `nice -n 10`.
It changes no model, stochastic key, optimizer or scientific criterion.

## Completed campaign and validation-only selection

`process_supervision/transport_212940/EXIT.json` under the ignored run directory
records campaign return code 0 at **2026-10-28 05:41:29 UTC**. Its worker log
SHA-256 is `0757ef573b443d3d7b1dc216b42ca04a81fe614dbaf8d48141d8666fce13386e`.
The complete launch/exit metadata is copied into `TRAINING_COMPLETION_AUDIT.json`.

The audit checked all nine contiguous receipt chains, both durable ring slots,
all **45** milestone payloads, finite model parameters and complete Adam m/v
trees, moment shapes/dtypes, nonnegative second moments, zero initial moments,
and the same frozen-parameter/moment fingerprint at every milestone. All initial
mean trees equal the accepted warm-start tree. Every counted receipt reports
executed and unchanged frozen state. All 731 historical preservation entries
were rechecked; G21 evidence was verified by hash, **not rerun**.

Eligibility and selection were independently recomputed from validation:
nonquery accuracy and loss guards must both pass; valid routing must be >=99%;
eligible checkpoints are 0/250/500/1000; minimize query length-aware loss;
within absolute 1e-8, prefer higher query accuracy then earlier update.
Update 100 is diagnostic, not selectable. C uses its frozen four-root sampled
selection metric. No protected-test data had been opened at audit/commit time.

| Seed | Teacher A | Argmax B | Sampled C |
| --- | ---: | ---: | ---: |
| 0 | 1000 | 1000 | 1000 |
| 1 | 1000 | **500** | 1000 |
| 2 | 1000 | 1000 | 1000 |

`SELECTION.json` SHA-256:
`ee0df89871b3afa74c5243225270086c29a80b17c229abc9ef73f6b7314dd57c`.
`TRAINING_COMPLETION_AUDIT.json` SHA-256:
`05406c377fbac718f8d3cac55d2f0cf7e1af50cfacada3b35cb5bea56795a9f2`.
Both were committed before final evaluation at
`47cb6b5eeff23f2ff3e9f64658ae03ae566c8d51`.

All arm curves retain every update, including original pre-recovery updates;
their initial row is explicitly uncounted. `TEMPORAL_CHURN.json` reports
root-coupled physical structural changes, separately from neutral-to-commit
distance. `CAMPAIGN_RESOURCE_SUMMARY.json` records timing, raw/clipped gradient
norms and process RSS from the complete curves, with explicit measurement and
missing failed-attempt segment limitations. It does not infer missing wall times.

## Remaining closure gates

The selected-validation sixteen-root evidence is diagnostic and cannot change
selection. Next are the frozen protected-test paired interventions, sixteen-root
sampling, stratified/three-seed scientific decisions, final preservation and
complete repository validation. Interim validation accuracy is not a scientific
closure result. Any later integration remains conditional on those gates and
unchanged main; H/I/Thermo and S4 remain out of scope.

Static prechecks passed on the current source: formatting (187 Python files),
lint, type checking (0 errors/warnings), public dependency boundaries, and an
explicit AST private-Torx import scan (187 files; no violations). These do not
substitute for the pending regular, slow and integration suites.
