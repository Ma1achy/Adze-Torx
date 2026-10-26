# Amended G preflight — engineering remediation

## Outcome

`G_ENGINEERING_CHANGES_REQUIRED`

`G_INTEGRATION_BLOCKED`

The amended implementation passes its focused correctness, independent
estimator, real-model gradient/update, and historical-preservation gates.
However, the mandatory post-teacher structural-necessity gate fails. Correct
semantic boundaries do not outperform shifted valid boundaries. This blocks
generated-arm training and integration; it is **not** a scientific negative
result for generated structure, nor a codec failure.

Execution source: `f571ab40659da872777f4deccbfcd05531c4c576`, descended from
approved G revision `6537b2e9c81ab660ad6da318a4627487f6add831`.
The prospective [protocol](PROTOCOL.md) was committed before runtime evidence.
All new execution evidence is in [f571ab40659d/](f571ab40659d/).
Historical F, G.0, G.0.1 results and earlier failed-G records are not rewritten.

## Changes and evidence

| Correction | Implementation | Evidence |
| --- | --- | --- |
| Bind exact G.0.1 codec, F1 parent, matched means and fresh Adam | `scripts/run_phase_g_amended.py:manifest`, `initialized`, `audit` | `provenance.json`; all three declared initial hashes identical; only teacher arm executed |
| Full-site commitment with dynamic capacity cuts | `src/adze_t/phase_g.py:generated_train_step`, `evaluate_committed_replay`; amended runner passes `dynamic_capacity=True` | `production_oracle.json`; real full-model traces; amended eager/JIT tests |
| Finite atomic optimizer update | `fail_closed_adamw_step`, `generated_train_step` | `tests/test_phase_g_amended.py`; prospective B32 update in `full_model.json` |
| Invalid attempts never complete/persist a new update | amended runner `train_arm`, hash-bound `save_current`/`load_current` | real invalid-update controller regression; immutable receipt/stale-protocol tests |
| Separate committed correctness from predictor accuracy | `generated_two_pass_objective` diagnostics | every checkpoint retains both sets of fields and per-example predicted/committed b/ell |
| Undefined overlap support made explicit | generated-byte diagnostics | zero-emission regression; overlap byte/example support saved |
| Actual post-teacher controls, paired uncertainty | amended runner `necessity` | `necessity.json`, including all 256 examples and all four controls; failed shifted cuts retained |
| Restore historical integration fixtures | hash-verified no-overwrite import from canonical storage | `fixture_import.json`; no accepted file regenerated |
| Preserve accepted artifacts | before/after inventories | `historical_baseline.json`, `historical_preservation.json`: 727 unchanged hashes |

No architecture component, loss weight, optimizer hyperparameter, score-bridge
semantics or historical F interface was replaced. Legacy fixed-capacity entry
points keep their defaults. New G execution explicitly selects the amended
dynamic policy. No threshold/hysteresis experiment was run.

## Validation and limitations

Focused execution: 88 tests passed in 176.83s. Whole-project validation:
258 regular tests passed in 1846.04s; 30 slow tests passed in 107.77s;
zero skips. These 288 unique whole-project tests include the focused subset.
Formatting (145 files), lint, typing (zero errors/warnings), boundaries and
explicit private-Torx scan all pass. Exact commands, results and deliberately
unexecuted dependent stages are in [VALIDATION.json](VALIDATION.json).

Arm A completed 100 updates, with checkpoints 0/10/25/50/100. Arms B/C have
**zero training updates** and no arm checkpoints. Their inference-policy
measurements on teacher-trained weights are explicitly not independently
trained-arm comparisons.

Required next work is a separately approved resolution of task-level semantic
boundary necessity. Do not tune the gate, change labels or train longer to
obtain a pass. Do not merge this blocked milestone into main. H/I remain paused.
No full G campaign, H/I training, codec training or S4 was executed.

Checkpoint receipts, finite arrays/moments and fresh update0 state passed a
separate [artifact audit](ARTIFACT_AUDIT.json). Twenty independent checkpoint/
evidence copies were hash-verified under the durable ignored archive:
`/Users/malachy/src/Adze-Torx/results/runs/phase_g/preflight_g01_v1/execution_f571ab40659d_archive/`.
The final post-suite preservation result is in `FINAL_PRESERVATION.json`.
