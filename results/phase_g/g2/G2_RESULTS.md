# G2 — learned structural commitment: blocked at prospective preflight

**PHASE_G_BLOCKED. Counted training: 0 updates.**

The prospective source, both scientific protocols and execution identity were
committed before preflight. The new query-loss peaked MC fixture failed its
unchanged acceptance criterion. No implementation defect has been identified
that can repair this failure without changing that statistical rule. Dependent
training stopped rather than waiving a gate or rerolling roots.

Rule basis: frozen `G2_PROTOCOL.md`, lines 118–138 at commit
`6e68e0f688261ec954ba718f9c7e8da0fa59ef47`, requires the new query-loss
exact/MC conformance before counted updates and blocks dependent training after
a failed prerequisite. `../g3/G3_PROTOCOL.md`, lines 144–162 at that same commit,
defines the blocked status and preservation of the isolated branch when gates
or overall science remain unresolved. The MC nonpass is not treated as evidence
of a biased estimator or failed architecture.

## Exact source, initialization and protocol

- Accepted starting G commit: 7120e7de6e491e4028c92e8dabe31f3b92de7d70.
- Prospective implementation/protocol commit: 6e68e0f688261ec954ba718f9c7e8da0fa59ef47.
- Identity commit / preflight launch HEAD: 10aec599e4ce08244e297d752c19ee5a770179fd.
- Frozen execution identity: 1d8ee3e970c9b8c98de1f027baa1adc4d3f03a941526238d91dbc827ad55a784.
- Warm-start candidate: accepted G11 teacher100,
  `results/runs/phase_g/g1_1/teacher_v0/official/payloads/100_000.pkl`,
  SHA-256 fa798be7d9d6244054b1053bd86c340c4d2f3582290dc72a31e6877c24f81a93.
- Full initialized parameter tree: 94f4b7de78406336b6e6bb87416981330a98e9d74b9bfe7c4280529807184bd2.
- Frozen G01 codec SHA 213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7.

See G2_G3_PROTOCOL_IDENTITY.json for exact source/protocol/configuration/dataset/
runtime/update-mask hashes. Fresh Adam 0 and matched three-arm training were
implemented but never initialized as official research runs. No F3 optimizer
state, new trained G2 parameter tree or selected G2 checkpoint exists.

## Gate evidence

| New query oracle | Exact 108-action gradient | Duplicate score rejected | 16 × 4,096 MC |
|---|---|---|---|
| Uniform | pass | yes | pass |
| Asymmetric | pass | yes | pass |
| Peaked | pass | yes | **nonpass: 1 component** |

The failed extent-site 0/class 2 component has absolute error 0.001083398238 and
estimated root SE 0.000243438757: 4.450393 SE, exceeding the frozen 4.0 bound.
Both old accepted 21-fixture oracle families and their source hashes remain
unchanged; the failed test is a newly added actual-query-loss prerequisite.

The separately committed same-draw diagnosis checks all 108 production-factor
log probabilities and production-bridge gradients; gradient discrepancies are 0.
All 327680 stochastic keys are unique. The rare extent class occurred 2 times
versus 8.0794 expected. Independently enumerated variance gives diagnostic
SE 0.000506495701 (error 2.13901 of those SE). This is consistent with empirical
SE underestimation under rare-event undercount, not evidence of wrong ownership.
**The diagnostic SE does not replace the frozen estimated-root-SE criterion.**

Full evidence: query_oracle_*.json, PREFLIGHT_FAILURE.json,
ORACLE_NONPASS_DIAGNOSIS.json and ORACLE_NONPASS_DIAGNOSIS.md. No extra roots,
new seeds, baseline change, criterion substitution or repeated-gradient-MC
confirmation was performed. Original draw identities were reconstructed only
to diagnose the saved nonpass.

## Training and evaluation inventory

| Arm | New updates | Scientific checkpoints | Policy performance |
|---|---:|---|---|
| A teacher continuation | 0 | none | not evaluated |
| B stopped argmax | 0 | none | not evaluated |
| C sampled | 0 | none | not evaluated |

Neither seed 1/2 nor the conditional 2000-update extension was started. No
validation-only checkpoint selection or protected-test evaluation occurred.
The accepted teacher100 remains immutable and is NOT relabelled a G2 result.

## Engineering scope and incomplete gates

Implemented observation-only generated inference, two shared-parameter S1
applications, full-site semantic commitment, separate dynamic capacity routing,
global length-aware query scoring, per-trajectory once-owned score correction,
strict evaluation shard identity, bounded-space resumable Adam journals,
matched training/selection/extension orchestration and G3 analysis.

Focused regressions exercise real small-model generated pack/DiT/unpool/decoder
execution, metadata-independent actions, connected/detached score gradients,
finite toy optimizer update, generated-byte order/missing emission, snapshot
immutability, checksum/partial-write recovery and statistical decision logic.
They do NOT replace the unexecuted full-reference, warm-started new-task gates.

Preflight stopped BEFORE full-reference parent gradients, prospective AdamW
updates, actual-parent zero-emission/reactivation tests and parent-model generated
negative-control/parity checks. Those are blocked, not silently passed/skipped.
Required repository validation and preservation results are in
../g3/VALIDATION.json and ../g3/FAILURE_SKIP_LEDGER.json.

## Interpretation

Generated-structure trainability remains unmeasured. No G2 functional, neutral
or negative conclusion follows from a finite-MC gate nonpass. The accepted G11
teacher result is intact. A separately versioned prospective uncertainty/
variance-control validation design may be needed before resuming; none was
executed to obtain a favorable gate. No main merge/push, H/I/Thermo changes,
LM/codec training or S4. STOP.
