# G23 V1 — project-only storage and exact-resume audit

Recovery base: `45e8d2c71a9e8172a3c0282ed4789b1f5ea1bc05`.
No scientific protocol, model source, optimizer, common1000 cap or2GiB guard
changes. G21 is accepted; its evidence is read for identity, not recomputed.
The historical disk failure and original oracle nonpass remain immutable.

**Audit PASS:** complete loaded states and nonzero finite Adam moments verified;
the update137 backup matches live state bitwise. The recorded free-space readings
were18,241,306,624 bytes before and22,579,482,624 bytes after the audit (about17/21
GiB), above the7GiB budget. All4137 prior updates are accounted for. Source,
protocol/runtime, codec/parent, regenerated train/validation and namespace/next
batch identities match. No model forward, gradient, optimizer update, protected
test generation or file deletion occurred in this audit. No duplicate official
runner was found immediately before resumption.

## Project-owned files only

Read-only `du` inventory, rounded physical allocations:

| Area | Size | Classification / handling |
|---|---:|---|
| G worktree `results/` | 4.0GiB | Scientific artifacts, not disposable |
| G `results/runs/phase_g/g1_1/teacher_v0/` | 2.7GiB | Accepted teacher history/full optimizer states; preserve |
| G `results/runs/phase_g/g23/v1/` | 943MiB | Active checkpoints, evidence caches, trials and resume backup; preserve |
| G23 `seed0/` / `seed1/` | 491 /268MiB | Counted optimizer states and immutable milestones; preserve |
| G23 `resource_stops/` | 26MiB | Immutable update137 backup; preserve |
| G23 `preflight/` | 157MiB | Prospective-update state/provenance evidence; preserve |
| G `results/phase_g/` | 45MiB | Immutable committed scientific evidence; preserve |
| Main worktree `results/` | 4.9GiB | Accepted historical experiments; preserve |
| G Python `__pycache__` | 2,513,639 bytes | Recomputable, but immaterial to capacity; left untouched |
| G pytest/ruff cache contents | 43,787 bytes | Disposable test/lint cache, but immaterial; left untouched |

No project-local persistent JAX/XLA cache directory was found under the scanned
G worktree. No unrelated folders/files were inspected for cleanup. H/I and
Thermo are outside this task and untouched. No file was deleted, moved or
deduplicated. Identical accepted copies are NOT declared disposable merely
because they are byte-identical. All meaningful large consumers are protected.

The available space naturally recovered after the previous runner exited:
initial `df` read about7GiB; subsequent byte readings were9,584,640,000 and
9,216,000,000 (about8.93/8.58GiB). These are separate observations, not a claim
that our cleanup created space. The exact pre/post-audit readings are saved in
`RESUME_AUDIT.json`. Space remains shared and may fluctuate; the existing guard
continues to apply before each counted update.

## Whole remaining campaign budget, not just next update

Largest observed full checkpoint:27,478,202 bytes. Budget32MiB each, including
complete means and both moments. Remaining milestones:23 (three for seed1/B,
twenty for four not-started arms), plus eight new alternating slots and one
atomic staging payload:32 x32MiB =1GiB. This conservatively ignores the savings
from immutable hardlinks and bounds checkpoint sizes above the actual payload.

96 evaluation archives exist. Largest compressed99,086 bytes; largest raw ZIP
member total916,304 bytes, at most55 members. Frozen array shapes/cardinality
do not grow with training outcomes. Remaining checkpoint validation <=144
archives; G3 selected validation <=45 and full test522 => <=711 new archives.
Budget2MiB per archive including headers/metadata and slack, rounded to1.5GiB.
This is NOT a plan to add evaluation roots or discard/rerun failed results.

| Budget component | GiB |
|---|---:|
| Remaining checkpoints/rings/atomic staging | 1.0 |
| Evaluation archives and metadata | 1.5 |
| Reports, all remaining receipts and evidence exports | .5 |
| Resume backups / accepted integration fixture copies | .5 |
| Compiler/test/validation temporary reserve | 1.0 |
| Additional volatility reserve | .5 |
| Existing frozen minimum disk guard | 2.0 |
| **Total free headroom required before resumption** | **7.0** |

No cleanup is needed if the immediate pre-resume audit confirms >=7GiB free.
If it does not, stop and report the shortfall; protected data is not deleted.
This is an operational capacity budget, NOT a changed scientific acceptance
criterion or a reduction of the frozen guard. The measured8.58–8.93GiB is
sufficient for this project-owned budget, but cannot guarantee against unrelated
future writes. Monitor free space during the one sequential runner; preserve
exact state if the existing resource failure recurs. Do not launch concurrent
JAX model/evaluation/test jobs or repeat unresolved resource failures blindly.

## Exact resumption prerequisites

`scripts/audit_phase_g23_recovery.py` verifies the frozen source/protocol/runtime,
accepted parent/codec, immutable update137 checksum, five full receipt chains,
both live slots, every milestone checksum and full loaded Adam structures,
shapes/dtypes/finiteness/nonzero moments/nonnegative variances. It compares the
live137 full state with the immutable backup, and teacher/rho/moment frozen
fingerprints with the unchanged original initialization. It regenerates ONLY
train/validation data and checks frozen dataset hashes; it records the existing
key namespaces and exact next batch at optimizer occurrence137 (new update138).
It never executes a model forward, gradient, optimizer update or protected test.

After a passing audit and no duplicate runner, resume the existing official
campaign at reduced scheduling priority. Seed0 and seed1/A are already complete;
seed1/B continues from138 with existing moments. Preserve the rejected extension
decision, complete the remaining1000-cap arms, freeze validation-only selection,
then run the existing scientific evaluation/closure stages and required suites.
No architecture/loss/benchmark/threshold change, fresh optimizer, G21 resampling,
H/I/Thermo work or S4 is authorized by this recovery record.
