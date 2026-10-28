# Phase H handoff prerequisites — conditional, not an execution authorization

This record describes the current G interfaces and the remaining H integration
work. It neither changes H/I/Thermo branches nor authorizes H training. G must
first complete its scientific, engineering and integration gates. H must then
integrate against that validated G revision and pass its own real-model gates.

## Historical descriptions versus the current policy

The initial `G_STRUCTURE_INTERFACE.md` and
`docs/design/phase_g_generated_structure.md` describe the original, subsequently
blocked/rejected preflight task, including its fixed K-spaced scaffold. They
remain historical records; that scaffold is **not** the current G2 action law.
Current scientific authority is the frozen G2/G3 V1 protocols, accepted G.0.1
codec, future-peer task and adjudicated oracle identity.

The current production adapter is
`src/adze_t/phase_g0.py::dynamic_capacity_routing`. Every owned nonterminal
semantic boundary and extent is selected once. A deterministic scan adds a
capacity cut only when a physical run would otherwise exceed K. Such cuts have
conditional probability one, no semantic label and no additional score term.
They do not overwrite the semantic boundary state. The fixed terminal sentinel
is never sampled or score-counted. Invalid original semantic state is rejected,
not repaired by adding cuts. Do not restore unconditional K-spaced semantic
commit masks when wiring H.

## Concrete producer/consumer boundary

| G value/API | Shape/meaning | Ownership and H requirement |
| --- | --- | --- |
| `ObservedStructure` | observed embedding inputs at persistent C sites | Not routing authority; no teacher topology at generated inference. |
| `PredictedStructure` | boundary logits `[B,C,2]`; extent logits `[B,C,L_max+1]` | Provisional predictions; never silently use as committed topology. |
| `SampledStructure` | selected categorical actions and selected-action log probabilities | Preserve action identity and score accounting; no score for forced capacity/terminal actions. |
| `StructureCommitResult` | previous/committed structure, change/reactivation masks, validity/errors and policy/owner | Structural-only return. A commit callback must not replace controller-owned h. |
| `CommittedStructure` | binary boundary-after-site, explicit extent `[B,C]`, activity `ell>0` | Persistent identity survives `ell=0`; H must supply explicit valid lengths. |
| `CapacityRouting` | `semantic`, separate `capacity_cut`, validated physical `routing` | Keep semantic versus physical edges distinct in traces and subsequent commits. |
| `GeneratedRoutingResult` | structure, pack metadata, per-example validity/errors, whole-batch `executed` | Repack after each authorized commit; never consume stale metadata or execute an invalid batch. |

These are real registered pytrees/array-based cores, not permission to bypass
their shape, dtype, domain, terminal, activity and capacity checks. Use the
device execution gate under JIT and the validated host boundary. Failed
execution must preserve model/Adam/step and must not be recorded as completed.
Pooling, KV and emission are inactive at zero extent, but persistent sites remain
query-active and explicitly eligible for reactivation under the chosen policy.

## Decisions H must freeze, not inherit by accident

- G2 uses full-site post-prediction commits. H partial commits require distinct
  extent-site and boundary-edge ownership. `closed_commit_masks` implements the
  proposed rule that an outgoing edge changes only if both adjacent sites are
  selected; the terminal remains forced. H must explicitly approve/freeze its
  region-closure and edge-preservation semantics with both boundary-side tests.
- Selection budgets must explicitly mean seeds or expanded regions. Halo
  expansion must not silently violate eligibility, inactive-reactivation or
  budget rules. Freeze reset/erase content and commitment semantics separately.
- H must define valid UNKNOWN representations for observed embeddings and any
  required parameter-row migration, preserving all accepted existing rows.
  `-1`, a wrapped/clamped row, an actual zero extent, or a neutral known topology
  is not an interchangeable UNKNOWN. Committed extents remain valid emission
  classes, not UNKNOWN classes.
- Threshold/hysteresis code is engineering infrastructure, not a scientifically
  selected G2 policy. H must separately freeze thresholds, confidence state,
  dwell/reversal behavior and estimator ownership if it uses them.

## Keys, gradients and parameter sharing

`StructureKeyContext` carries root, occurrence, original global example IDs,
actual S index, refinement index and commit occurrence. It derives independent
kind/site keys. H must add its real R identity and preserve actual-S versus
learned-conditioning indices, with Q cycle/physical-layer/operator occurrences
independent. Do not use G3's zero-operator-noise diagnostic wrappers as a general
R-loop key implementation.

Preserve once-only per-trajectory selected-action score correction and
`StructureEstimatorOwner` exclusion of simultaneous local/native ownership.
Continuous dependencies remain pathwise. The accepted G oracle covers
structural b/ell decisions, not a newly learned stochastic region selector:
such a selector needs its own loss-dependency/estimator validation if trained.

Draft and refinement must use the **same** heavy-core parameter tree, with
block-causal draft and global-refinement communication. Each R application must
execute its declared complete S trajectory. G's predictor/commit/replay is two
S=1 applications to the same source carrier, not S=2 and not an R controller.
G's learned conditioning code is 0; do not silently introduce untrained S codes.

## Required real integration evidence

Before H science, independently verify initial/draft/reset/refined/committed/
final validity; actual causal/global masks; shared parameter identity; complete
S execution inside each R iteration; validated repacking; inactive reactivation;
and bitwise preservation of every unselected h/structural/optional prediction
field required by H's controller contract. Prove that structural-only commits
cannot alter h. Trace selections, erased regions, supports/validity, changes,
collateral effects, semantic/capacity edges, churn, example IDs and R/S/Q/operator/
commit occurrences. Empty support is unavailable, not a meaningful zero.

Run H focused contract tests before applicable static, boundary, broader and
full-model tests, with public Torx only. Mock callback identity does not prove
real model sharing, mask semantics, differentiability or trainability. Initial H
experiments require a separately frozen and authorized protocol after those
gates. Neither a G merge nor this document completes H integration or science.
