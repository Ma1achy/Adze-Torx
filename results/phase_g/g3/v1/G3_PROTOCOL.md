# G23 V1 — G21-adjudicated continuation

This separately versioned protocol replaces ONLY the G23 V0 new query-loss
Monte Carlo prerequisite with the complete42-coordinate finite-action
Bernstein procedure prospectively frozen and independently confirmed in
`results/phase_g/g2_1/G2_ORACLE_ADJUDICATION_PROTOCOL.md`.
G21 decision: G21_GRADIENT_ESTIMATOR_CONFIRMED; manifest SHA-256
0eb4e639db7c7ed3f5ab021b861a3ed0ce2c377bf8eff88e283873b67608544f.
The original V0 PHASE_G_BLOCKED and all its artifacts remain immutable.
No task, architecture, scientific threshold, model/loss/update mask, optimizer,
initialization, dataset, root semantics for TRAINING/EVALUATION, training budget,
checkpoint-selection or extension criterion is changed. Existing V0 core
training/inference code is reused unchanged. Only the statistical oracle gate,
versioned evidence/state paths and immutable adjudication dependency are new.
A source-conformance regression compares unchanged function ASTs across versions.
All new counted training starts at update0 after full-reference preflight,
from the SAME accepted teacher100 model with fresh Adam, not a V0 optimizer.
New results: g2/v1, g3/v1, runs/phase_g/g23/v1, G_FINAL_RESULTS_V1.md.
Do NOT overwrite original G_FINAL_RESULTS.md. Protected-test rules remain frozen.
The inherited two21-fixture oracles remain separate unchanged required gates.
New adjudication confirmation replaces, does not rerun/waive, the failed V0 gate.
No H/I/Thermo/S4 work. The original scientific protocol follows, with only its
statistical preflight clause replaced where applicable.

---
# Phase G3 prospective protocol V1 — three separate scientific axes

This protocol and G2_PROTOCOL.md form one frozen campaign identity. No protected
test generation/evaluation before validation-only selections are durably fixed.
Reuse accepted future-peer benchmark, not the rejected iid-byte task.

## Matched training and prospective extension

After G2 eligibility, seeds0/1/2, arms A teacher/B argmax/C sampled, identical
teacher100 warm-start tree/frozen codec/fresh Adam0 and exact G2 losses, roots,
two-application topology, update mask and optimizer. Source0 G2 trajectory is
reused, never retrained. Each additional seed varies data order/structural roots,
not the pretrained parameters or corpus. Base budget1000 per arm/seed.
The recommended2000 maximum is CONDITIONAL, not automatic available-compute use.

Determine common final budget from SEED0 validation alone, before seeds1/2:
extend ALL arms/seeds to2000 iff (i) both B and C primary query loss is smallest
at1000 among eligible checkpoints; (ii) from500 to1000 their averaged query
accuracy improves>=1pp AND averaged query length-aware loss drops>=2%; (iii)
both1000 checkpoints pass nonquery guards and no numerical/gradient alarm.
C primary mode is actual sampling averaged across fixed4 validation roots.
Otherwise final budget1000. Record exact predicate operands before continuation.
Seed0 extension resumes exact moments at1001; seeds1/2 start fresh0 and run to
the SAME chosen cap. No5000/10000 or other extension. At2000 still improving
is explicitly reported, not an excuse to tune/extend. Budgets are justified by
bounded CPU reference-scale study, not a general convergence claim.

Checkpoints0/250/500/1000/2000(if authorized by predicate), plus100 diagnostic
for consistency. Select each arm/seed independently by the EXACT G2 validation
eligibility/minimum primary query loss/tie rule. Primary mode A teacher/B argmax/
C actual temp1 root-mean. No test/MC-root-based selection, no favorable-seed
discard. Save selected path/SHA/means/moments/count, candidates/exclusions and
rule in an immutable selection record BEFORE opening test. Additional selected
validation16-root confirmation cannot reselect. Missing seed/cap incomplete =>
UNRESOLVED training reproducibility, not a single-seed bootstrap proof.

## Frozen evaluation

Full912 protected test, original seed1142/hash/IDs, and selected validation912.
Generated inference receives observation h/public prompt/IDs/key only. Targets,
true masks, b/ell/h0 and peer IDs enter ONLY metrics and explicitly named controls.
All deterministic interventions use identical weights/input/nonstructure keys.
Common generated extents are held fixed for boundary-policy comparisons; report
fully teacher b+ell as separate upper reference, not a pure boundary comparison.

For EVERY selected model/seed evaluate:
- teacher b+teacher ell;
- generated argmax b+ell;
- actual sampled b+ell over16 independent eval roots0..15;
- fixed-neutral K8 boundaries + argmax ell;
- shifted-valid TEACHER boundaries (roll nonterminal bits by1, terminal1) +
  argmax ell: explicitly labelled oracle-derived incorrect intervention, never
  a generated inference input;
- isolated query cut (edges q-1/q, terminal1) + argmax ell;
- valid q-anchored overconnected semantic span (q-1 and terminal only, publicq;
  dynamic K8 cuts apply) + argmax ell;
- no-attention and direct causal decoder + argmax ell;
- teacher-boundary + argmax ell (separates boundary/extent errors);
- cut teacher b+teacher ell (teacher reference matched-cut).
For each sampled root ALSO replay its matched isolated cut and neutral boundaries
with that SAME sampled ell. Report16-root mean/SD/range and paired example effects
after root averaging. Do NOT average logits/vote or compare an ensemble to one
argmax application. Root means estimate expected single-draw policy performance.
Terminal-only semantic boundaries with dynamic cuts equal the fixed K8 partition;
verify equivalence, do not pretend they are an independent baseline.
Q-anchored span is an observation-valid overconnected alternative, not a larger
architecture. It fits K8 dynamically, knows no teacher peer or target length.

Whole-model generated negative C/D-substitute and fixed-action downstream graph
cuts follow G2 information-ownership definitions. Retain true label-invariance
<=1e-6, metadata mutation invariance, no-attention/direct-decoder controls and
policy-mediated counterlabel response as a separately labelled diagnostic.
Never call an information-bearing recomputed policy a chance-constrained cut.

## Stratification, pairing and uncertainty

Mandatory neutral reachability computed from TARGET-INDEPENDENT K8 boundaries
and fixed carrier IDs. Validation expected780 reaches/132 discordant; verify,
never select by performance. Test membership computed before metrics, size may
differ. Also gaps1..5, each length8..64, outside competitor yes/no, block ordinal,
semantic boundary configuration (query block size and preceding cut), and
full/final-partial packet. Query is always full4; partial-query support0/NA.
Store supports, root IDs, source/global example IDs, committed semantics, capacity
cuts/routing and per-example metric vectors. Missing/extra/missing-query support
and per-class extent recalls stay visible. Invalid structures not discarded.

2000 paired whole-example percentile bootstrap replicates,2.5/97.5, independent
named bootstrap roots; lengths stratified within support present in subgroup.
Resample whole examples, not query bytes. All seeds and stochastic roots use
identical example resampling indices within a contrast. Primary aggregate effects
average PER-EXAMPLE metrics over roots and then3 training seeds; contrast NLL
relative is ratio of aggregate means. Report example-bootstrap and training-seed
variability separately. Seed mean/SD/range and Student-t95% CI with n3,df2
critical4.302652729911275; no treating3 seeds as912 independent training runs.
Endpoint mode/root/contrast identity folds into bootstrap namespace, no reruns
or root escalation shopping. Conventional overlap-only NLL and penalized query
loss remain distinctly labelled; all efficacy gates require both directions.
If a relative loss/NLL comparison has an exactly zero comparator, record the
relative statistic as undefined and the comparative axis as UNRESOLVED rather
than inserting a numerical floor, infinite gain or an invented equivalence.

## Axis decisions (fixed practical effects)

Engineering contract violation => affected axis BLOCKED, no dependent science.
Incomplete seeds/evaluation or mixed uncertainty => UNRESOLVED, never force sign.

Axis1 generated functionality: report each B-argmax/C-argmax/C-sampled against
G2 exact functional thresholds. A policy is reproducibly functional if at least
2 of3 seeds pass all point/paired-example gates, third has no material query
reversal, aggregate meets all thresholds/paired CIs, and seed t-CIs support
positive accuracy and negative query-loss effect versus matched cut. All seeds
must remain valid/finite/nonleaking/frozen. BENEFIT means functional generated
commitment, not superiority to oracle. NEGATIVE only if all generated modes
reliably fail useful query recovery with stable/nonimproving endpoint curves;
otherwise NEUTRAL or UNRESOLVED, preserve continuous gaps. Late-improving failures
are UNRESOLVED, not proof of architectural incapability.

Axis2 semantic routing: PRIMARY within selected B weights, argmax generated
extents fixed across all compared boundary interventions. Contrasts versus
neutral AND shifted valid cuts on overall and neutral-discordant examples.
Material benefit=accuracy+>=1pp AND query penalized CE/overlap NLL>=2% lower,
paired example CIs/seed t-CIs support both signs, at least2/3 positive seeds,
third no material reversal. A discordant-only benefit with no material pooled
harm is BENEFIT explicitly scoped to that subset, not universal semantic value.
Require agreement against BOTH neutral and shifted, not merely deliberate cut.
Q-anchored/no-attention/direct-decoder contrasts are separately reported and may
limit claims of uniqueness; no requirement to outperform every conceivable
valid routing to establish Axis1. If both primary contrast95% example AND seed
intervals lie within +/-1pp and +/-2% relative CE/NLL, NEUTRAL equivalence at
this scale/budget. Reliable converse on both controls =>NEGATIVE. Mixed or wide
intervals=>UNRESOLVED. C deterministic/sampled contrasts secondary, no swapping
primary strategy after observing outcomes. Empty/very small discordant support
is a stated power limitation, not a manufactured pooled benefit.

Axis3 sampling training value: PRIMARY selected C ARGMAX readout versus selected
B ARGMAX readout, paired seeds and examples, identical inference compute. Same
1pp/2% material thresholds, seed/paired-CI direction and2-of3/no-material-reversal
rule. Equivalence requires both example and seed95% intervals within those
practical margins. Converse=>NEGATIVE; mixed/underpowered=>UNRESOLVED. Separately
show actual C temp1 stochastic policy and B temp1 diagnostic, root variance,
entropy/structural readouts and score-gradient contribution variance. No benefit
claim from a favorable sample, extra evaluation averaging or one seed's CI.

Overall: PHASE_G_GENERATED_STRUCTURE_FUNCTIONAL only if Axis1 meets its real
generated-execution/reconstruction criteria. Comparative axes may be NEUTRAL
without invalidating functionality. If reliable functional failure=>
PHASE_G_GENERATED_STRUCTURE_NOT_DEMONSTRATED; incomplete/ambiguous=>
PHASE_G_UNRESOLVED; contract/protocol failure=>PHASE_G_BLOCKED. Record engineering
and trainability separately. These are current reference-scale conclusions,
not proofs about general language, all Q/S/R scales or asymptotic optimization.

## Closure and integration

Save all source/protocol/data/codec/init/mask/runtime/key identities, resumable
snapshots and selection records, per-seed/root/stratum metrics/uncertainty,
failure/skip ledger and historical preservation proof. Full format/lint/types/
boundaries/regular/slow/public-Torx/full-model tests. No thresholds weakened.
Incremental isolated G commits. If engineering/scientific claims fully supported
and repository clean, verify main still2a0cab0c9c647832596013d56037c842ff3844ea,
normal G merge, rerun integration validation, push if permitted. Main divergence,
conflicts/integration failure or unresolved overall science=>preserve G branch,
report blocker, no forced merge. No H/I/Thermo work, S4 or LM/codec training.
Record Phase-H integration prerequisites; do not integrate/train H. STOP.
