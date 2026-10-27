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
# Phase G2 prospective protocol V1 — warm-start generated commitment

Freeze/hash/commit this document, G3_PROTOCOL.md, identity and implementation
before counted updates. Parent accepted G commit7120e7de6e491e4028c92e8dabe31f3b92de7d70.
No old benchmark/evidence is amended. This is observation-latent-conditional
future-peer reconstruction, NOT prompt-only generation or scratch trainability.

## Immutable inputs and model

Task PHASE_G11_FUTURE_PEER_OBSERVATION_LATENT_V0, public query-payload erasure
BEFORE external frozen encoding; no Gaussian corruption. Original generator,
grammar and split identities remain unchanged. Train1140/32 per length1824,
validation1141/16 per length912; test1142/16 per length912 stays unconstructed
until G3 selection is durably frozen. IDs0/10000/20000 offsets. Existing tuple
hashes f439ec73d08a803115621d2f0dc831affc376c8e2491ae4fa510b047189a2205 /
b3f442b077dd38ce45906a91808686a0a5f1fa9560a3dfa736e7ca59ca1afedf /
734c261c182c6818347a27b975f0f453d3b6d2911b1cedab60e0a3bf92f6286e.

Warm-start ALL arms/seeds from G11 teacher100 full parameter tree
94f4b7de78406336b6e6bb87416981330a98e9d74b9bfe7c4280529807184bd2,
payload results/runs/phase_g/g1_1/teacher_v0/official/payloads/100_000.pkl,
SHA fa798be7d9d6244054b1053bd86c340c4d2f3582290dc72a31e6877c24f81a93.
External G01 codec results/runs/phase_g/g0_1/v1/frozen_codec.pkl SHA
213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7.
Copy means AND frozen rho bitwise; fresh Adam m/v=0, absolute step0 for each
arm/seed. No new parameter leaves or parameter draws. Predictor b_head/l_head
and shared core/output decoder were trained in teacher100; they are adapted
now, not newly initialized. External codec/target encoder/rho stay frozen under
the unchanged accepted stochastic_model_update_mask; output decoder trainable.

C32/h64/Lmax4/M32/K8, L4/Q3/S1/R0, draft block-causal attention, native condition0,
actual S0, noise0, lambda_op0. Full faithful Mamba/context/proposal/pack/DiT
attention+SwiGLU/unpool/residual/heads/Mamba decoder; no architecture ablation.
Two same-parameter applications: neutral predictor -> explicit commit -> replay
of IDENTICAL source h/prompt. No predicted-x0 re-corruption, S2 or R. Observed
b/ell are neutral constants in BOTH applications; only committed state routes.

## Arms and action law

A teacher b/ell (oracle control), B full-site stopped argmax, C full-site
temperature1 sample. All have identical initialization, optimizer, corpus/order,
loss weights, forward-application count and update budget. No teacher b/ell,
target bytes/h/masks/packing may enter B/C inference API. Loss/metrics may read
labels AFTER inference; A's teacher adapter is separately and explicitly typed.

Use existing commit_predicted_structure_core/full_commit_masks with reactivation
allowed, LOCAL_SCORE_BRIDGE ownership in C and no native estimator. Sample31
boundary edges and32 extent classes0..4 once each; terminal1 is never sampled,
keyed or scored. No target-conditioned action masks, rejection/resampling,
temperature tuning, threshold/hysteresis experiment or straight-through.
All semantic actions are domain-valid. Dynamic capacity policy V1 adds mechanical
cuts without changing semantic actions/probabilities; forced cuts add no keys or
score terms. M32 handles any32-site partition; invalid inputs still fail closed.
ell0 preserves query identity, removes KV/pooling/emission; reactivation allowed.

Keys: master JAX key12000. Stable named fold-ins distinguish parameter_init
(declared unused), dataset_order, training_structure, evaluation_structure,
operator, bootstrap and diagnostic. Fold training seed0/1/2 into each namespace.
Shuffle fixed IDs with epoch;57 full32-example batches/epoch. Structural context
includes optimizer occurrence, ORIGINAL global ID, actual S0, R0, commit0,
boundary/extent kind and site. Operator lambda0 context is shared by the two
applications (same nonstructure randomness), physical L/Q coordinates unchanged.
Evaluation roots are independent of training; root identity and global IDs bind
every shard. No Python/NumPy RNG in stochastic execution.

## Loss, generated lengths and ownership

Weights unchanged from G11: bCE1 over31 nonterminal edges, ellCE1 over32 sites,
predictor hMSE0, replay hMSE0.5 over32x64 to separate frozen clean target h,
global length-aware reconstruction1, query payload reconstruction1, proposal0.
All examples count equally, all classes remain supervised; no reweighting/tuning.

Flatten GENERATED emitted slots into global byte order using generated ell only.
Target flattening is loss-only. Global loss=(overlap CE+log256*(missing+extra)) /
max(nt,1), both-empty0; report overlap-only conventional NLL with its support.
Accuracy=correct/max(nt,ng,1), both-empty1. Never align wrong carrier slots.

Required G11 adaptation: target query payload global positions are determined
from target slot mask ONLY in loss (for this grammar4*q+1..3). Score emitted byte
j against true byte j. Missing query bytes cost log256 each and accuracy0;
divide by3. Report this length-aware query CE and overlap-only query NLL/support
separately; never call missing penalties conventional probabilistic NLL. Every
valid zero/partial/extra emission action has finite scalar loss. Exact query
requires all3 correctly present bytes; extra/missing global bytes are reported.
Nonquery means target bytes outside query carrier, globally aligned: missing
bytes count errors/cost log256; extra bytes count nonquery errors/cost log256.
Accuracy denominator nonquery target count+extra; loss denominator nonquery
target count. This equals the G11 nonquery definition when lengths match.
Report overlap-only nonquery NLL/support independently.

L_common=bCE+ellCE; L_desc=.5*replay_hMSE+global_loss+query_length_aware_CE.
A/B mean(L_common+L_desc). C mean(L_common+score_corrected_structure_loss(
L_desc, commit)). Baseline0, per-example selected-action LP summed ONCE before
batch averaging. Direct continuous gradients stay pathwise; only descendants
receive the local score correction, never supervised CE. No native double owner.

## Information-flow controls (distinguish two causal questions)

Generated predictors can legitimately see future labels where NEUTRAL predictor
attention reaches j. Therefore a recomputed policy may mediate information
through b/ell; do NOT assume the entire generated-policy cut experiment is
label-independent. Quantify that mediator response separately.
For the controlled downstream cut, hold the complete committed action fixed
across counterlabels, insert boundary q-1 (if q>0) and q to isolate query/prefix
from future j, keep ell fixed, and require unavailable PHYSICAL query-carrier
slots1..3 logit response<=1e-6. Global query alignment can point to another
carrier when lengths are wrong; do not claim that shifted information set is
chance-constrained. Global efficacy metrics still use the declared byte order.
This tests the replay path, not absence of information in the predictor.
End-to-end C/D-substitute negatives replace/erase future source payload BEFORE
encoding, recompute the entire policy with identical keys, and require exact
counterlabel invariance<=1e-6. No answer exists in those permitted observations.
Teacher-metadata mutation at fixed h/prompt/IDs/keys must leave B/C predictions,
actions, routing, emissions and logits bitwise unchanged. Report both controls;
never relabel a failed true negative as a pass. Cut control alone is insufficient
evidence of semantic value. No protected labels used in policy construction.

## Preflight before counted updates

Audit accepted main/G/source/checkpoint/codec and every historical manifest/hash.
Reuse immutable accepted108-action/21 MC and capacity-pushforward oracles if
their action/estimator sources are unchanged; rerun their integration regressions.
Require the immutable passing G21 complete42-coordinate adjudication (all three
108-action query-loss/direct-dependency fixtures;64 unused roots x4096 each;
exact known-variance/range Bernstein family alpha=.01 plus numerical tolerance;
2048 known-correct calibration experiments). Verify full evidence/source hashes,
all42 accepted components, LP/once-only ownership and independent keys. Preserve
V0's four-SE nonpass; no reroll, selected-component-only pass or gate waiver.
Full-model connected-vs-detached score difference nonzero, finite gradients and
discarded prospective AdamW update in EACH arm, shared-tree identity, exact frozen
leaves/moments and external codec. No trial state becomes training parent.
Measure actual C gradient variance over16 independent diagnostic roots on the
same32 training IDs chosen by named gradient_diagnostic_examples permutation;
report masked gradient mean norm/summed component variance globally and for
DiT/b/l heads/decoder. Repeat for selected C weights in G3. No optimizer update
or loss tuning follows those measurements.
Eager/JIT, scan/vmap where supported, actual pack/unpool identity, attention,
ell0/partial/reactivation, fail-closed invalid/overflow optimizer+checkpoint,
public-Torx parity and metadata/negative-label tests required. Failure blocks
dependent training until a non-protocol implementation defect is repaired;
protocol-breaking fixes require a separately versioned protocol, never V0 edits.

## G2 budget, validation and functional gate

Seed0, sequential A/B/C, EXACT maximum1000 new updates EACH. AdamW3e-4 constant,
wd.01, clip1, batch32, beta .9/.999 and epsilon accepted unchanged. Checkpoints
0/100/250/500/1000. No early scientific success stop or unannounced extension.
At every checkpoint full912 validation: A teacher, B argmax, C argmax AND actual
temperature1 sampling (four fixed eval roots0..3, average metrics NOT logits).
Record full structural/emission accuracy, active-boundary accuracy/F1 (active
truth sites, terminal excluded), vector exactness/per-class recall, reaches-j
with generated peer KV activity, byte/query/exact/nonquery metrics, invalidity,
entropy/churn, gradient norms/variance, timing/memory and teacher gap. Temporal
policy churn on fixed examples is distinct from neutral->commit site changes.
Validate matching teacher and cut controls; cut preserves the policy ell.

G2 selection uses common G3-eligible0/250/500/1000 (100 reported, not selectable).
For each arm, among finite/provenance-valid checkpoints satisfying nonquery
guards versus warm-start TEACHER baseline, choose smallest validation primary
query length-aware CE; ties within1e-8 choose higher query accuracy, then earlier.
No eligible checkpoint => explicit nonpass, no test-based substitution.
Nonquery BOTH accuracy drop<=2pp AND length-aware nonquery loss increase<=5%
against immutable teacher100 A validation (91.371542%,0.332729131).

A mode is functional only if active-b accuracy>=95%, exact emitted total length
>=98%, valid generated structures>=99%, true-peer reach>=95% of teacher reach,
query accuracy−matched-cut>=10pp, query length-aware CE<=.9*cut AND overlap-only
query NLL<=.9*cut, their paired95% CIs support the appropriate directions,
query accuracy>=1/28+.10, both competence guards and all correctness gates pass.
Root averaging is expected sampled-policy performance, NOT an ensemble decoder.
Use2000 paired whole-example length-stratified bootstrap replicates, fixed
bootstrap namespace, percentile2.5/97.5, preserve3 correlated query bytes/root
pairing. Missing-support overlap NLL is null, not0; insufficient support cannot
pass a NLL criterion. C sampled and C argmax decisions separate.

Proceed to G3 if ANY of B-argmax/C-argmax/C-sampled is functional and all common
correctness gates pass. Sampling advantage is NOT required. Otherwise record
G2 nonpass/unresolved and do not train seeds1/2 or open protected test.

## Persistence and failure rules

Every completed update has atomic durable full means/m/v/step/protocol state.
Two alternating latest slots plus immutable per-step receipts/hash-parent chain;
pointer/JSONL are derived recoverable views. Keep milestone snapshots immutable.
Old nonsnapshot ring states are explicitly transient, not claimed retained.
On crash recover only matching verified tip/contiguous receipts; retain orphan
attempts and reject stale/mixed source/runtime/config/data/seed/mask checkpoints.
No failed/invalid update advances state or emits a completed receipt. Fresh
optimizer means fresh only at initial0, NEVER reset at resume or stage transition.
Save state/frozen hashes and finite checks per receipt; snapshots at statedsteps.

One heavy runner with exclusive campaign lock; no unrelated process termination.
Normal CPU/swap contention allowed. Stop safely for allocation/JAX failure,
persistent unresponsiveness, critical disk<2GiB, persistence/correctness failure.
No S4, H/I/Thermo changes, codec/LM training. All old artifacts remain immutable.
