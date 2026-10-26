# Frozen prospective G.1.1 bounded teacher protocol V0

**Not execution authorization.** Enable only after G11_TASK_VALIDATED and a
separate explicit user authorization. If task validation is unresolved/invalid,
this conditional document remains blocked. No training is run in G.1.1 validation.
The original iid-byte G arms and failed thresholds are not amended or resumed.

## Question and controls

Can the unchanged shared denoiser learn future-peer retrieval under independently
observable, committed semantic structure? This is an observation-latent,
public-payload-erasure task, NOT prompt-only generation or a Phase-F denoising
experiment. A100 is a bounded optimization/necessity experiment, not full G science.

Use PHASE_G11_FUTURE_PEER_OBSERVATION_LATENT_V0 and the exact generator/grammar
in VALIDATION_PROTOCOL.md. Fixed corpus:1824 training(seed1140),912 validation
(seed1141),912 protected test(seed1142), balanced over57 lengths8..64. Immutable
tuple hashes are recorded in dataset_identity.json. No test metrics are exposed
during task validation or this bounded experiment. Fixed index/labels, no
resampling of successful examples, no duplicate source documents across splits.

Training observation query payload is blank4 before frozen encoding, while role
and length are retained; no Gaussian corruption is added. Source lengths/masks
are properties of declared observed bytes and enter only their frozen encoder.
Clean target bytes/h/b/ell are loss/control data, never predictor routing input.
All actual model target placeholders/observed structure remain fixed neutral.

## Exact initialization and topology

Copy accepted F1 means, overlay only frozen G.0.1 codec means, preserve all rho.
Initial tree84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb.
No teacher100 or F3 trained weights/moments. Fresh Adam first/second moments and
absolute step0. No new trainable leaves. F1 SHA and G.0.1 SHA are as in validation
protocol; codec external artifact and target-encoder weights stay frozen bitwise.
The output decoder is trainable under the accepted F/G update mask; rho updates0.

L4/Q3/S1/R0, actual-S0/condition0, lambda_op0. Two same-parameter applications:
neutral predictor → explicit TEACHER commit → replay of IDENTICAL source h.
No re-corruption, predicted-x0 trajectory, S2, R, architecture or decoder change.
Semantic b is defined by END markers; dynamic capacity cuts are mechanical only.
Full-site argmax/sampling are NOT trained or evaluated as scientific arms here.

## Explicit new loss version

PHASE_G11_TEACHER_QUERY_RECONSTRUCTION_LOSS_V0:

    L = boundary_CE(predictor, weight1, mean over31 nonterminal edges)
      + extent_CE(predictor, weight1, mean over32 sites)
      + 0.5 * h_MSE(replay, frozen clean TARGET h, mean overC×h_dim)
      + length_aware_global_byte_loss(replay, weight1)
      + query_payload_byte_CE(replay slots1..3 at public q, weight1)

Proposal auxiliary0. Predictor h-MSE0 is an EXPLICIT new-task amendment:neutral
predictor routing can exclude the future peer, so it must not be required to
predict answer-bearing target h before commitment. This is not a retroactive
change to old G/F losses. The replay has the permitted teacher route needed
for that target. No loss weight is tuned after validation observations.

Global emitted-byte flattening/alignment and missing/extra penalties retain the
accepted length-aware formula; teacher controls have equal true extents, so
overlap NLL equals reconstruction loss. Query CE is ordinary full256-way CE,
averaged over3 payload bytes; its fixed QUERY marker is excluded. No restricted-
alphabet renormalization is used for model performance. Clean target h/bytes
inform only loss/metrics; no clean state is injected into production input.

All gradients to shared means are ordinary continuous gradients. Teacher
structure is stopped discrete control; no score term is appropriate here.
Structural logits learn through predictor CE only. Native/local score-estimator
ownership remains unchanged for any separately authorized future sampled arm.

## Optimizer, budget, persistence

AdamW lr3e-4,wd.01,clip1,batch32,100 NEW updates, no extension. Shuffle fixed
training IDs by JAX seed1160 folded with epoch;57 complete batches/epoch.
No data augmentation/noise resampling; operator root1146 is paired and lambda0.
Checkpoints0/10/25/50/100; latest exact Adam state durable after every update.
Do not treat absolute step100 as continuation of F1's step5000.

Future authorized execution first requires finite first full-model gradient,
finite discarded trial update, frozen-state preservation and fail-closed
checkpoint checks. These are NOT executed by this validation milestone.
Bind every checkpoint to source/config/task/dataset/codec/F1 hashes, actual
optimizer count, full means/m/v, update mask, roots and completed-update flag.
Atomic payload/receipt writes; reject stale/mixed protocols or hash mismatches.
No invalid/nonfinite attempt advances state or gets a successful checkpoint.
Proposed new ignored directory:results/runs/phase_g/g1_1/teacher_v0/.

## Fixed validation and primary bounded gate

At each checkpoint report losses separately, b accuracy/F1/exactness, ell
per-class/total/partial/exactness, query3-byte accuracy/NLL/exactness, nonquery
accuracy/NLL, whole emitted-byte accuracy/NLL/exactness/MSE, churn/invalidity
and numerical state. A and B use identical observations, parameters, keys and
lengths; only the cut after q differs. Retain q/j IDs and all local supports.

Primary endpoint is fixed update100 (not validation-best checkpoint selection).
Report every earlier curve. Compare A to B and negative C/D-substitute/E;
D-outside is an independent nuisance robustness control. Evaluate C-post and
counterlabel/28-symbol invariance again under trained weights to rule out
alternative paths. Never add teacher information to a generated predictor.

A bounded teacher necessity pass requires all numerical/flow gates and:

- A−B query payload accuracy>=10pp AND mean query NLL(A)<=.90*NLL(B);
- paired95% intervals support positive accuracy and NLL differences;
- A query accuracy>=1/28+.10; negative graph invariance still holds;
- no >2pp nonquery accuracy loss OR >5% nonquery NLL increase versus update0 A.

These new query-focused10pp/10% thresholds are fixed before task neural
validation, based on the28-way chance/available-information contrast. They do
not replace old G's1pp/2% whole-sequence gate. Do not claim scientific superiority
of a commitment policy or trainability convergence from100 updates. A healthy
nonpass at100 is unresolved, not a G-negative verdict or permission to extend.

## Paired uncertainty

2000 paired bootstrap replicates, percentile2.5/97.5, resample WHOLE query
examples within each of57 length strata (16 draws/stratum), preserving A/B/C/D/E
pairing and their3 correlated payload bytes. Accuracy delta is the macroexample
mean difference. NLL absolute delta and relative change use differences/ratios
of aggregate means, not the mean of per-example ratios. JAX seed1161 for
accuracy and1162 for NLL, fold in declared control code B0,C1,Dsub2,E3,Doutside4.
Primary inference is A−B; remaining controls are reported, not searched for a
favorable label. No adaptive bootstrap settings or protected-test selection.

## Stop and recovery

Stop on invalid structure/overflow, hidden target leakage, failed invariance,
nonfinite loss/gradient/state, frozen teacher/rho change, protocol mismatch or
checkpoint failure. Preserve failed attempts and last matching durable state;
never silently reinitialize or discard moments. Ordinary CPU contention is not
a correctness failure; avoid concurrent heavy jobs. No automatic restart under
new source/protocol, and no extra budget to force a favorable result.

No argmax/sampled G training, H/I experiments, full G campaign, language/codec
training or S4. Stop after the separately authorized bounded teacher report.
