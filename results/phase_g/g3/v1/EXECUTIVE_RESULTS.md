# Phase G23 V1 — generated structure at the reference scale

## Scientific outcome

**PHASE_G_GENERATED_STRUCTURE_FUNCTIONAL**

| Preregistered axis | Outcome | Scope |
| --- | --- | --- |
| Generated b/ell functionality | BENEFIT | Real generated commits support useful future-peer reconstruction in all three training seeds. |
| Semantic routing value | BENEFIT | Within the selected argmax-trained weights, learned routing beats the frozen neutral and shifted interventions, including the neutral-discordant subset. |
| Sampled versus stopped-argmax training | UNRESOLVED | No demonstrated sampling-training advantage or practical equivalence; mixed training-seed effects and wide seed uncertainty. |

These are three distinct conclusions. Sampling superiority is not required to
establish functioning generated structure. This is a **warm-start, low-alphabet,
masked-observation conditional retrieval benchmark**, not scratch trainability,
prompt-only generation, natural language, or universal architectural superiority.
The original rejected benchmark and the original G23 V0 `PHASE_G_BLOCKED` record
remain unchanged. Current results are separately versioned in `g2/v1`, `g3/v1`
and `G_FINAL_RESULTS_V1.md`.

Repository validation and conditional integration are recorded separately; the
scientific label alone does not assert that those delivery gates have passed.

## Fixed experiment and selection

All nine arms used the same accepted G.1.1 teacher100 warm-start means, frozen
G.0.1 codec, fresh zero Adam moments/step 0, corpus and loss weights. Training
seeds 0/1/2 vary ordering and structural draws, not pretrained weights or data.
The external codec, target encoder and rho remained frozen; the approved output
decoder remained trainable. AdamW: lr 3e-4, weight decay .01, clip 1, batch 32.

L=4, Q=3, S=1, R=0; C=32, h=64, L_max=4, M_max=32, K=8; operator lambda=0.
Each update performs a neutral-routing predictor, explicit commit, and committed
replay with the **same parameters and original observation carrier**. Both use
conditioning code 0. There is no predicted-x0 re-corruption, S=2 or R refinement.
All owned nonterminal semantic boundaries/extents are selected once; deterministic
capacity cuts remain separate and receive no semantic label or extra score term.

Training: 1,824 examples (seed 1140). Validation: 912 (1141). Protected test: 912
(1142), 16 per length 8–64. The test was opened only after all nine caps and the
validation-only selection were committed. Every stochastic evaluation uses the
same 16 prescribed roots, paired inputs/IDs and controls; no logit ensembles,
favorable-root selection or failed-draw exclusions.

**9,000 counted updates completed**, exactly 1,000 per arm/seed. The prospective
extension was rejected: seed-0 averaged 500→1000 accuracy improvement was only
0.2924 pp, below the required 1 pp AND loss-improvement gate. No 2k extension ran.
The cap is a bounded reference study, not a claim of asymptotic convergence.

| Seed | Teacher A selected update | Argmax B | Sampled C |
| --- | ---: | ---: | ---: |
| 0 | 1000 | 1000 | 1000 |
| 1 | 1000 | **500** | 1000 |
| 2 | 1000 | 1000 | 1000 |

`SELECTION.json` contains every candidate, eligibility flag, value, tie rule,
path, full payload/mean/Adam hash and selected step. Update 100 was diagnostic,
not selectable. Additional 16-root validation did not reselect any checkpoint.

Critical immutable identities:

- Accepted F1 checkpoint (preserved, not a resumed G optimizer):
  `520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57`.
- G.1.1 teacher100 warm-start payload:
  `fa798be7d9d6244054b1053bd86c340c4d2f3582290dc72a31e6877c24f81a93`.
- Separately frozen G.0.1 external codec:
  `213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7`.
- Complete common initialized mean tree:
  `94f4b7de78406336b6e6bb87416981330a98e9d74b9bfe7c4280529807184bd2`.
- Frozen G23 V1 protocol/runtime/source binding:
  `5cb5e7363b14239b4c65a7dda1fb70b569e0b92a935e1f2c67311496a15f1d99`.
- Validation-only selection record:
  `ee0df89871b3afa74c5243225270086c29a80b17c229abc9ef73f6b7314dd57c`.

The actual protected-test launch source was
`0652f818956739a2aaf2f04fca10bd3a44cd4d93`, after selection commit
`47cb6b5eeff23f2ff3e9f64658ae03ae566c8d51` and complete selected-validation
commit. `SCIENCE_LAUNCH.json` retains contemporaneous argv/source hashes and
test-opening evidence. Later commits add evidence/docs, not changed numerical
kernels. The binding records full source/protocol, data, codec, update-mask,
Python/JAX/backend/package and PRNG identities.

## Protected-test outputs

Three-seed display means; NLL is overlap-only nats per query payload byte.
The query has three payload bytes. Full-sequence exactness includes length and
all bytes. Authoritative gated aggregates/intervals are in `DECISION.json`.

| Training / inference | Query byte accuracy | Query NLL | Exact query | Full-sequence exact |
| --- | ---: | ---: | ---: | ---: |
| Teacher A / oracle b+ell | 99.4518% | 0.017305 | 98.3553% | 95.2120% |
| Argmax B / generated argmax | 99.2812% | 0.022688 | 97.9167% | 92.1784% |
| Sampled C / generated argmax | 98.9766% | 0.039132 | 96.9664% | 88.5965% |
| Sampled C / actual temp-1 policy | 98.9020% | 0.064837 | 96.8910% | 88.2744% |

| Training seed | A teacher accuracy / NLL | B argmax | C argmax | C sampled |
| --- | ---: | ---: | ---: | ---: |
| 0 | 99.5980% / 0.017360 | 99.3787% / 0.019814 | 99.0497% / 0.029443 | 98.9812% / 0.054061 |
| 1 | 99.3787% / 0.019658 | 99.4883% / 0.013856 | 98.6477% / 0.062686 | 98.6157% / 0.072152 |
| 2 | 99.3787% / 0.014898 | 98.9766% / 0.034393 | 99.2325% / 0.025268 | 99.1091% / 0.068298 |

B and C argmax readouts attain 100% committed boundary, extent, exact-structure,
emitted-length and peer-reach accuracy on this test. The actual C sampled policy
has 99.9661% active-boundary accuracy, 99.3307% exact structure, 99.6756% exact
emitted length, 99.9794% peer reach and 100% valid routing. Its prediction-head
argmax correctness and its sampled commitment are distinct measurements.

All B/C inference modes pass every functionality point/paired-example gate in
**3/3 seeds**, with positive training-seed accuracy and negative query-loss CIs
against their matched cut controls. For B, generated-minus-cut accuracy is
**+95.9186 pp**, example-bootstrap CI **[95.3338, 96.5034] pp** and training-seed
t-CI **[95.6412, 96.1960] pp**. Both nonquery competence guards pass in every
selected model. B nonquery accuracy/loss are 99.8260% / 0.005687; C sampled
99.7170% / 0.017723, well inside the frozen teacher100 baseline guards.

Length-aware reconstruction and conventional overlap-only NLL are not conflated.
C sampled global-byte loss is 0.022177 versus overlap NLL 0.021531, with mean
missing/extra bytes 0.002239 / 0.001325 per example/draw. Generated inference
uses generated emission lengths; truth controls only loss/metrics and named
oracle interventions. See `EVALUATION_FIELD_SEMANTICS.md` for retained base-policy
fields that must not be read as counterfactual total objectives.

## Semantic routing controls and the neutral confound

Primary comparisons use the **same selected B weights and generated extents**.
Only the declared boundary/attention intervention changes.

| Routing within B | Query accuracy | Query NLL |
| --- | ---: | ---: |
| Learned semantic argmax | 99.2812% | 0.022688 |
| Fixed-neutral K8 | 47.5999% | 9.142762 |
| Shifted valid oracle-derived boundaries | 17.3124% | 22.040599 |
| Query-isolating valid cut | 3.3626% | 26.106490 |
| Valid q-anchored overconnected alternative | 46.7958% | 8.520815 |
| No-attention diagnostic | 3.1189% | 7.727212 |
| Direct causal decoder diagnostic | 3.1433% | 29.065299 |

Learned-minus-neutral accuracy: **+51.6813 pp**, paired CI
**[49.1590, 54.2398] pp**; query NLL reduction 99.7518%.
Learned-minus-shifted: **+81.9688 pp**, CI **[80.2629, 83.7597] pp**; NLL reduction
99.8971%. Both primary controls pass the frozen practical, paired-example and
training-seed gates. The cut is therefore not the only adverse comparator.

| Neutral path availability | Support | Learned accuracy / NLL | Neutral accuracy / NLL |
| --- | ---: | ---: | ---: |
| Reaches peer | 789 | 99.3663% / 0.019249 | 54.6684% / 7.460969 |
| Cannot reach peer | 123 | 98.7353% / 0.044745 | 2.2584% / 19.930845 |

The protected discordant subset differs from validation's 132 cases. It is
defined by fixed carrier IDs/K8 reachability, not observed performance. Its
learned-minus-neutral effect is +96.4770 pp, CI [95.1219, 97.8320] pp; n=123
limits subgroup precision. Even reachable neutral routing is not equivalent to
the trained semantic partition: it can expose competing information.

| Query-peer gap | Support | Learned query accuracy | Neutral query accuracy |
| --- | ---: | ---: | ---: |
| 1 | 403 | 99.4210% | 47.8908% |
| 2 | 194 | 99.0836% | 51.2027% |
| 3 | 118 | 98.9642% | 45.7627% |
| 4 | 114 | 99.3177% | 42.3977% |
| 5 | 83 | 99.4645% | 47.5234% |

All 57 length buckets, final-partial/full packets, competitor presence, block
ordinal and boundary-configuration supports/results remain in
`COMPARISON_TABLES.json` and the nine complete model reports. With no outside
competitor (n=21), learned/neutral/shifted/q-anchored routes all score 99.4709%
and NLL 0.011283: there is no universal necessity claim for every example.
Very late block strata have n=2/3 and are descriptive, not new acceptance gates.
The query itself is always full extent; partial-query support is 0/NA.

These are controlled interventions **within semantic-trained weights**, not
separately optimized neutral architectures. Even the overconnected comparison
does not establish superiority over every valid learned routing, scale or task.

## Sampling-training value is unresolved

Primary matched-inference contrast: C **argmax** versus B **argmax**.
Accuracy difference is **-0.3046 pp**, example CI **[-0.5848, -0.0365] pp**, but
training-seed t-CI **[-1.6675, +1.0583] pp**. Seed effects are -0.3289, -0.8406,
and +0.2558 pp. NLL increases by 0.016444 nats (+72.4807% aggregate ratio), with
wide mixed training-seed uncertainty; one seed improves while two worsen.

This is neither a material joint degradation meeting the 1 pp gate nor
equivalence: the seed accuracy CI and NLL intervals do not lie inside the
preregistered practical margins. A small point accuracy change is not grounds
to force NEUTRAL. No sampling superiority, equivalence or general harm claim
is made. Actual sampled-policy readout is separately reported, never substituted
for the primary contrast after seeing its result.

## Gradients, numerical health and recovery

First full-model raw norms A/B/C: 75.2228 / 75.2228 / 102.5066. The connected versus
detached structural-score gradient difference is 57.0708. Prospective updates
were finite, preserved frozen states and were discarded before counted training.
Every counted update has finite saved state and an unchanged-frozen-state receipt.

The accepted G21 independent exact/Monte Carlo adjudication remains unchanged
and was **not rerun**. The original 4.45-SE nonpass remains permanent. The new
prospective complete-42-component confirmation passed its joint finite-action
procedure. This validates the tested score/loss ownership, not an unbiased
clipped Adam update or an isolated scientific advantage for sampling.

Selected-C fixed-batch 16-root mean-gradient norms: 4.8286 / 2.0621 / 3.7466.
Empirical summed variances: 0.002490 / 0 / 0. Identical observed estimates in two
small fixed-batch samples do **not** establish zero true rare-event variance.
These are total trainable-gradient diagnostics, not an isolated score-variance
decomposition. All diagnostic gradients were finite; none was applied.

The original disk stop resumed B seed1 from immutable update137 at update138.
A later stdout BrokenPipe after durable C270 resumed at C271. Complete Adam
state, source/data/key identities and receipt chains were preserved. Both
failures remain archived; no completed update was duplicated or invented.
The 2 GiB guard was unchanged, no project/unrelated files were deleted, and
the whole remaining storage budget was verified before resumption.

The final audit checked 954 selected-validation/test shards, actual IDs, pairing,
finite values, selected checkpoint/optimizer/protocol hashes, and independently
reconstructed every physical capacity cut and carrier pack coordinate. All 45
milestone checkpoint files remain byte-identical to the full loaded-state audit.
All 731 historical baseline entries plus accepted manifests/checkpoints pass
preservation checks. Full source/environment/receipt and recovery records remain
linked in `RECOVERY_AND_TRAINING_COMPLETION.md`, `SCIENCE_LAUNCH.json`,
`EVALUATION_COMPLETENESS_AUDIT.json` and `HISTORICAL_PRESERVATION.json`.

## Validation, integration and next phase

Focused G/full-model checks passed (65 tests), as did formatting, lint, typing
and boundaries. The first monolithic regular-suite attempt was interrupted by
SIGINT after 73 passing tests at approximately 600 seconds; it is **not** recorded
as a complete pass. No assertion failed. No project timeout/installed timeout
plugin was found, and the signal's issuer is unknown. `validation_branch/`
preserves the exact commands, source, logs, times and interrupted return code.

Complete coverage passed sequentially from `validation_sharded/PLAN.json`:
all **323 regular and 30 slow** collected node IDs, in 66 groups of at most eight,
with **zero failures and zero skips**. The separate focused G/full-model run
passed 65 tests. Formatting (187 Python files), lint, typing (zero errors and
warnings), public boundaries and the private-Torx scan (187 files, no violations)
passed. `VALIDATION.json` independently reconciles the complete collected-node
union, every JUnit case, exact requested identities, return codes and log hashes.
Test bodies, assertions, fixtures, markers and scientific thresholds are
unchanged. Only process partitioning and JUnit reporting differ; XML is checked
against every requested node ID, with no hidden skip/deselection. The original
interrupted attempt remains immutable and is not counted as a suite pass.
Accepted Phase-B fixtures were available, so the historical missing-fixture skip
did not occur. `PREINTEGRATION_PRESERVATION.json` rechecks all 731 historical
baseline files, accepted manifests, immutable recovery backups and all 45 G
milestone checkpoint byte hashes, plus frozen source/protocol hashes.

Branch validation is complete. Conditional main integration and its fresh
validation are recorded separately in the delivery record; no merge/push is
implied by the scientific decision or the branch test results.

Phase H requires the real structural-only commit/repacking/key/UNKNOWN/shared-
weight causal/global R/S integration gates in `PHASE_H_HANDOFF_PREREQUISITES.md`.
This result does not authorize H experiments, I codec/LM training or S4, and does
not silently validate H mocks. H/I/Thermo branches were not modified.
