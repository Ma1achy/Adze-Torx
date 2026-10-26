# Phase G.1 — Structural necessity diagnosis

## Diagnostic decision

`G1_TASK_BYPASS_CONFIRMED`

The original iid-byte, target-latent-conditional task admits strong reconstruction
paths without semantic-boundary-sensitive attention. The structural intervention
is real, and the faithful implementation is not shown defective. It is the
original **necessity claim** that is unsupported. This is a diagnostic label,
not a Phase-G scientific result or permission to train.
This conclusion concerns semantic b necessity, not irrelevance of ell.
Extents remain essential emission/participation controls; G.1 holds them fixed.

No optimizer, gradient, additional teacher update, argmax/sampled training,
codec retraining, S4, H/I experiment or merge was executed. The failed G gate
and all historical thresholds remain unchanged.

## 1. Identity, scope and replication

Parent branch: `235ff5ba2be2579c848c988f6910513fd706123e`.
Teacher Arm A checkpoint:
`results/runs/phase_g/preflight_g01_v1/teacher/update_100.pkl`,
SHA256 `203926286c4f991b316e9f9d8cb8bee529861da5124f374279b82d143decd220`.
The entire checkpoint file (including Adam state) is unchanged.

Frozen G.0.1 codec SHA256:
`213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7`.
Diagnostic protocol: [PROTOCOL.md](PROTOCOL.md), written before execution.
Runner SHA256: `1280cd325529cc92b5b9ae7bff28a1be78d67aec3ae6797ceaa4a543420f27db`.
Supplement analysis SHA256: `8af9927d6ca0a6489b3ebe6aa0c9208f64be396f1210448cbd9a8c53ffa57e18`.
New source is confined to diagnostic scripts/tests; production `src/` is unchanged.

All original256 validation examples, IDs1024..1279, seed951, fixed ν and
diffusion/operator roots/occurrence0 from the frozen preflight were retained.
Both structures use identical carrier inputs, means, keys and oracle extents.
The deterministic backend is exactly the zero-noise Torx mean backend on the
checked first8 examples:maximum byte-logit error0.
Maximum per-example accuracy/NLL replication error versus saved evidence is
4.57e-6 (correct) /2.71e-6 (shifted), below the prospectively recorded1e-4
diagnostic tolerance. This is numerical replication, not new selection.

The original equal-example means remain:
correct95.20513% vs shifted95.21647% byte accuracy; NLL0.1412898 vs0.1382994.
Full-sequence exact accuracy is39.84375% vs39.0625% (102 vs100 sequences).
This two-sequence difference does not reverse the frozen failed byte/loss gate.

Local tables below are **valid-byte-weighted**, not equal-example means.
All these controls use the same extents, have no missing/extra bytes and
therefore conventional overlap byte NLL equals length-aware loss. Oracle
extents here are an explicitly declared diagnostic control, not a new claim
of generated-length inference or prompt-only generation.

## 2. What the original task actually contains

Source:`scripts/run_phase_g.py:277–313`,
`src/adze_t/phase_g.py:69–91`.

Targets are iid bytes1..32, lengths8..64. The semantic boundary label is a cut
after a carrier whose first byte is1..16, plus the inherited K-spaced task-label
cuts and final sentinel. There is no independent shared block payload, deliberately
erased query, or designated clean informative peer. ν is positive on every
example, and Gaussian corruption applies to every carrier. There is **no
original uncorrupted-carrier stratum**; its support is zero, not a fabricated
control or zero quality score.

Of2297 emitting sites,2190 have the same binary first-byte signature somewhere
in another block. This binary signature is a coarse boundary-class bit, not a
shared byte payload:there are **zero** full four-byte payload repeats at another
full-extent carrier in these examples. Other iid carrier payloads do not supply
the missing bytes of a noisy query. Their labels are not an intended clean-peer
code. Predictability of b from local content is not necessity of b for decoding
that already-observed local content.

The first8/16/24/32 scaffold positions and prefix-active extent pattern introduce
structural/positional priors. These are features of the accepted task, not
newly imposed fixes. Reusing them to predict labels can improve auxiliary CE
without making the labels useful for byte reconstruction.

## 3. Verify actual intervention and communication

The production mask is block-causal:
`src/adze_t/masking.py:7–23`. A query can read **all earlier blocks**, and
its own block bidirectionally. It is not block-isolated. Altering a boundary
cannot remove an earlier active key under these draft semantics. Treating
earlier peers as inaccessible after a cut would be an incorrect diagnosis,
not a defect in this faithful mask.

| Measurement | Result |
| --- | ---: |
|Examples with changed pack maps|250/256|
|Examples with changed attention reachability|248/256|
|Changed semantic boundary bits|1186|
|Changed mechanical boundary bits|1186|
|Correct / shifted reachable carrier-ID edges|63826 /63999|
|Removed / gained edges|693 /866|
|Correct / shifted future-key edges|1464 /1637|
|Emitting queries losing at least one future path|447/2297|
|Bytes at those queries|1788/8810 (20.30%)|
|Earlier active paths removed|0|
|Persistent query-active sites per example|32|
|KV-active emitting sites, both interventions|2297 total|
|Pack→unpool identity|exact for both routes|

Thus47.34% of original future-key edges are removed, while866 others are
added. The shifted control is a real **rewiring**, not a pure edge deletion.
Two examples change packing but not carrier-ID reachability; six change neither.
No carriers are dropped, duplicated or reassigned. Non-emitting sites remain
query-active/KV-inactive. All8192 persistent queries and2297 active KV sites
are distinguished from padded M×K slots.

Correct and shifted original structures already fit K8; no additional dynamic
capacity cuts are required. Mechanical cuts remain conceptually separate from
semantic labels; no capacity-policy or sampling distribution was changed.

Identity-aligned numerical differences (mean of per-site RMS, not comparison
of differently packed positions):

| Output | All persistent sites | Emitting sites | Emitting queries losing future paths |
| --- | ---: | ---: | ---: |
|Unpooled DiT output|0.042381|0.090564|0.104958|
|h_hat|0.022520|0.048956|0.056715|
|Byte logits|0.132411|0.472228|0.584392|

Byte logits at non-emitting sites are invariant because their decoder emission
mask is false. Their inactive outputs are **not** counted as trivial correct
bytes. The intervention is numerically substantial on emitted queries, not
just a metadata flag.
Partition changes also change logical block/within-block positional embeddings
(`dit.py:330–355`), even though observed b/ell embeddings remain neutral.
Consequently these numerical differences cannot all be attributed to attention.
The explicitly declared exploratory geometry follow-up below isolates this.

All semantic/mechanical b, pack maps[B256,M32,K8], inverse maps[B256,C32],
query/KV masks and carrier-ID reachability[B256,C32,C32] are saved in the raw
NPZ. `supplement.json:first8_partitions` gives human-readable partitions.
For example, correct example0 starts [0],[1,2],[3,4,5,6,7],[8],[9],[10,11]…
with the same persistent identities recovered after either packing.

**Required-peer caveat:** “lost future peer” in the original analysis is only a
geometric proxy. The original task does not define a peer required to determine
the desired byte. The deliberate peer-erasure fixtures below do.

## 4. Local versus aggregate output effects

| Region | Bytes | Correct accuracy | Shifted accuracy | Correct NLL | Shifted NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
|All emitted bytes|8810|95.199%|95.221%|0.14377|0.14068|
|Queries losing future paths|1788|96.421%|96.644%|0.11348|0.10788|
|Queries not losing future paths|7022|94.887%|94.859%|0.15148|0.14903|
|First carrier in correct block|5235|94.422%|94.499%|0.17146|0.16842|
|Later carrier in correct block|3575|96.336%|96.280%|0.10322|0.10006|

Restricting to geometrically affected queries does **not** reveal a hidden
correct-boundary benefit. Accuracy is0.224pp lower, and NLL is higher there.
Dilution by the other79.70% of bytes is not the explanation for this failure.
Among219 sequences with at least one affected emitting query, exact-sequence
accuracy is35.616% correct vs35.160% shifted. Small byte/sequence interactions
are retained, not promoted into a new success gate.

Low-noise bytes are easy (about98.99% accuracy at ν<.1); highest-noise bytes
are about90.76%. The full stratification is in Appendix A. These overlapping
descriptive strata have no new significance-based selection or scientific
threshold. No uncorrupted or zero-emission byte support is invented.

## 5. Controlled information-path ablations

These inference-only diagnostics use the **same frozen weights**, true extents
and original inputs. They are out-of-distribution interventions, not trained
alternative models or additive estimates of causal importance. No production
topology was replaced.

| Diagnostic | Equal-example byte accuracy, % | Exact sequence, % | NLL |
| --- | ---: | ---: | ---: |
|full|95.205|39.844|0.14129|
|direct_noisy_frozen_codec|92.099|31.250|0.46487|
|direct_clean_frozen_codec|99.351|79.297|0.02213|
|residual_only|4.686|0.000|67.11001|
|delta_only|8.870|0.000|40.04361|
|decoder_carrier_reset|84.870|5.469|0.63499|
|direct_noisy_codec_carrier_reset|83.896|4.688|1.68426|
|no_attention|93.396|25.781|0.23028|
|zero_context|95.215|39.844|0.14084|

### Local representation and the carrier residual

Clean frozen-codec decoding gives99.351% on this original validation set.
Even noisy h, without DiT, gives92.099% through the frozen codec decoder.
Resetting that decoder per carrier still gives83.896%. This is direct empirical
evidence that a large fraction of answers can be recovered without within-block
peers. It is consistent with the task definition:the observation itself is a
lightly corrupted encoding of the target, not a masked missing answer.
This is allowed target-latent conditioning, not evidence that hidden teacher
topology or an extra clean target was injected into the inference path.

The actual model residual is `pre_head = h_input + carrier_delta`
(`src/adze_t/model.py:205–247`). Removing delta alone gives4.686%; removing
the direct residual alone gives8.870%. Therefore **the trained model is not
merely an identity residual decoder**. The two terms are co-adapted.
These failed OOD ablations cannot be added/subtracted to assign unique credit.
The direct input remains an answer-information bypass around any requirement
to retrieve a peer, even though the learned head needs its local transformation.

### Attention versus other heavy-core computation

Suppressing attention outputs at all12 physical occurrences retains93.396%
byte accuracy, versus95.205% full model. SwiGLU, conditioning, learned local
transforms and residual paths still execute. Attention helps by about1.81pp;
the result is **not** “DiT is unused.” Native logical block/within embeddings
remain in that ablation, so it is not a wholly b-independent control.

An exploratory follow-up was specified in [GEOMETRY_FOLLOWUP.md](GEOMETRY_FOLLOWUP.md)
before its model execution. It zeroes only block/within embeddings, preserving
carrier/observed-b/ell codes, RoPE identities and every other weight:

| Geometry embeddings | Attention | Structure | Byte accuracy | Exact sequence | NLL |
| --- | --- | --- | ---: | ---: | ---: |
|zero|retained|correct|94.700%|34.766%|0.157751|
|zero|retained|shifted|94.730%|34.766%|0.157164|
|zero|zero outputs|correct|92.299%|19.922%|0.254791|
|zero|zero outputs|shifted|92.299%|19.922%|0.254791|

Correct/shifted logits in the both-zero control are **bitwise identical**
(maximum difference0), despite different pack maps. Thus actual repacking
preserves identities, and high reconstruction survives without either
attention-based boundary communication or committed geometry codes. With
attention retained but geometry codes removed, correct boundaries still have
no advantage. This isolates the mask question without claiming an additive
decomposition of native attention/geometry effects. Full data are in
[geometry.json](geometry.json); no result replaces the original necessity gate.

### Decoder contribution

The baseline Mamba decoder scans the flattened carrier-slot sequence
(`src/adze_t/decoder.py:40–76`), with no reset at semantic b boundaries.
Resetting the same weights independently at every carrier, preserving original
position embeddings, reduces the full-model accuracy to84.870%.

A controlled .1-scale h perturbation at carrier2 changes carrier3 logits by
RMS0.333922, carrier4 by0.034459 and later active sites progressively less.
Carriers0/1 are exactly invariant. First-carrier full/reset decoder parity is
exact0 error. This pins down the bypass path:
earlier h_hat → causal decoder convolution/selective state → later byte logits,
independent of semantic cuts. Reset accuracy loss is not an exclusively
attributable “decoder information percentage”:state calibration also changes.

### Encoder contribution

The frontend and target branch are causal Mamba stacks before pooling
(`src/adze_t/encoder.py:68–167`); semantic b does not reset these stacks.
Counterfactual payload changes leave all encoder prefix sites exactly unchanged
but change104 later sites **outside the query’s semantic block**, with maximum
per-site h0 RMS0.1720. This confirms cross-boundary representation spread.
It does not show future information leaking backwards:the prefix invariance
is exact. Downstream earlier-to-later attention and decoder paths can exploit
such spread in the original task.

### Context/proposal and position shortcuts

All original prompts are0xF1; context spread across the checked examples is
exactly0. Zeroing the prompt-context part of DiT conditioning barely changes
accuracy (95.215% vs95.205%). Zeroing the complete proposal parameter subtree
with explicit carrier h fixed changes byte logits by exactly0.
The proposal is executed but its h is not the carrier source on this path
(`model.py:202–205`). No example-specific clean-target information was found
in these paths; do not call constant context a target-information leak.

Position and extent priors remain explicit confounds. Site0 is much harder
(88.77% accuracy) than sites1–3 (96.72%). The original max64-byte target only
uses sites0–15; sites16–31 have zero emitted-byte support. Appendix A reports
these supports instead of hiding inactive sites in an accuracy denominator.

## 6. Deliberate positive/negative information-flow fixtures

Sixteen paired teacher-controlled fixtures randomize query positions1..5,
four-byte payloads and competing earlier payloads independently; lengths32..60.
The query and its following clean informative peer share a random payload.
Other background, lengths, boundaries and PRNG keys are fixed between paired
labels. The query h is erased; the peer h is clean. Incorrect structure inserts
a cut immediately after query. A boundary before query prevents earlier
queries from importing future information. All other noisy inputs use ν=.1.

These are **diagnostic instruments**, not a silent modification of the accepted
task or an approved production task. Random teacher boundaries alone are not
claimed inferable by a generated-structure predictor.

| Measurement | Correct structure | Peer-excluding structure |
| --- | ---: | ---: |
|Query can attend informative future peer|16/16|0/16|
|RMS query h_hat change when payload changes|0.0118765|0 exactly|
|RMS query byte-logit change|0.129854|0 exactly|
|Query byte accuracy,128 bytes over both labels|3.125%|3.125%|
|Mean query NLL|7.38084|7.59606|
|Paired label-matching log-score gain|0.049444 nat|~0 (1.79e-7 rounding)|

Correct structure improves query NLL by0.215216 nat on average;14/16 paired
fixtures favor it (SD0.258659). The matching statistic compares own-label CE
to cross-counterfactual-label CE, not just arbitrary logit movement. It supports
weak payload-aligned information use, but byte accuracy remains at the1/32
domain chance rate. NLL is poorly calibrated and worse than uniform1..32.
There is **no useful copying/training success** to claim from this frozen model.
The absolute correct/wrong NLL comparison also changes geometry codes, so the
0.215-nat advantage is not exclusively attributable to attention. The paired
payload-response/matching statistic holds each route's geometry fixed and
demonstrates label information arriving through the positive route.
The matching-gain statistic and encoder-only supplement are exploratory
follow-up analyses of the fixed fixtures, not preregistered acceptance gates.
No fixture, threshold or example was selected or changed using these outcomes.

The full frozen codec reconstructs these structured fixtures at99.768% and
99.545% across the two payload variants; isolated clean-peer payload readout
is93.750% /92.188%. This is a small representability screen, not the required
57-length held-out compatibility gate for a new task.

### Exact information-flow argument and bypass check

The query is first in its block; all earlier observations are identical across
the two independent payloads. Causal target encoding cannot write a future
payload into that prefix, and the query’s own answer-bearing h is erased.
With the extra cut, no earlier/query DiT state can read any later label-bearing
site during any of12 block-causal applications. A causal decoder cannot bring
future peer information backwards. Prompt, positions, lengths and observed
neutral structure are identical. Therefore the negative query output must be
label-invariant; it is exactly invariant in both h_hat and logits.

Without that cut, the only later site in query’s block is the clean peer.
Its payload can flow through carrier_in → attention → unpool/carrier_out →
query h_head → decoder. The observed nonzero query h_hat and matched-score
changes identify this path. Encoder spread into still later blocks does not
defeat the negative control:those sites remain unreachable. No unintended
remaining route was detected in these fixtures.

## 7. Separately versioned task proposal — NOT EXECUTED

Proposed identifier:
`PHASE_G1_FUTURE_PEER_ERASURE_CONDITIONAL_ON_G01_H_V0`.
It is a proposal requiring protocol review, not permission to train or resume
the original arms.

1. Keep the faithful encoder/proposal/DiT/residual/Mamba decoder, L4/Q3/S1/R0
   and frozen G.0.1 codec where a new compatibility screen permits.
2. Generate variable semantic groups of2..6 carriers, independent randomized
   per-group payloads, variable total length8..64, ell0..4 and all final-partial
   classes. Query and informative peer must be full extent; partial tails
   occur elsewhere and retain well-defined global emitted-byte loss.
3. Make semantic boundaries **observable**, rather than assigning random hidden
   labels. One concrete candidate reserves raw byte values1..4 as role/end
   markers in slot0 and uses5..32 for three payload slots. End markers define
   semantic b. This stays byte-level; it adds no tokenizer or learned component.
   Marker/domain allocation, frequency and final-partial handling need explicit
   approval and a balanced audit; these are proposed choices, not approved
   scientific facts.
4. Put each independent group payload in its query-first carrier and clean
   future peer. Erase query h completely. Randomize the public query index,
   independent of payload, and expose that index through existing prompt bytes.
   The erasure selector must depend only on declared query roles/indices, not
   hidden teacher b/ell or true target length. Do not put the answer in prompt,
   emission masks, padding metadata or a positional schedule.
5. Supply independently sampled competing payloads in earlier/outside groups.
   Do not reject payload collisions or select favorable positions. Constant/
   majority, position-only and no-peer controls must be retained.
6. In correct structure, the query can read a later informative peer. In a
   graph-cut negative, keep the preceding boundary and insert a boundary after
   query, making it a singleton. Its entire reachable prefix is independent of
   its new payload. As above, encoder causality, erased residual and decoder
   causality prevent an answer bypass. Groups≤K ensure capacity cuts do not
   accidentally remove the intended positive path.
7. Preserve inference-visible neutral observations and explicit post-prediction
   commitment in any later generated arm. Teacher topology is an explicitly
   labelled oracle control only. Raw end markers make non-erased b locally
   inferable; the public query role fixes its known nonterminal status without
   revealing all topology. Extent/readout identifiability must be checked
   separately; lengths never control generated routing.
8. Predeclare masked-query byte metrics **and** whole-sequence accuracy/loss,
   plus each carrier’s b/ell, emission, invalidity and churn metrics. An otherwise
   easy full sequence must not dilute query errors. The original G thresholds
   and failure remain intact; a new task’s criteria/budget require independent
   preregistration, not retroactive selection to satisfy1pp/2%.
9. Before any training:run full held-out codec compatibility on the proposed
   structured distribution, positive/negative identifiability, graph-cut
   invariance and exact score/oracle conformance if action ownership changes.
   The tiny frozen counterfactual above gives a measurable NLL/path advantage,
   **not** adequate byte advantage. A matched teacher-control learning protocol
   is a later authorization, not something performed in G.1.

This is an information-theoretic necessity argument for future-only query
information, not proof that the current weights have learned that computation.
No task is selected because it happened to clear the old threshold.
The marker-coded candidate itself has not been executed or codec-validated;
the numerical results above belong only to the explicitly named teacher-controlled
diagnostic fixtures. Its complete useful-byte advantage remains unestablished.

## 8. Recommendation, preservation and validation

Do not restart G preflight or train B/C on the original task. Keep its gate
failure as a valid record. Do not redesign Adze to block legitimate Mamba or
residual paths. Instead review a separately versioned, identifiable future-peer
erasure task with a causal graph-cut negative, then preregister its codec,
teacher-control and generated-arm gates before authorizing training.

New diagnostic tests:5 passed in6.93s. Formatting149 Python files, lint,
typing (0 errors/warnings) and public boundaries passed. Explicit private-Torx
scan found no matches. Tests include exact draft future/earlier access,
counterfactual pairing, empty support and real decoder-reset/prefix behavior.
Actual checkpoint diagnostics supply the full-model integration evidence;
mock tests are not represented as trainability proof.

No whole-project regular/slow rerun was performed:production source is unchanged,
and this milestone adds only isolated diagnostics/tests. The prior258 regular
and30 slow passes remain historical evidence, not misreported as a new run.
All25 prior preflight evidence hashes and727 historical accepted artifact hashes
remain unchanged, including teacher100/Adam, F1/F3 and G.0/G.0.1.
No previous gradient oracle was edited or replaced.
New raw arrays:
`results/runs/phase_g/g1/v1/diagnostics.npz`,
SHA256 `a4e1c5dede7b0f67219d06b695491e8c87aec285aa424fe48aab8a5f56c06e6e`.
Summary:[diagnosis.json](diagnosis.json), supplement:[supplement.json](supplement.json).
Source/validation/preservation hashes are listed separately in this directory.

## Appendix A — complete descriptive stratification

Accuracy columns are percent, NLL is nats/valid byte. Supports are bytes; full
site/sequence supports and exact-sequence metrics for every stratum are in
`diagnosis.json:strata`. Strata overlap and were not searched for significance.
Block lengths count persistent sites, including non-emitting sites; ordinals
are zero-based in correct semantic partitions.

| Stratum | Bytes | Correct accuracy,% | Shifted accuracy,% | Correct NLL | Shifted NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
|all|8810|95.199|95.221|0.14377|0.14068|
|lost_later_peer|1788|96.421|96.644|0.11348|0.10788|
|no_lost_later_peer|7022|94.887|94.859|0.15148|0.14903|
|first_in_block|5235|94.422|94.499|0.17146|0.16842|
|later_in_block|3575|96.336|96.280|0.10322|0.10006|
|original_uncorrupted|0|—|—|—|—|
|extent_0|0|—|—|—|—|
|extent_1|59|98.305|98.305|0.10472|0.10750|
|extent_2|132|95.455|95.455|0.14875|0.14863|
|extent_3|207|93.720|94.203|0.19535|0.18677|
|extent_4|8412|95.209|95.221|0.14270|0.13965|
|block_length_1|2739|93.501|93.465|0.19470|0.19205|
|block_length_2|2663|95.794|95.719|0.12572|0.12505|
|block_length_3|1568|95.281|95.663|0.14138|0.13607|
|block_length_4|806|96.030|96.154|0.12267|0.11221|
|block_length_5|379|96.834|96.834|0.07268|0.06802|
|block_length_6|289|97.232|96.540|0.08618|0.08884|
|block_length_7|157|97.452|97.452|0.08316|0.08123|
|block_length_8|209|98.565|98.565|0.05954|0.05918|
|absolute_site_0_1|1024|88.770|88.672|0.35724|0.35644|
|absolute_site_1_4|2930|96.724|96.519|0.09462|0.09428|
|absolute_site_4_8|2803|95.540|95.719|0.13644|0.12881|
|absolute_site_8_16|2053|95.762|95.957|0.11745|0.11549|
|absolute_site_16_32|0|—|—|—|—|
|block_ordinal_0_1|1913|92.473|92.368|0.23205|0.23232|
|block_ordinal_1_4|4350|96.460|96.437|0.10845|0.10415|
|block_ordinal_4_8|2232|95.206|95.296|0.13761|0.13496|
|block_ordinal_8_32|315|94.286|95.238|0.13907|0.12909|
|nu_0.025_0.1|2466|98.986|98.743|0.03179|0.03321|
|nu_0.1_0.175|2806|97.470|97.648|0.07905|0.07730|
|nu_0.175_0.250001|3538|90.757|90.842|0.27315|0.26586|

