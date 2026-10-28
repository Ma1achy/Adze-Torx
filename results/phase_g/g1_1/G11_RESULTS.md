# Phase G.1.1 — Future-peer benchmark validation

## Decision

`G11_TASK_VALIDATED`

The new benchmark passes the prospectively frozen information-flow, codec and
fixed-readout identifiability checks. A **separately authorized bounded teacher
experiment is justified**. None was run. This validates the finite benchmark
contract, not neural trainability, convergence, generated-structure superiority
or a Phase-G scientific outcome. The original iid-byte task remains rejected;
its teacher/argmax/sampled arms were not resumed.

## 1. G.1 preservation and immutable execution identities

G.1 report/code/evidence were reviewed against their13-entry manifest and all14
independent archive files. All hashes matched. The727 protected historical
artifact hashes matched, five G.1 regressions and all static checks passed.
G.1 was committed unchanged on the existing G branch at
`404f2e208793a197a0c206c1febfea235a613375`. The old manifest's “uncommitted”
wording is its historical capture status, not a claim about the new commit.

New validator, generator, tests and both prospective protocols were committed
**before model execution** at `f75e75d5d8f1883dc506315798c06f92414f9877`.
The working tree was clean at launch. The actual command was
`PYTHONPATH=src python scripts/run_phase_g11_validation.py`.
`VERDICT.json` binds the execution to that commit and exact source/protocol hashes.
The later routing export/coverage summaries are labelled postprocessing, not
new acceptance gates. No example/threshold was chosen from performance outcomes.

Frozen identities:

- F1 checkpoint SHA256:
  `520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57`.
- G.0.1 codec SHA256:
  `213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7`.
- Old teacher100 checkpoint SHA256:
  `203926286c4f991b316e9f9d8cb8bee529861da5124f374279b82d143decd220`.
- Planned new initialization tree:
  `84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb`.

Both the planned F1/G.0.1 mean tree and the unchanged old teacher100 weights
were tested for continuity. The latter is NOT the proposed training initializer.
No new optimizer was created and no parameter/moment file was written.

## 2. New versioned task and permitted observations

`PHASE_G11_FUTURE_PEER_OBSERVATION_LATENT_V0` retains bytes1..32,
C32/Lmax4/M32/K8, L4/Q3/S1/R0 and the frozen faithful encoder/decoder.

Each emitting carrier starts with an observed raw role byte:END1, CONTINUE2,
QUERY3. Its three payload slots use iid uniform5..32. Blank4 is an observation-
only erasure symbol. These are byte-level synthetic data rules, not tokenization.
Query q is the first carrier in a semantic group of2..6 full carriers; peer j
is its END. Intermediate/other payloads are independent nuisance signatures.
Other semantic ENDs are independently sampled, not fixed packing-capacity cuts.
The final emitting carrier is an END; inactive b=0 except the forced terminal.
Extents use the frozen codec's prefix-full/final-partial/inactive-tail layout,
not arbitrary interior holes. All0..4 classes are represented; unrestricted
interior extent mutation/reactivation is not a scientific claim of this benchmark.

The true query payload equals j's payload. **Before any frozen frontend/target
encoding**, observation query payload is replaced by blank4. Its role and extent
remain visible. This is the important refinement of G.1:erasing the whole h
would unnecessarily erase structural information, while encoding the true query
first would spread its answer into intermediate carriers.

Model input h is encoded from this answer-free-at-query **observation sequence**,
not the clean target. Target h is separate loss/compatibility data. Corruption
is `PHASE_G11_PUBLIC_QUERY_PAYLOAD_ERASURE_V0`, with no added Gaussian noise and
noise conditioning0. It is not a redefinition of Phase-F cos/sin corruption.

Prompt[0xF1,q+1] reveals only the public query index. It does not reveal j,
target length, payload, b or ell. Source sequence termination/padding is part
of the declared observed packet and its frozen encoding; its mask is never
passed directly into the Adze predictor. Actual model target analysis and
observed b/ell are fixed neutral placeholders. Only explicitly labelled
teacher controls receive committed structure. Output length is not a hidden
inference-budget oracle. This remains latent-conditioned packet reconstruction,
not prompt-only language generation or unknown-length language generation.
Observation h is supplied by the benchmark's frozen external conditioning
instrument. The canonical training-only target encoder has not been moved into
the production generation interface.

All role markers/boundaries/extents are independent of the query payload.
Public position/length knowledge cannot reveal an iid payload. Counts-only
training-majority and position-prior baselines achieve3.911%/3.363% on validation,
near the population3.571% chance rate. They are descriptive baselines, not fitted
neural probes or acceptance gates.

## 3. Data integrity and coverage

| Split | Seed | Examples/length | Total | IDs |
| --- | ---: | ---: | ---: | --- |
|Training construction only|1140|32|1824|0..1823|
|Task validation|1141|16|912|10000..10911|
|Protected test construction/hash only|1142|16|912|20000..20911|

All57 lengths8..64 are represented. Exact source overlaps across every pair of
splits are zero. Complete ordered observation/target/prompt/q/j/b/ell tuple
hashes are in `dataset_identity.json`. No protected-test model metrics were
exposed. Symbolic source-only reconstruction and b/ell rules match every ordinary
example exactly; negative controls are intentionally ambiguous interventions,
not contradictory ordinary training examples.

Query positions span0..14; gaps1..5 have425/201/126/87/73 validation examples.
The64 neural-flow rows were selected by fixed permutation seed1143, never by
accuracy, and contain every gap. Exact IDs/q/j are in `routing_trace.json`.

889/912 examples have an emitting competitor outside the query block;816 have
a full outside signature. The23 without an outside emitting competitor are
short/boundary edge controls. At8 bytes only two full carriers fit, so it would
be false to claim every example contains an outside full distractor. Among64
flow examples,62 have an outside emitting competitor. Report these strata in
later experiments; do not treat all lengths as equally hard disambiguation tasks.

A metadata-only fixed-neutral K8 audit finds the peer unreachable in132/912
examples, versus always reachable with correct semantic groups. This supplies
a meaningful cross-capacity subset; it is not a measured neural performance
comparison and was not used to select examples or change the primary gate.

## 4. Information-theoretic controls

Let Y be the three query payload bytes, iid over28 values. The relevant
information set is the query's **causal ancestors in this architecture**,
including the earlier h_hat states read by its causal decoder. It is not the
entire globally stored input array:that array still contains Y in B/E.

| Control | Intervention | Ideal payload-byte accuracy | Ideal per-byte NLL |
| --- | --- | ---: | ---: |
|A|Correct semantic group contains later informative j|1|0|
|B|Valid cut after q; q becomes singleton|1/28|log28|
|C|Erase peer payload before source encoding|1/28|log28|
|D-outside|Randomize all nonquery/nonpeer payloads; preserve role/answer peer|1|0|
|D-substitute|Replace peer payload with independent nuisance before encoding|1/28|log28|
|E|Direct causal decoding of source h, without DiT path|1/28|log28|
|C-post|Zero peer h after encoding; later encoder spillover retained|1/28|log28|

D-outside includes intermediate nuisance carriers as well as outside blocks;
it does not secretly replace the answer peer. Both interpretations of the
requested randomized-distractor control are therefore tested.

The reference entropy is log28=3.33220451 nats/byte; exact three-byte chance is
28^-3=0.0000455539 (0.004555%). The known query role is excluded from these
metrics. Including its trivially known marker would inflate optimal whole-
carrier accuracy in unavailable cases to27.679%.

### Why the negative information sets contain no answer

Y first appears in the observation at j. Source encoder states strictly before
j are identical across independent Y counterfactuals, including q and all
intermediate carriers. Role, q, lengths, boundaries and noise/root context
are held fixed. Query is first in its block, so earlier DiT queries cannot
import the future answer through another same-block site.

B puts q alone before j:neither q nor any earlier query can read a label-bearing
future state in any of12 block-causal applications. C and D-substitute remove
Y from the entire observation. C-post removes j's answer-bearing h while
subsequent label-bearing encoder states remain behind j's semantic END and
cannot communicate backwards. E's decoder can only read source-h prefixes.
The query residual holds a known placeholder representation, not Y. Context
and proposal depend only on the unchanged public prompt. Committed geometry
has no Y side channel because its labels are independent of Y.

A has access to j; D-outside preserves that access. Reading the first END
after q recovers the raw answer exactly. This ideal raw-symbol statement is
separate from a universal injectivity claim about a learned codec. Actual latent
observability is measured below, and finite28-symbol representation slices are
checked explicitly. No claim of perfect inversion of every possible h is made.

## 5. Actual Adze tensor results

Both frozen parameter states pass every flow gate on64 paired examples.

| Actual tensor check | Fresh F1/G.0.1 | Frozen old teacher100 |
| --- | ---: | ---: |
|A query→j attention paths|64/64|64/64|
|B query→j attention paths|0/64|0/64|
|Source encoder prefix error before j|0|0|
|Maximum ordinary A payload-induced query-logit change|2.316856|1.484178|
|B/C/C-post/D-substitute query h/logit label response|0|0|
|No-attention and direct-decoder label response|0|0|
|Context/proposal paired label response|0|0|
|Eager/JIT max byte-logit error, A/B/C|0|0|

These are complete actual Mamba frontend/context/proposal, hard-pack,
L4×Q3 attention/SwiGLU, unpool/residual, h-head and Mamba decoder executions.
The model target placeholders never contain source or target bytes.
No hidden teacher target analysis is used. E was checked with both the model
decoder and frozen codec decoder; neither can obtain j's future answer at q.

### Oracle-information injection, not ordinary input

Start with answer-independent C observation, then inject only the clean true
query latent into the future j site. The two labels share every other tensor.
This deliberately oracle test is forbidden as ordinary benchmark/training input.

| Diagnostic | Fresh F1/G.0.1 | Frozen old teacher100 |
| --- | ---: | ---: |
|A query logit-response RMS|0.103606|0.075090|
|A query h-response RMS|0.0052013|0.0059432|
|Minimum response RMS among8 injected examples|0.042487|0.040441|
|B, attention-zero A, direct decoder response|0 exactly|0 exactly|

Thus the real network path can carry the injected signal. Blocking only the
permitted communication makes it vanish; a toy attention surrogate or assumed
mask is not being used as proof. This establishes capacity of the path, not
successful decoding or stable learning from gradients.

### Chance controls and untrained accuracy

On two declared finite slices (q3→j4 and q1→j6), all28 values of one payload
coordinate produce distinct actual peer latents. B/C/C-post/D-substitute/E
query logits remain bitwise identical across those labels. A valid-domain
argmax therefore matches exactly1/28 in every negative slice. Raw256-way
argmax scores0 or1/28 if it selects a role/blank/out-of-domain value. Projected
scores are a diagnostic reference only; model performance is never renormalized.

The actual unadapted model does NOT solve the copy task:ordinary A query accuracy
is1.042% at the fresh means and0% at old teacher100, with NLL35.773/29.984.
This is expected out-of-distribution behavior, not a requirement violation:
these weights were not trained for payload erasure/peer retrieval. Do not call
the information-flow or readout results neural training success. The planned
bounded teacher experiment is precisely what would test learning.

## 6. Real packing, identity and capacity

All64 A and B traces preserve pack→unpool identity for all32 persistent sites.
Each control has2048 active query sites and567 active KV/emitting sites;
non-emitting carriers retain identity and queries but are KV-inactive.
Both contain150 dynamic mechanical cuts, without altering semantic labels.
The decoder-prefix ancestor set can reach j in64 A cases and0 B cases;
neither can reach label-bearing states after j. This corroborates the C-post
test rather than ignoring encoder cross-boundary spillover.

Full carrier-ID reachability, pack/inverse maps, q/KV masks, semantic and
mechanical b and capacity-cut masks are archived in
`results/runs/phase_g/g1_1/validation_v0/routing_trace.npz`.
Trace export recomputes deterministic metadata and verifies its peer-path bits
against the bits recorded during BOTH actual model executions.

Capacity stress with one32-site semantic run yields mechanical cuts7/15/23,
preserving semantic terminal-only b. q0→j9 is correctly blocked by K8;
q8→j9 is allowed. Eager/JIT routing matches, block-count overflow M1 fails closed,
and an invalid actual-model test skips execution/returns zeros. The benchmark
does not silently count a >K positive:query group≤6 and its preceding boundary
resets the mechanical run. Deterministic/public-Torx parity error is0.

## 7. Separately established identifiability

These are912 validation examples, not the protected test or a fitted probe.
All57 target lengths and all partial classes are represented.

| Check | Result | Prospective criterion |
| --- | ---: | ---: |
|Clean target codec byte accuracy, equal-example|99.6202%|>=99%|
|Clean target codec byte accuracy, byte-weighted|99.5949%|reported|
|Worst length bucket (29 bytes)|98.7069%|>=98%|
|Clean target exact sequence|86.7325%|reported|
|Clean target byte NLL|0.0143032|reported|
|Role/signature recovery|99.3446%|>=99%|
|Frozen readout per-site extent|100%|>=99.5%|
|Exact total length / extent vector|100% /100%|total>=99%|
|Partial extent / zero-length recall|100% /100%|partial>=99%|
|Active boundary accuracy|99.7074%|>=99%|
|Exact boundary vector|97.2588%|>=95%|
|Composed observation-reader query payload|99.8173%|>=98%|
|Numerical/route failures|0|0|

Extent supports0/1/2/3/4 are20640/224/224/224/7872 sites; each class recall is100%.
The independent frozen target_l head receives ONLY observation h. The frozen
decoder then uses its PREDICTED extents, classifies END roles and selects the
first decoded END after the public q. It is never supplied teacher j or lengths.
END is found in100% of cases. This composed reader demonstrates information
available to a later routing mechanism; it is a noncausal diagnostic oracle,
not a replacement for the faithful causal Mamba decoder.

Pure clean codec reconstruction above uses oracle lengths as a separately
labelled compatibility test. The length/boundary/query identifiability readout
does not. Boundary prediction is therefore not made impossible by erasing its
only marker, and length recovery is not being asserted from a teacher mask.
Exact b vectors are not already perfect:generated heads still require learning,
and passing this benchmark does not prevalidate an argmax/sampled system.
The prospectively declared95% exact-vector cutoff is a NEW diagnostic
sufficiency criterion for an unfitted decoder-based END extractor, not a
lowered G.0.1 trained-boundary-head acceptance criterion. The label semantics
and readout differ. Raw markers determine b exactly; the fixed reader supplies
an empirical lower bound on latent observability, not proof of error-free
recovery for every possible context. No old codec/training threshold is rewritten.
The current query CE is defined for teacher extents (query is full extent).
Any later generated-length query objective must separately define missing/extra
query-byte scoring and score ownership before a generated arm is authorized.

## 8. Frozen scientific protocol and execution recommendation

[BOUNDED_TEACHER_PROTOCOL.md](BOUNDED_TEACHER_PROTOCOL.md) was frozen in the
pre-execution commit, not tuned after these measurements. It specifies:

- Fixed new task/corpora/roots, frozen G.0.1 codec, exact F1-overlaid initialization,
  fresh Adam0, rho/target teacher frozen; no old teacher100/F3 optimizer state.
- L4/Q3/S1/R0, neutral predictor→teacher commit→same-input replay, condition0
  in both applications; no S2 re-corruption or R refinement.
- New explicitly versioned loss:predictor b/ell CE1; replay h-MSE.5, global
  length-aware byte loss1 and query3-byte CE1; predictor h-MSE0. The latter avoids
  asking neutral routing to predict an answer before it has peer access. This
  is a declared new-task amendment, not a rewrite of old G/F losses.
- AdamW lr3e-4/wd.01/clip1/batch32;100 NEW updates only; checkpoints0/10/25/50/100.
- Query-specific and whole-packet metrics, teacher-cut/erased/distractor/direct-
  decoder controls, frozen endpoint100 rather than test-based selection.
- Primary bounded gate:A−B query accuracy>=10pp AND>=10% relative NLL reduction,
  paired uncertainty supporting direction, plus nonquery competence guard.
  Original G's1pp/2% gate remains unchanged for its original failed task.
- 2000 paired whole-example bootstrap replicates stratified by57 lengths,
  seeds1161/1162,95% percentile intervals. No resampling individual correlated
  payload bytes or tuning settings to obtain significance.
- Finite first-gradient/discarded update gates only AFTER future execution
  authorization, exact provenance/resume state, stop on numerical/structure/
  leakage/persistence failure, no extension or policy-arm training.

**A bounded teacher experiment is justified, not authorized or executed here.**
It must be a new matched run; never resume the rejected original arms or use
old teacher100 as initializer. Query-gap, competitor-presence and cross-K
strata should remain visible in its report, not disappear into an average.
The benchmark shows a controlled graph cut matters; it does not prove the true
partition uniquely optimal among all overconnected alternatives. Full G science
and generated commitment comparisons remain later, explicitly authorized work.
The chance claims depend on draft S1/R0 routing; they must not be carried into
global refinement or adaptive multi-commit trajectories without a fresh audit.

## 9. Validation, preservation and limitations

New G.1.1 regressions cover raw grammar, before-encoder erasure, metadata
independence, C/D-source equality under counterlabels, both routing identities,
actual-model future transmission/cut/no-attention controls and eager/JIT invalid
capacity. An additional independent host parser checked all3648 ordinary rows
and its mutation test rejects a corrupted private peer identity; it does not
reuse the generator's symbolic-target helper. Together with G.1 regressions:
11 passed in14.79s, zero skips.
Formatting154 files, lint, type checking and public boundaries pass. Explicit
private-Torx scan finds no imports. Existing architecture/model/encoder/decoder/
score-bridge files were not edited; only a new benchmark data module and isolated
validators/tests/docs were added. No action distribution/estimator changed, so
the historical exact/MC score oracles remain unchanged.

Whole-project regular/slow suites were not rerun for these isolated additions;
the prior258+30 result is historical, not claimed as fresh validation. Native
full-model inference and focused tests supply this milestone's runtime evidence.

All727 protected artifacts and G.1 archived evidence are rechecked at delivery.
No F1/F3/G.0/G.0.1 state, old teacher100/Adam, threshold or historical evidence
was changed. Main/H/I remain unmodified; no merge or push. Protected-test neural
metrics are unexposed. No neural training, optimizer update, H/I experiment,
full G campaign or S4 was performed. Counts-only prior baselines are not neural
fitting. Stop at this task verdict and the conditional frozen teacher protocol.
