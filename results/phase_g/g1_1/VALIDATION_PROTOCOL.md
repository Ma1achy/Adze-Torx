# G.1.1 task validation — prospective, inference only

Task version: `PHASE_G11_FUTURE_PEER_OBSERVATION_LATENT_V0`.
G.1 evidence commit:404f2e208793a197a0c206c1febfea235a613375.
No neural fitting, optimizer update, argmax/sampled training, merge or push.

## Representation and grammar

Byte domain1..32, C32/Lmax4, target lengths8..64. Slot0 is a raw role byte:
END1, CONTINUE2, QUERY3. Payload slots use iid uniform5..32 (28 values).
Blank4 is an observation-only erasure symbol. These are raw-byte data rules,
not a tokenizer or architecture change.

Choose q uniformly among full carriers with a later full carrier. Choose span
uniformly2..min(6,full_count-q); j=q+span-1. Query is first in its semantic
block; j is its END. Outside END markers are independent Bernoulli(.5), with
last emitting carrier an END. Other groups may be singleton/long; no semantic
cut is added merely to fit K8. Inactive b=0 except fixed terminal b31=1.
Dynamic capacity cuts remain separate. Query block<=6 always fits K8 because
the preceding semantic boundary starts its capacity run.

True target query payload copies the independently sampled payload at j.
The **observation sequence**, not the target, has query payload replaced by
blank4 BEFORE frozen encoding. Query marker, observed extent and prefix remain
visible. This refines G.1's whole-h erasure:answer information is erased without
unnecessarily erasing structural information. No answer-bearing target latent
is used for ordinary model input. Later encoder spread can occur only at/after j.

Prompt bytes[0xF1,q+1] expose only the public query index, not j, length, b/ell,
payload or target masks. Model input is clean frozen G.0.1 **observation** h,
with no added Gaussian noise:the corruption is public query-payload erasure,
`PHASE_G11_PUBLIC_QUERY_PAYLOAD_ERASURE_V0`, not Phase-F cos/sin corruption.
Target h is separately computed for compatibility/later loss only. Forward
target analysis and observations are fixed neutral placeholders; b/ell enter
only as explicitly labelled committed teacher controls. No ground-truth mask
controls a generated predictor or an identifiability readout.

## Data and initialization

Balanced lengths:32 examples/length training(seed1140,1824 examples),16/length
validation(seed1141,912),16/length protected test(seed1142,912). Global IDs are
0..1823 /10000..10911 /20000..20911. Hash the complete observation/target/query/
peer/b/ell/prompt tuples and reject cross-split duplicate observation sequences.
Do not evaluate protected-test model metrics now.

Flow controls use64 validation rows chosen by seeded permutation1143, independent
of performance. Alternate query labels use seed1144; nuisance distractors1145.
All payload collisions are retained. Model/root keys fixed and paired, lambda0.

Verify actual Adze tensors at both frozen teacher100 and the planned fresh
G.0.1-overlaid F1 means(tree84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb).
F1 SHA520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57;
G.0.1 SHA213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7.
No model/moment/teacher/rho mutation. L4/Q3/S1/R0, native condition0, actualS0.

## A–E and additional causal controls

A correct teacher structure; B add a valid cut after q (q singleton).
C replace peer payload by blank4 BEFORE observation encoding.
D-outside randomize independent nonquery/nonpeer payloads, preserving roles;
D-substitute replace peer payload with an independent nuisance signature before
encoding. This covers both meanings of 'randomized distractor'.
E decode input observation h causally with no DiT path (both model/frozen codec
decoder). C-post additionally zeros peer h AFTER encoding to stress that later
label-bearing encoder states cannot bypass the block cut. All extents and
structural labels are held fixed across paired payload counterfactuals.

At the query's causal-ancestor information set, ideal payload accuracy is1 for
A and D-outside, and1/28 for B,C,D-substitute,E,C-post. Ideal payload NLL0 versus
log(28), exact3-byte payload accuracy1 versus28^-3. This is not a claim that an
untrained model is Bayes optimal. Raw full-vocabulary argmax can score BELOW
chance if outside5..32. Full global inputs still contain the answer in B/E;
their chance statement is graph-constrained at q, not for an unconstrained reader.

Oracle-information injection:encode answer-independent C observation; put the
clean true query latent into j only, explicitly a diagnostic NOT benchmark input.
Compare paired label signals under A/B, attention-zero A, and direct decoder E.
Positive path must transmit a nonzero signal; negatives must be label invariant.
No requirement for existing weights to decode injected information accurately.

Enumerate all28 values of one payload coordinate on two fixed validation rows
(first row with gap1 and first with maximal supported gap). Verify distinct
peer representations and query-output invariance in unavailable controls. Score
only that varied coordinate in this finite conditional slice.

## Predeclared validation checks

- Exact symbolic reconstruction, boundary and extent labels from declared raw
  observation/public role; no contradictory observations/labels or dataset leakage.
- Source encoder prefix before j invariant across counterfactual payloads.
- Actual query↔peer reachability A=true/B=false; exact pack/unpool identity.
- Negative query logits/h invariant within1e-6 for B,C,C-post,D-substitute,E.
- Oracle injection transmits query h/logit signal>1e-6 in all first8 rows;
  blocked/no-attention/direct-decoder response<=1e-6.
- Eager/JIT max logit error<=1e-5, no invalid routes/numerical failures.
- Capacity stress:semantic run>K forces cuts without modifying semantic labels;
  a forced cut may legitimately remove a future path. No invalid positive task
  fixture may be silently admitted. Both overflow/terminal failures fail closed.
- Frozen target codec clean byte accuracy>=99%, each length>=98%, role/end
  signature recovery>=99%, zero numerical/capacity failures. These reuse the
  prior compatibility thresholds on a NEW distribution, not the old results.
- Independent fixed-weight representation readout, with PREDICTED extents:
  per-site extent>=99.5%, exact total length>=99%, partial extent>=99%, active
  boundary accuracy>=99%, exact boundary vector>=95%. These are operational
  sufficiency checks, NOT proofs that failure means irreversible information loss.
  Decoder/target_l weights stay frozen; no probes are fitted. Query metadata are
  read from its un-erased role-bearing observation h, not overridden by truth.
- Composed fixed-weight observation reader (predicted extents, decoded END
  selection, decoded peer payload; never teacher j) must recover>=98% of query
  payload bytes. This is an operational observability check, not model success.
  Codec aggregates are equal-example means on balanced lengths; report micro
  support as well. Per-length accuracy is over that fixed-length bucket.

Raw-information/graph contract failure => G11_TASK_INVALID. If logical checks
pass but codec/readout/injection evidence is insufficient =>G11_TASK_UNRESOLVED.
All checks pass =>G11_TASK_VALIDATED. Chance-level untrained query accuracy is
not itself a failure. Preserve every result; no task tuning/retry to get a pass.

If structural flow checks pass, write the bounded teacher scientific protocol
as a separately frozen document, with remaining execution blockers explicit.
Do not execute it. No original G arms, H/I training or S4. Historical evidence,
oracles and thresholds are immutable. Stop after task verdict and protocol.
