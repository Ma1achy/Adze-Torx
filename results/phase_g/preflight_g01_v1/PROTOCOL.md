# Amended G engineering preflight, bound to G.0.1

Prospective implementation/execution record. This does not rewrite any prior
F, G.0 or G.0.1 result. No full G campaign, H/I training or S4 is authorized.

## Initialization, data and graph

Frozen codec SHA256:
`213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7`.
F1 parent SHA256:
`520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57`.
All arms start independently from the same G.0.1-overlaid F1 means, complete
tree hash `84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb`.
No G.0.1 or F1 optimizer state is reused: fresh zero Adam m/v and step0.
Existing noncodec means/rho are preserved in initialization. Target encoder
and rho remain frozen; the model output decoder is trainable as in the
approved F/G mask. The separately frozen codec artifact never changes.

Preserve the approved data generator:1024 training examples seed950,256
validation seed951, bytes1..32, lengths8..64, global IDs0..1023/1024..1279.
Stochastic seed952 with the existing nu/operator/diffusion/structure namespaces.
nu uniform(.025,.25). Training ν and initial epsilon vary by update; validation
uses fixed ν/root/occurrence0 and retained global IDs at every checkpoint.

L4/Q3/S1/R0. Each arm performs the same two applications: predictor, explicit
commit, then committed replay of the IDENTICAL corrupted carrier. Both use
condition0/actualS0 and the same nonstructure randomness. No S2 re-corruption.
Observed structure and target placeholders are fixed neutral objects; clean
teacher h/bytes/ell stay outside the inference graph for corruption/loss only.

Full-site stopped argmax or temperature1 sampled SEMANTIC b/ell commitment;
terminal is forced, never keyed/scored. Dynamic capacity cuts use the already
validated `PHASE_G0_SEMANTIC_PUSHFORWARD_DYNAMIC_CAPACITY_V1`. They remain
separate from semantic labels, add no score term and do not mask sampled edges.
All valid semantic actions retain their selected log probabilities once.

## Loss and optimization (unchanged)

Common: boundary CE + extent CE + .5 predictor h-MSE.
Descendant: .5 replay h-MSE + globally aligned length-aware byte loss.
Teacher/argmax: common+descendant. Sample: common+once-only local-score corrected
descendant; ordinary continuous derivatives stay pathwise. Proposal weight0.
The balanced/noisy G.0.1 codec loss is NOT used for these G arms.
AdamW lr3e-4/wd.01/clip1/batch32. No tuning, no 10k extension or S4.

## Execution and gates

1. Bind immutable source/codec/tree/dataset identities; check fresh moments.
2. Repeat production exact/MC conformance at the frozen16x4096/seed975/four-SE
   rule; focused eager/JIT/identity/reactivation/fail-closed tests.
3. Actual B32 full-model first gradient, same-key connected/detached sampled
   score gradients, finite prospective AdamW update, bitwise teacher/rho and
   moment preservation. Trial update is discarded, not Arm A update1.
   Continuous teacher loss eager/JIT parity uses rtol1e-5/atol1e-6; full-model
   deterministic/public-Torx prediction/logit parity requires max error<=1e-6.
   Sanitized placeholder mutation requires bitwise identical predictions/logits.
4. Real structural interventions with identical inputs/keys: carrier-identity
   attention reachability, packing/repacking, unpool/residual, all12 attention/
   SwiGLU occurrences, emission and byte outputs. Codec overlay/Torx parity.
5. Teacher Arm A100 updates, checkpoints0/10/25/50/100. Do NOT repeat Arm A.
6. Frozen post-teacher necessity gate on all256 paired validation examples:
   correct separately versus shifted cuts, permuted extents, joint changes,
   and fixed neutral. Each control needs >=1pp accuracy advantage AND >=2%
   mean per-example relative length-aware loss reduction. Deterministic2000
   paired bootstrap percentile95% intervals must have positive lower endpoints
   for BOTH differences; accuracy seeds9600..9603/loss9700..9703. No pooling
   controls, filtering examples, or threshold changes after seeing results.
7. Only if all gates pass, independently initialized argmax Arm B100 then
   sampled Arm C100, same checkpoints/data/compute/codec/optimizer.

Gradient-variance diagnostic: fixed32-example batch and fixed corruption/operator
keys, four independent structure roots (base1100) at initialization and update100.
Teacher/argmax have exact zero stochastic-structure variance by definition;
sampled gradient variance is measured. These are engineering diagnostics only,
not a new acceptance criterion. Training gradient magnitudes are retained at
every durable checkpoint. No stochastic failure is silently discarded.

## Persistence, validation and stop

New results only under this directory, source-keyed subfolders; ignored states
under `results/runs/phase_g/preflight_g01_v1/`. Exact protocol matching is
required for every saved result/resume. Atomic checkpoint/receipt writes;
invalid attempts retain failure evidence but never a completed update or
changed durable optimizer state. Preserve historical hashes before/after.

After the experiment (or dependent-stage block), run full regular/slow suites,
format/lint/type/boundaries/private-Torx checks. Copy only exact hash-verified
historical fixture binaries if available, never synthesize accepted fixtures.
Record skips/failures. Issue separate engineering/integration decisions.
Merge normally to main/push only if EVERY G gate passes and final integration
validation is clean. Never merge H/I. Stop without full G scientific training.
