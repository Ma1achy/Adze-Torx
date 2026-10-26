# G.1 inference-only structural-necessity diagnosis

Parent source235ff5ba2be2579c848c988f6910513fd706123e. No optimizer,
training, generated-arm restart, production architecture change or merge.
Frozen teacher100 checkpoint SHA256:
203926286c4f991b316e9f9d8cb8bee529861da5124f374279b82d143decd220.
Frozen G.0.1 codec SHA256:
213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7.

## Diagnostic scope fixed before execution

Use all original256 validation examples (seed951, IDs1024..1279), fixed
corruption occurrence0/seed952 and nu uniform(.025,.25), unchanged correct
and shifted-cut controls. Verify replication against the saved paired results.
Deterministic backend uses exact frozen Torx means; check zero-noise Torx
parity on the first8 examples. Batch8, no model mutation or gradients.

Save committed semantic/mechanical boundaries, pack/unpool identity maps,
query/KV masks and attention reachability in carrier-ID coordinates. Compare
identity-aligned unpooled DiT outputs, h_hat and byte logits. 'Lost peers' in
the original task means only a geometric proxy:active later sites reachable
under correct but not shifted routing. The task specifies no required peer,
deliberate erasure or uncorrupted carrier. Do not invent those strata.

Local metrics are valid-byte-weighted accuracy/NLL with site/byte support,
and paired changes. Strata:lost versus retained future-peer paths, block
length/ordinal, first/later in block, absolute site bands, extent0..4, nu
bins[.025,.10,.175,.25]. Empty support is null. Sequence accuracy and
equal-example aggregate metrics are reported separately.

Named diagnostic ablations only:

- Zero attention outputs at all12 occurrences; preserve every other operation.
- Zero prompt-context portion of DiT conditioning (constant prompt).
- Direct decode of frozen noisy/clean h with frozen codec decoder.
- Decode h_head(current_input) (carrier-delta removed) and
  h_head(pre_head-current_input) (direct residual removed), retaining head bias.
- Reset the same decoder weights independently for each carrier, retaining
  original carrier/slot embeddings; compare full recurrent decoder.
- Zero proposal parameters on8 examples while keeping explicit h fixed;
  verify outputs unchanged, since proposal is not the explicit carrier source.
- Perturb one earlier decoded h and measure later-versus-earlier logit effects.

These are out-of-distribution causal/path diagnostics, not replacement
architectures, trained controls or uniquely additive causal contributions.

## Counterfactual information-flow fixtures

16 paired fixtures (seed1101), variable lengths32..60, random query position
1..5, clean future peer immediately after query, independent four-byte block
payload and competing earlier payload, bytes1..32. Target/peer repeat payload.
Background/length/position/boundaries/noise are held identical across payload
counterfactuals. Query h is zeroed; future peer h is clean; other sites use
nu=.1. Correct groups query+peer, wrong inserts a cut after query, leaving
the query singleton with all earlier keys still reachable. Previous semantic
boundaries prevent earlier queries from importing future label information.

Quantify encoder prefix invariance and later spread, decoder prefix effects,
target-specific output sensitivity and local accuracy/NLL for correct/wrong.
Also evaluate clean codec reconstruction on these fixtures. These fixtures
are teacher-controlled diagnostic instruments, NOT an approved new training
task; their random boundaries alone are not declared inferable. Distinguish
graph reachability, information availability and actual frozen-model decoding.

No new success threshold is substituted for the failed G gate. Report any
unlearned counterfactual behavior honestly. A task proposal must explicitly
address observable boundaries, answer-independent selection, future-only
information, encoder/decoder paths and codec compatibility before training.

Preserve all original evidence, checkpoint/moments, gradient oracles and
thresholds. Hash inputs and outputs, record source/environment and verify
historical hashes before/after. No full suite is required for production
changes because there are none; run focused diagnostic regression/static/
public-API checks. Stop after report/task proposal and one diagnostic label.
