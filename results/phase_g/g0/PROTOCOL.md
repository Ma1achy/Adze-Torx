# Phase G.0 codec and capacity remediation V1

This is a new codec milestone, not a correction to accepted F1/F3 results or
the original failed G compatibility evidence. No G policy arms run here.

## Prospective representation audit

The exact accepted deterministic codec (SHA-256
`55f057f78f795f1585d7aac28fa7ff37846d4c1098507ede375e86b21c4f8ff6`)
provides the frozen F1 target representation. Diagnose all lengths 8–64 using
64 training examples and 16 held-out examples per length, seeds 960/961.
Per-carrier ridge one-hot probes see only h0 plus fixed carrier/slot identity;
they never see target mask, true length, or teacher boundaries as features.
Separate heads recover slot bytes, extent classes, and content boundaries.
A destroyed-h0 control retains positional priors. Report exact emitted length,
partial-carrier recovery, per-length byte accuracy, and baseline decoder NLL.
Failure of a finite probe cannot establish information-theoretic loss. Decoder
only is permitted only when held-out byte/signature compatibility gates and
99% extent/partial-extent recovery pass. Otherwise jointly train the faithful
target-side encoder and Mamba decoder, conservatively addressing both causes.

## Candidate codec training and freeze

Architecture and dimensions are unchanged: C32, h64, Lmax4, original Mamba
frontend/target-analysis/decoder. Candidate parameters are copies of the exact
accepted codec, never references to mutable accepted state. No DiT, proposal,
context, rho, F1/F3 moments or other model leaves enter candidate optimization.
Joint loss: byte CE + content-boundary CE + extent CE, weights 1 each. Fresh
AdamW, lr 3e-4, wd .01, clip 1, batch32. Corpus: 256 examples per length,
seed963; per-update JAX sampling seed964. Fixed validation seed961 and untouched
final test seed962, 32 examples per length. Text domain remains bytes 1..32.
Maximum 5000 updates; validate at 0/100/250/500/1000/2000/3000/4000/5000.
Select the first scheduled validation checkpoint satisfying >=99% valid-byte
aggregate accuracy, >=98% in every length, >=99% task signature, zero numerical
or capacity failures. Evaluate test once; freeze/hash only if it also passes.
Failure or budget exhaustion is not permission to relax thresholds or tune.
Noisy denoising is not codec reconstruction and is not trained in G.0.

## Dynamic capacity policy

`PHASE_G0_SEMANTIC_PUSHFORWARD_DYNAMIC_CAPACITY_V1` samples/argmaxes every
nonterminal semantic edge and every extent. The terminal is deterministic and
excluded from PRNG/log probability. A separate deterministic left-to-right
scan adds a routing-only capacity cut after K sites since the most recent
semantic or capacity cut. It does not overwrite semantic boundaries or labels.
No unconditional fixed-position cuts, resampling, validity conditioning, or
straight-through estimator is used. Every sampled semantic edge retains its
selected-action log probability exactly once, even if routing hides its effect.
Capacity cuts have conditional probability one and zero additional score term.
The routed distribution is the pushforward: sum probabilities of all semantic
actions mapping to each routed topology. The exact oracle independently sums
all 108 actions using a Python partition oracle, compares the production score
gradient including direct continuous paths, and checks many-to-one mass.
M overflow still fails closed. Arbitrary malformed semantic state is rejected,
not repaired by capacity cuts. Capacity edges are not semantic predictions.

## Future G controls

Teacher, full-site argmax and sampled arms must use the exact SAME new frozen
codec hash, same complete initial model means/hash and fresh Adam, and the
approved G data seeds/targets. F1/F3 stay historical controls, not matched codec
arms. A new versioned G runner/manifest must bind this policy and new codec;
the old runner must not be silently reused. G.0 does not authorize those arms,
a full G campaign, H experiments, full-byte or natural-language training.
