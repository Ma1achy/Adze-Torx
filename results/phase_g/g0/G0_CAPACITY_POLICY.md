# Full-site semantics and dynamic capacity routing V1

Policy identity: `PHASE_G0_SEMANTIC_PUSHFORWARD_DYNAMIC_CAPACITY_V1`.

## Three distinct objects

1. Predicted logits define semantic b/ell probabilities.
2. `CapacityRouting.semantic` is full-site committed semantic b/ell. Every
   nonterminal b is independently argmaxed or sampled at temperature one;
   every extent is committed. Terminal b=1 is deterministic, never sampled,
   assigned a stochastic occurrence, or score-counted.
3. `CapacityRouting.capacity_cut` and `.routing` are mechanical subdivision
   metadata. Scan left-to-right, count persistent carrier sites since the last
   routed cut; if the count reaches K and no semantic cut is present, insert
   a capacity cut. Reset the count at either kind of cut. Inactive carriers
   retain identity and count toward K. No semantic b/ell is overwritten.

The routed blocks can therefore subdivide a semantic region. The actual DiT
uses the routed blocks and their causal/global masks: subdivisions CAN change
attention reachability and downstream outputs. They are not claimed to be
semantically neutral or equivalent to an unbounded original block. The policy
is explicitly versioned and visible, not a silent reinterpretation of labels.
No attention/MLP/SSM/Q/S/R component is replaced or removed.

## Distribution and gradient ownership

For semantic action a, p(a) is the product of every nonterminal boundary and
every extent selected-action probability. Let r=f_K(a) be the deterministic
capacity subdivision. Then p(r)=sum_{a:f_K(a)=r} p(a). Forced routing cuts have
conditional probability one given a, zero additional log probability and no
new RNG occurrence. A semantic action whose routing effect is hidden by a
capacity cut is STILL sampled and score-counted once. This is an augmented
latent-action estimator of the same pushforward expectation, with potentially
higher variance than a marginalized estimator; it is not double counting.

The existing local score bridge receives the per-example descendant scalar
and summed selected semantic log probability once, before batch reduction.
Ordinary continuous derivatives remain pathwise. No straight-through or
native-Torx double estimator is introduced. No rejection, conditioning on
validity, clipping or renormalization of samples occurs.

Key context remains the established structure namespace with root, optimizer
occurrence, global example ID, actual S, R, commit occurrence, action kind and
carrier site. Capacity cuts do not consume keys. Terminal excludes its key.

## Validity and ownership

`dynamic_capacity_routing` first validates the original semantic domains and
terminal, then validates the capacity-resolved partition before scattering.
It does not repair a malformed terminal or invalid extents/activity. M overflow
still fails closed; for G C=32/M=32, all valid semantic actions fit by count.
Both semantic and mechanical structure are retained in the typed result.

Historical G behavior stays the default. The new model objective requires
explicit `dynamic_capacity=True`; this chooses full-site commit masks and
passes only validated routed metadata to the existing replay. No old protocol
manifest can claim it used the new distribution. A future G training runner
must version and bind the new codec and this policy before any arm starts.

## Independent evidence

The partition oracle is a separate Python scan checked exhaustively on 128
seven-edge configurations at K=1/2/3/8. Eager/JIT, scan/vmap, inactive-site
query/KV semantics, pack/unpool identity, terminal rejection and M overflow
are tested. A compiled real-model S1 predictor/replay test executes the new
full-site policy through real packing/DiT/unpool/residual/decoder computation.

The 108-action gradient oracle uses an independent Python route map for the
exact expectation, and the production public-Torx categorical commit path for
Monte Carlo. It covers uniform/asymmetric/peaked logits, inactive/reactivation,
zero/missing/extra bytes, direct continuous paths and the production global
length-aware reconstruction loss. All 21 MC cases pass at 16x4096, seed975,
componentwise <=4 root SE, zero-SE <=1e-6. Many-to-one action probability mass
is summed, not discarded. See `capacity_oracle.json` and `tests/test_phase_g0.py`.

This validates estimator correctness for the finite oracle, not full-model
scientific trainability or superiority of any structural policy.
