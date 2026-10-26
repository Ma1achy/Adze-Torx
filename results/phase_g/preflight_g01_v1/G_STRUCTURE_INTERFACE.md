# G.0.1-bound generated structure interface

This version amends execution, not historical result records. See
`docs/design/phase_g_generated_structure.md`, `src/adze_t/generated_structure.py`,
`src/adze_t/phase_g0.py` and the prospective [protocol](PROTOCOL.md).

`ObservedStructure`, `PredictedStructure`, `SampledStructure`,
`CommittedStructure`, `StructureCommitResult`, `GeneratedRoutingResult`,
`StructureHysteresisState`, `StructureKeyContext` and
`StructureEstimatorOwner` remain distinct. JAX records carry explicit arrays;
commit/routing cores return whole-batch validity and error codes. Host wrappers
reject malformed inputs. Consumers must branch on validity before execution.

`b[i]=1` is a semantic boundary AFTER carrier i. `ell=0` suppresses emission,
pooling and KV participation but preserves persistent identity and active
queries. A later authorized positive extent can reactivate the same site.
The terminal boundary is forced, never sampled/keyed/scored.

Observed neutral structure controls embeddings. Validated committed structure
controls packing, activity, masks and emission. Predictions cannot affect the
application that produced them. Full-site edge/site commit masks include all
nonterminal semantic edges and every extent. Partial ownership/hysteresis
mechanics are tested but not exercised by this smoke protocol.

`dynamic_capacity_routing` deterministically derives mechanical K-capacity
cuts from committed semantic boundaries without overwriting semantic labels.
Capacity cuts own no random occurrence or score term. Even when multiple
semantic actions map to identical mechanical packing, their original selected
semantic log probabilities are retained exactly once. Block-count overflow
still fails closed; dynamic cuts do not repair malformed semantics.

The inherited synthetic truth function still includes its historical K-spaced
task labels plus content-signature cuts. Those labels are not forced commit
positions in this amended run. They must not be confused with newly derived
mechanical capacity cuts. The failed shifted-cut necessity result is retained,
not repaired by redefining labels or packing.

The explicit-carrier model adapter receives sanitized target placeholders,
neutral observations, validated committed lengths/activity and routing.
Teacher h/bytes/lengths are external corruption/loss/control information only.
Full-model placeholder mutations leave predictions/logits bitwise unchanged.

Both shared-parameter applications use the same explicit carrier and
nonstructure keys, actual-S0/conditioning0, L4/Q3/S1/R0. Predictor→commit→replay
is not S2 and has no predicted-x0 re-corruption. Structure action keys bind
root, example, optimizer/actual-S/R/commit occurrence coordinates separately.

H must eventually consume a structural-only commit and validated repacking
at a controlled boundary; evaluation must never own commits or inject teacher
topology. No live G/H integration or H merge is authorized by this record.
