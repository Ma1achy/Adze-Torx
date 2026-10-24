# Phase G generated-structure interface record

Status: engineering implementation only; no scientific gate or training result.

## Frozen interface

`PHASE_G_GENERATED_STRUCTURE_INTERFACE_V0` keeps three structural coordinates
separate:

- observed `s_b/s_l`: model inputs only, never routing authority;
- predicted `b_logits/l_logits`: supervised distributions;
- committed `c_b/length/activity`: the only inputs to hard pack and emission.

`commit_predicted_structure` supports controlled `prediction_only`, stopped
`argmax`, and explicitly keyed `sample` policies. A site mask permits commits at
declared boundaries. Carrier indices never change. The final boundary is a
forced terminal sentinel and is excluded from sampled log probability.

`generated_routing` is the fixed-to-generated adapter. Both fixed teacher
structure and committed generated structure enter the existing hard pack through
the same `CommittedStructure -> PackMetadata` contract at static `M_max, K`.
Overflow is surfaced in metadata and is never silently truncated.

`ell=0` derives `activity=False`: the site stays query/identity present but is
key/value, pooling, and emission inactive. A later nonzero committed extent
reactivates the same carrier identity.

## Gradient ownership

Prediction-only and argmax commits use ordinary boundary/extent cross-entropy;
argmax decisions stop gradients. Sampled downstream commits return only the log
probabilities of actual, selected stochastic occurrences. Their descendant loss
may be passed to the restored, byte-for-byte-semantic M1.5 local score bridge.
Native Torx estimators must not be wrapped, and continuous paths remain ordinary
pathwise JAX gradients.

## Diagnostics

The interface reports boundary/extent/activity churn, reactivation and
deactivation separately. Calibration covers accuracy, NLL, Brier score, and
fixed-bin ECE; the forced terminal boundary is excluded.

## Proposed shared-interface changes

Integration will require `apply_model` to accept a `CommittedStructure` or the
equivalent `GeneratedRouting`, instead of three loosely coupled optional arrays.
That proposal is intentionally not applied on this branch while F3 provenance is
live. The existing scalar-array arguments can be adapted at the integration
boundary without changing model topology.

## Unresolved dependencies

- The Phase G training/evaluation protocol must freeze commit timing, site masks,
  sampling temperature, and block-churn acceptance gates.
- Live sampled categorical factors must use public Torx APIs and prove
  sample/log-probability consistency before authoritative training.
- Overflow bucket policy remains the architecture's documented provisional
  micro-semantic; authoritative runs need an explicit bucket choice.
- Phase H must consume this committed interface only after G gates pass.

## Resource requirements

Current tests are eager, small-array JAX unit tests. Integration later requires a
focused JIT/full-model pass, score-gradient enumeration/Monte Carlo oracles, and
eventually generated-structure training; none are run while F3 is active.

## Exact integration prerequisites

1. F3 completes and its source/protocol evidence is committed.
2. Shared-interface proposal is reviewed.
3. Full regular/slow suites and public/private Torx boundary scans pass.
4. Boundary and extent recovery, internal deletion, reactivation, UNKNOWN
   isolation, overflow, and churn gates are preregistered and pass.
5. Sampled downstream structure passes the existing score-gradient oracle with
   no estimator double counting.

No Phase H live rollout or Phase I training may rely on this branch before those
conditions hold.
