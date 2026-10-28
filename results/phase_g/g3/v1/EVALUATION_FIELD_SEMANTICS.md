# Reading the unchanged G23 V1 intervention records

This is a source-inspection clarification, not a protocol amendment, new
experiment, changed acceptance rule or rewritten result. The frozen evaluation
kernel remains unchanged. See `scripts/run_phase_g23_v1.py`, especially
`replay_metrics`, `evaluate`, `mean_roots`, `functional`, and the frozen G3
`contrast` implementation in `scripts/science_phase_g23_v1.py`.

For a non-policy intervention, the evaluator first obtains the base policy's
predictions/actions, then replays from the **same original carrier input** with
the declared intervention. This is not another S denoising transition. Generated
extents are held fixed across boundary interventions; teacher mode is explicitly
an oracle control. Interventions are validated and must execute finitely before
their results are saved.

## Recomputed counterfactual measurements

`aligned_metrics` is recomputed from the intervention's actual decoder output.
Consequently `query_accuracy`, `query_exact`, `query_loss`, `query_nll` and its
support, the corresponding global-byte/nonquery metrics, and missing/extra-byte
counts describe the intervened output. `h_mse`, peer reachability and real
replay packing maps are also recomputed for routing interventions. Committed
boundary accuracy/F1 counts and exact structure are recomputed after substitution.
These are the measurements used by the scientific comparisons and gates.

The primary query loss is the frozen length-aware payload reconstruction cost;
overlap-only query NLL is separate. A length-penalized cost is not conventional
probabilistic byte NLL. Query payloads contain three bytes. Global-byte summaries
and their support should not be confused with query-only performance.

## Retained base-policy measurements

The raw intervention dictionaries also retain other fields from the
underlying policy evaluation. In particular, `descendant` and
`loss_per_example` are **not recomputed total counterfactual objectives**.
Do not interpret their repeated values across controls as evidence of no
downstream effect, or use them for intervention loss comparisons. Structural
prediction CE, prediction accuracy, entropy and neutral-to-policy `commit_change`
remain measurements of the underlying predictor/policy, not a new prediction
or churn measurement of the manually substituted boundary policy.

`base_log_prob` retains the policy's original selected-action log probability;
intervened `log_prob` is zero. A manual boundary intervention is not a fresh
sample whose score should be counted. These routing comparisons take no
gradients or optimizer updates. The separately prescribed selected-C gradient
variance diagnostic does compute gradients, but applies no update. No-attention
and direct-decoder controls are explicitly labelled
diagnostics: their retained base packing maps do not establish that the omitted
attention path was executed. `mask_peer_reach`, when present, distinguishes mask
reachability from that diagnostic's removed computation path.

## Averaging and compute scope

Per-root physical actions/maps remain in their original NPZ shards. Root means
are metric estimates, not voted structures or averaged-logit ensemble decoding.
Overlap NLL root aggregation uses its valid-byte support within each example.
Summary NLLs macro-average valid examples; they are not a pooled micro-average
over every byte of unequal-length sequences. Example bootstrap
uncertainty, training-seed uncertainty and stochastic-root variability remain
distinct as prescribed by the frozen protocol.

All training arms execute two shared-weight applications per counted update.
Intervention evaluation additionally computes a replay/diagnostic to hold the
original policy decision fixed; this is evaluation overhead, not additional
training or a deeper trajectory. The 18,000/216,000 application/block counts in
`CAMPAIGN_RESOURCE_SUMMARY.json` explicitly cover counted training updates only,
not these evaluations, gradients, preflight or backward computation. No inference
compute-efficiency advantage is inferred from those counts.
