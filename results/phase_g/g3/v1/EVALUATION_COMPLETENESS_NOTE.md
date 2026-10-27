# Frozen selected-validation confirmation completion

Prospective implementation note, before final selection or protected-test output.
No G2/G3 protocol or scientific decision rule is changed.

The frozen G3 protocol's matched-training section specifies that additional
selected-validation16-root confirmation cannot reselect; its frozen-evaluation
section covers the full912 protected test and selected912 validation examples,
with sixteen sampled-policy roots and paired cut/neutral controls. The bound
`science_phase_g23_v1.py` already performs those sixteen-root test interventions
and five deterministic selected-validation interventions, but does not itself
complete sixteen-root selected-validation sampling.

`scripts/complete_phase_g23_validation.py` fills that evidence-completeness gap
using ONLY the unchanged `g.evaluate`, `g.mean_roots` and `g.summary` APIs.
After all nine1000-update caps and complete validation-only `SELECTION.json`,
run roots0..15 of sampled policy/cut/neutral for EVERY selected model, all three
seeds/arms. Existing matching root0..3 shards are reused, never discarded or
reselected. Cache checkpoint/protocol/example/root identity checks remain those
of the frozen evaluator. Preserve actual per-root commitments/routing/emission
and metrics in the existing strict shards. These are expectations of single-draw
policies, not averaged logits/ensemble outputs. Check model/moment bitwise trees
before/after, serialize per-root summaries and root variability, and retain the
unchanged selection identity. No optimizer, training or protected-test generation
is performed by this driver; no new threshold or comparative decision is added.

Execute sequentially after the campaign and before authoritative test evaluation
where feasible. The campaign lock forbids overlap with training/science. If
interrupted, immutable matching evaluation shards permit exact resumption; do
not silently overwrite mismatches or resample roots. Primary scientific decisions
remain exclusively the frozen G3 procedure; confirmation cannot change selection.

The recovery storage forecast counted only the bound driver's deterministic
selected-validation passes. This completion adds at most432 archives; at the
same conservative2MiB each, reserve an additional1GiB. The whole remaining
project budget including the unchanged2GiB guard is therefore8GiB, not7GiB.
The actual recovery audit recorded17GiB before/21GiB after and subsequent
monitoring about48GiB, so headroom is still sufficient. The original recovery
measurements/budget record is preserved, not rewritten. No cleanup occurred.

Reproduction (only after training and frozen selection):

```sh
nice -n 10 env PYTHONPATH=src:scripts python scripts/complete_phase_g23_validation.py
```

This is completion of predeclared validation evidence, NOT a new experiment,
new oracle adjudication, adaptive root escalation or retrospective policy edit.

## Temporal structure churn (derived existing evidence only)

The frozen G2 reporting clause distinguishes temporal policy churn on fixed
examples from neutral-to-commit changes. The kernel's `commit_change` is the
latter. Complete the former with `scripts/summarize_phase_g23_churn.py` after
frozen selection. It uses existing per-root validation action/routing arrays at
0/100/250/500/1000, preserving original example order and identical root words.
For sampled training compare each of the four COMMON checkpoint-validation roots
before averaging disagreement rates; never average actions before measuring
churn. Predicted versus committed changes, emission/activity changes, mechanical
capacity cuts and packing coordinates are reported separately. Teacher commits
may remain identical while their predicted heads change. A packing-coordinate
change is not a loss of persistent carrier identity.

The postprocessor verifies archive/checkpoint hashes, dataset/example identities,
source/protocol binding and root coupling. It performs no inference, sampling,
gradient, optimizer update or decision override. Seven lightweight exact/file
oracles cover rootwise versus averaged-action changes, teacher prediction/commit
separation, reactivation, empty support and stale checkpoint/example rejection.
Those tests are not evidence of model trainability; real execution of the
selected-validation driver and full scientific gates remain required.
