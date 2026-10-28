# Phase G23 V1 validated delivery

**G_ENGINEERING_READY — G_INTEGRATION_READY**

Scientific result: **PHASE_G_GENERATED_STRUCTURE_FUNCTIONAL**.
Generated functionality and semantic-routing value: BENEFIT. Sampling versus
stopped-argmax training: UNRESOLVED, not a claim of superiority or equivalence.
See [executive findings](EXECUTIVE_RESULTS.md) for full metrics, uncertainty,
selected checkpoints, training budget and scientific limitations.

G branch evidence commit: `b06218c9b84923f413e89a663f60fcc0b0863901`.
Normal main merge: `399f5902aaadfe3426c96b12548e70f399239ccc`, from unchanged
accepted main `2a0cab0c9c647832596013d56037c842ff3844ea`; no conflicts or rebase.
The merged tracked tree exactly equals the fully validated G tree.

- Complete branch suite: 323 regular + 30 slow passed, zero skips/failures;
  separate focused G/full-model run: 65 passed. All required static checks pass.
- Fresh main: focused G/full-model 65 passed; format, lint, typing, boundaries
  and private-Torx scan passed. Exact model replay reproduced all 52 recorded
  fields on 32 fixed validation examples, with parameters/Adam unchanged.
- The complete 353-node suite was not repeated after the byte-identical merge.
  Fresh main checks are explicitly integration/static/reference replay checks.
- Main's first pyright attempt also scanned pre-existing ignored historical
  source archives and found three unresolved test-helper imports. That record
  remains in `validation_main/`; `validation_main_completion/` records the
  passing static-only import-path correction (`src:scripts:tests`). No source
  or historical archive was changed or excluded to make it pass.
- All 11,932 files in the scientific manifest and all 731 historical baseline
  entries were rehashed successfully after main validation. The preserved raw
  collection log's trailing blank line was the sole explicit Git whitespace
  check exception; the raw output was not edited.

[Delivery metadata](DELIVERY.json), [complete branch validation](VALIDATION.json),
[main replay](validation_main/REFERENCE_PARITY.json), main attempt/completion
logs, and the separate integration manifest bind exact commands and outputs.
Git push/remote synchronization is verified after the delivery-evidence commit;
the final reported Git SHA is not falsely embedded in its own hashed contents.
Ignored binary checkpoints and raw evaluation shards remain local, immutable,
and hash-bound; they are not represented as uploaded Git objects.

Phase H's G dependency is established, but H must still pass the real typed
structural-only commit/repacking/UNKNOWN/shared-weight causal/global/R gates in
[its handoff prerequisites](PHASE_H_HANDOFF_PREREQUISITES.md). No H/I experiment,
language training, S4, further G extension or Thermo-Adze work was performed.
