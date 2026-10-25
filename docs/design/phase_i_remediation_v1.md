# Phase I engineering remediation v1

Status: branch-local engineering implementation; no codec update, training run,
generated-language evaluation, integration claim, or scientific result.

Review authority: `PARALLEL_GHI_CODE_REVIEW.md` SHA-256
`8e1bd7ef30525af696c0bb093331602542206bb3d7450020557d9af0c158eb97` and
the approved isolated Phase-I remediation instruction.

## Preserved architecture and artifact boundary

- The external alphabet remains raw bytes `0..255`; padding has a separate mask.
- The existing shared byte frontend, target-analysis Mamba/SSM path, carrier,
  hard pack, DiT, decoder, and deterministic/Torx topology are unchanged.
- The accepted Phase-B/F codec and teacher are immutable and are never loaded as
  a parent. `PHASE_I_FULL_BYTE_CODEC_V0` remains a fresh, separate candidate.
- Candidate checkpoints live under a run-protocol hash and identify codec-owned
  leaves separately from fresh frozen non-codec leaves. Resume recomputes the
  tree description from parameters and both Adam moment trees and requires it
  to match the checkpoint transaction rather than trusting the serialized
  summary.

## Data protocols

`PHASE_I_NATURAL_TEXT_BYTES_V0` reads source bytes with strict UTF-8 decoding
and preserves their raw newline and Unicode representation. It performs no
newline, NFC, NFKC, or other normalization.

`PHASE_I_RAW_BINARY_V0` reads arbitrary bytes without claiming UTF-8 scalar
boundaries. Text and binary documents cannot be mixed in one prepared corpus.

Exact content duplicates share a group ID. Groups are deduplicated and assigned
to a split before segmentation; every overlapping window from a group therefore
stays in one split. Positive-fraction empty splits fail preparation explicitly.
Near-duplicate detection remains deferred and is not silently approximated.

Every serialized segment binds protocol, split, document, exact-content group,
source/source hash, offsets, content hash, byte values, canonical segment ID,
and full-record hash. Loading validates byte range, offset/length agreement,
identities, record order, uniqueness, split ownership, source-instance hash,
cross-split group disjointness, deterministic group assignment, derived document
and byte-coverage accounting, and canonical segmentation bounds. UTF-8 window
advancement uses scalar boundaries without dropping a scalar that crosses a
nominal stride boundary.

## Incremental codec preparation

The batch API computes one deterministic epoch permutation and materializes only
the requested batch. Validation and test use lazy batch iteration. The frozen
engineering calibration policy places canonical all-byte calibration batches at
the start of each epoch, guaranteeing exposure within the calibration prefix.

Three evidence classes remain separate:

1. training byte exposure;
2. evaluation input coverage;
3. reconstruction correctness.

Validation aggregates byte loss by valid emitted-byte support, boundary loss by
scored boundary support, extent loss by carrier support, and sequence accuracy
by sequence count. Validation and test evidence retain per-byte counts.

## Reproducibility and recovery

`PHASE_I_CODEC_RUN_PROTOCOL_V0` hashes source commit and implementation files,
dataset and resolved scale, candidate and seeds, update/cadence settings,
optimizer scalars, calibration/evaluation/structure/refinement protocols, Q/S/R,
and runtime/backend identity. Any change produces a different candidate path.

Each durable step is a transaction:

1. immutable checkpoint payload;
2. checkpoint metadata with payload SHA-256 and tree/ownership identity;
3. evidence record with checkpoint, metrics, and transaction hashes;
4. atomic latest pointer.

Checkpoint metadata may reference only the exact canonical relative payload path
`checkpoints/step_{step:08d}.pkl`; absolute, traversal, alternate-name, and
cross-step paths fail normal reconciliation and are quarantined by explicit
recovery. Normal resume also requires `latest.json` to match the last complete
transaction exactly.

Resume accepts only the last mutually durable matching checkpoint/evidence pair.
One-sided records are reported incomplete. Mixed protocol, checkpoint, payload,
metric, exposure, coverage, or reconstruction identities fail closed. Durable
steps are never overwritten. The explicit `recover` command moves incomplete,
stale, malformed, duplicate, and temporary artifacts into a numbered quarantine,
writes a hashed quarantine manifest, and reconstructs `latest.json` from the last
complete pair. Recovery never silently deletes an artifact.

## Scale registry

`configs/phase_i_language_scales_v0.yaml` is the sole scale source. The typed
loader resolves and hashes a complete `ReferenceConfig` and checks byte,
carrier/emission, attention-width, packing, and batch constraints. Only `smoke`
is marked executable as an engineering candidate. `small` and `medium` remain
explicit unresolved numeric proposals and the runner refuses them.

## Evaluation contract

The harness uses explicit task types:

- teacher-latent reconstruction;
- denoised-latent reconstruction;
- conditional generation;
- unconditional generation;
- infilling.

Each case owns a preregistered byte budget independent of its expected output.
Expected bytes are scoring-only and unconditional cases cannot contain them.
Roots derive from the global root and stable case ID, not list position. Metrics
are emitted separately by task; reference-only metrics are absent for
unconditional generation. Case IDs must be unique, outputs must be raw `bytes`,
and each task reports an aggregate 256-entry generated-byte histogram and
coverage count.

## Deferred G/H inference integration

Main does not yet expose validated G/H inference records. I therefore does not
invent a live generated-structure adapter. The eventual read-only result must
carry committed structure and validated routing, dynamic length and all masks,
S/R carrier traces, byte logits/output bytes, prompt/suffix masks, explicit
diffusion/operator/structure roots and case IDs, Q/S/R and conditioning/mode
coordinates, resource metrics, and versioned provenance. Evaluation may not
commit structure, mutate model state, or inject teacher topology at generation.

### Exact integration prerequisites and unresolved decisions

Integration requires, in order: accepted G routing/commit records and overflow
guards; accepted H draft/refine result and trace schemas; an adapter proven on a
real shared-weight model path; dynamic committed-length and all query/KV/pool/
emission masks; explicit diffusion/operator/structure roots and Q/S/R/mode
coordinates; then a read-only I consumer test with no teacher topology or state
mutation. A compiling structural type alone is not sufficient.

Still unresolved are corpus source/licence/privacy/removal policy,
near-duplicate policy, reviewed small/medium numeric scales, authoritative codec
and language-training protocols, and the generated-structure stopping semantics
that will implement the declared bounded-byte policy. These are not assigned
defaults on this branch.

Corpus preparation and artifact verification require CPU time plus storage
roughly proportional to source bytes and serialized segments. Later smoke codec
validation requires one coordinated JAX compile and candidate checkpoint space.
Small/medium execution requires separate compile-memory, wall-time, and disk
profiling. Natural-language training and authoritative generation evaluation
remain separate, resource-intensive authorizations.

## Adversarial regression matrix

Branch tests cover strict raw UTF-8 and invalid input; full-domain binary input;
exact duplicate grouping; overlapping-window confinement; positive-fraction
empty splits; byte-zero masks; mutated ID/group/source/split/offset/value/record;
duplicate IDs and order changes; canonical scale hashes; protocol-field drift;
stale resume identities; one-sided crash artifacts; durable-step overwrite;
payload-only, temporary, duplicate, and stale-protocol quarantine; independent
hash binding for exposure/coverage/reconstruction evidence; task separation; stable
case roots; non-oracle budgets; unconditional metric separation; and generator
budget enforcement; all five task classes; generated-byte aggregation; missing
and stale latest pointers; noncanonical and malformed payload names; mutated
checkpoint payloads; exact parameter/Adam-tree resume identity; nonfinite and
invalid protocol scalars; and UTF-8 stride-boundary preservation.

Deferred gates requiring later authorization are a real finite codec optimizer
update, all-byte trained reconstruction evidence, full regular/slow/model suites,
G/H inference integration, and any natural-language training.

## Branch-local validation record

- Focused Phase-I adversarial suite: `46 passed`.
- `make format-check lint typecheck boundaries`: passed; Pyright reported zero
  errors and zero warnings, and the boundary/private-Torx scan passed.
- `git diff --check`: passed.
- A duplicate regular-suite invocation was deliberately interrupted during
  resource coordination after `65 passed, 9 deselected`; this is recorded as a
  non-result and is not claimed as validation.
- The regular and slow suites, a real finite codec update, codec reconstruction,
  and live G/H integration remain pending their coordinated gates. No
  `I_ENGINEERING_READY` decision is issued by this branch-local record.
