# Phase I language preparation — interface record v0

Status: superseded preparation prototype. The corrected branch-local contract is
`phase_i_remediation_v1.md`; neither document is a scientific milestone.

## Frozen constraints

- Input and output remain raw UTF-8 bytes with a 256-way categorical alphabet.
- Padding is represented by a separate boolean mask; byte value `0` remains valid data.
- The shared frontend, split context/target-analysis paths, carrier, hard pack, looped
  DiT, Mamba/SSM decoder, and deterministic/Torx correspondence are unchanged.
- `results/phase_b/checkpoints/target_codec_b1.pkl` and the Phase-F target teacher are
  immutable. `PHASE_I_FULL_BYTE_CODEC_V0` is fresh-seeded and separately checkpointed.
- No authoritative natural-language run is permitted before F completes, then G and H
  integrate and pass their gates.

## Interfaces delivered

`adze_t.language_data` provides local text/JSONL ingestion, explicit normalization,
UTF-8-boundary-safe byte segmentation, global exact deduplication before splitting,
seeded content-hash split assignment, independent byte masks, auditable manifests, and
hash-checked JSONL artifacts.

`adze_t.phase_i` provides the separately versioned codec candidate, complete-byte
calibration corpus, deterministic epoch ordering, batch construction, fresh
initialization, codec-only training and per-byte validation, scaling configurations,
and non-authoritative small-run recipes.

`adze_t.language_eval` provides conditional and infilling cases, a byte-generator
protocol, exact/aligned/edit/UTF-8/repetition/diversity quality metrics, and explicit
wall-time, throughput, memory, model-application and FLOP fields.

`scripts/run_phase_i_language.py` is a resumable preparation driver. It never opens the
accepted codec. Candidate checkpoints and moments live only below
`results/phase_i/candidates/full_byte_codec_v0/`.

## Proposed shared-interface changes

These are proposals for integration review, not cross-branch edits:

1. Add a public inference-only byte-generation entry point accepting prompt bytes,
   optional suffix bytes, `max_bytes`, explicit PRNG key/root, and Q/S/R controls.
2. Add a typed model-size loader that maps a reviewed language scale into the existing
   `ReferenceConfig`; do not mutate `REFERENCE_SMALL_V0`.
3. Have G expose its committed-structure snapshot type to the generation harness.
4. Have H expose draft/refine state traces and model-application counts to compute
   metrics without the harness reaching into private orchestration state.
5. Version checkpoint manifests independently from the Phase-B/F teacher manifest.

## Unresolved dependencies

- G must freeze generated-structure commit and packing interfaces.
- H must freeze R-step orchestration and trace interfaces.
- F3 must finish while its source/protocol identity remains untouched.
- The generation API cannot be wired to a live model until G then H integrate.
- Corpus licensing, source revisions, removal rules, and privacy policy require an
  approved corpus specification before real data ingestion.
- Scaling values in `phase_i_language_scales_v0.yaml` are capacity plans, not accepted
  science. Memory/compile profiling is required before promotion beyond `smoke`.
- Stop/EOS behavior for unconstrained byte generation needs a frozen policy; this
  preparation harness currently evaluates bounded target-length output.
- Near-duplicate detection beyond exact byte equality needs a predeclared method and
  false-positive audit before use.

## Resource requirements

- Corpus preparation: CPU/storage only; streaming-scale implementation should replace
  the current in-memory preparation path before very large corpora.
- Codec smoke: one JAX process, reference-small model, 128-byte sequences.
- Small/medium LM configurations require dedicated compile-memory profiling and disk
  budgets after F/G/H; they must not run concurrently with the active F3 experiment.
- Authoritative evaluation needs paired deterministic/Torx runs and explicit Q/S/R
  application accounting.

## Exact integration prerequisites

1. F3 completes, records one decision, and releases main from provenance-sensitive
   checkpoint resumption.
2. This branch is rebased/cherry-picked onto that completed main and all static/unit,
   boundary, regular and slow suites pass.
3. G integrates first and publishes its committed-structure interface and gates.
4. H integrates second using G's frozen interface and publishes refinement traces.
5. Shared generation/config proposals above receive review; no hidden architecture
   substitution is accepted.
6. `PHASE_I_FULL_BYTE_CODEC_V0` passes full 0..255 per-byte coverage, finite loss,
   held-out byte accuracy and exact-sequence gates under a separately approved plan.
7. A versioned, licensed, deduplicated corpus manifest is approved.
8. Only then may a non-smoke natural-language training milestone be authorized.

## Reproducible smoke recipe

1. Prepare local files with `run_phase_i_language.py prepare`, recording every input
   revision outside the model code.
2. Review `manifest.json`, especially cross-split disjointness and byte coverage.
3. Generate a non-authoritative recipe bound to that manifest.
4. When resource use is approved, train only `PHASE_I_FULL_BYTE_CODEC_V0` from its
   fresh seed and validate against the generated all-byte calibration set plus held-out
   natural segments.
5. Do not load, overwrite, or rename the accepted Phase-B/F codec checkpoint.
