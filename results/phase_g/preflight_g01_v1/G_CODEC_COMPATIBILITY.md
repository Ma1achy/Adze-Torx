# Frozen codec compatibility reconciliation

The accepted G.0.1 evidence remains
[G01_RESULTS.md](../g0_1/G01_RESULTS.md). Its protected test was not rerun or
reselected during this preflight. Its frozen receipt is
[frozen_codec.json](../g0_1/frozen_codec.json).

Artifact:`results/runs/phase_g/g0_1/v1/frozen_codec.pkl`.

SHA-256:`213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7`.

Parameter-tree hash:
`e65b5d6dc6e1300dd80c6bd8046fb9df469cab78842009d4b15db9369fa4ad15`.

Inherited held-out results on3648 examples across lengths8–64:

| Metric | Accepted G.0.1 result |
| --- | ---: |
| Known-extent clean byte accuracy |99.469%|
| Worst length bucket |98.684%|
| Inferred-extent clean byte accuracy |99.476%|
| Known/inferred exact sequence |83.169%/83.169%|
| Known-extent NLL |0.0194321|
| Exact total length / extent vector / partial extent |100%/100%/100%|
| Exact complete b/ell structure |99.589%|
| Historical signature recovery |99.965%|
| Missing/extra bytes |0/0|
| Numerical/capacity failures |0/0|

The accepted G.0.1 oracle-byte tradeoff relative to G.0 remains disclosed in
that record; high byte accuracy does not imply perfect sequence reconstruction.
These codec scores are not the G denoiser's scores.

New evidence verifies exact artifact identity, exact target-side encoder
overlay, deterministic/Torx teacher representation equality, matched complete
initial tree and fresh zero Adam moments. Noncodec F1 means/rho are copied
exactly; F1 moments/step5000 are not continued. The predictor receives only
corrupted frozen-teacher h and target-independent neutral structure/masks.
This is structural prediction conditional on a target-latent representation,
**not prompt-only language generation**.

The frozen codec artifact and target encoder remain unchanged. The separately
initialized model output decoder trains in Arm A under the approved update
mask. This does not mutate/retrain the frozen external teacher artifact.
Post-execution historical preservation covers727 files, including accepted
F1/F3/G.0/G.0.1 checkpoints/evidence and canonical teacher fixtures.

The current blocker is structural necessity, not codec compatibility. No codec
criterion was weakened and no accepted teacher was modified.
