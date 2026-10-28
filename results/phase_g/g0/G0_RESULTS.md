# Phase G.0 — codec remediation and capacity reconciliation

## Outcome and scope

The new, separately frozen `PHASE_G0_VARIABLE_LENGTH_CODEC_V1` passes the
**unchanged historical aggregate compatibility gate** on held-out lengths
8–64. It does **not** establish reliable exact emitted-length recovery.
No G teacher/argmax/sampled arm was run. No G scientific claim is made.
The old failed gate is unchanged, not superseded or relabelled as a pass.

## Representation diagnosis

The accepted F1 target representation is the exact accepted Phase-B codec's
h0. Its parity with F1 Torx means was established in the original G gate.
The new audit uses 3648 probe-training examples and 912 held-out examples,
stratified over every length. Features are h0 and fixed carrier/slot identity
only: no target length, target mask, teacher boundaries or bytes enter the
probe input. Those quantities are labels/support for fitting and scoring only.

On this new stratified sample the original codec decoder obtains 66.1276%
byte accuracy, while independent linear readouts obtain 99.7289%; destroying
h0 drops probe accuracy to 3.1098%, near the 3.125% chance rate. This identifies
a substantial decoder/readout generalization failure, not wholesale loss of
byte information. The old 69.74% figure used a different, much smaller sample;
it is preserved in the historical record.

However, worst-length probe accuracy is 97.2656%, field-signature accuracy is
96.1814%, and partial-extent readout fails badly in some buckets. These fail
the prospective decoder-only eligibility rule. A finite probe failure is NOT
proof of information-theoretic encoder loss. Accordingly, G.0 conservatively
trains the target codec jointly rather than claiming decoder-only sufficiency.

`representation_audit.json` contains every length's byte, signature, exact
emitted-length and partial-carrier results. `counterfactual_audit.json` checks
visible last-byte edits, one-byte length edits and masked-byte edits at every
length. Both old and new representations distinguish every tested visible
edit; masked edits are exactly invariant. This establishes local
distinguishability, not a proof of global invertibility.
The n=8 shortening contrast uses seven bytes only as a diagnostic neighbor;
all training, selection and compatibility examples remain in 8–64.

## Training and frozen identity

The architecture is unchanged: original Mamba byte frontend, target branch,
slot pooling, C=32 / h=64 / Lmax=4 and Mamba byte decoder. Only the 636807
target-codec parameters enter G.0 optimization. Context, DiT, proposal, F1/F3
optimizer state and rho are absent from the codec optimizer.

The candidate starts from a copy of the exact deterministic codec. Fresh
AdamW uses lr=3e-4, weight decay .01, clip 1, batch 32. Byte CE, content-boundary
CE and extent CE have weights one. The prospective protocol, seeds, budget and
selection rule are in `PROTOCOL.md` and the JSON manifests. No tuning occurred.

Update 100 failed the signature threshold. Update 250 was the first scheduled
validation pass, so training stopped there, before final-test exposure. The
untouched test has 32 examples per length (1824 total). Checkpoints at 0, 100,
250 preserve finite parameters and Adam moments; initial moments are all zero.

Frozen artifact (ignored, separate from accepted checkpoints):

`results/runs/phase_g/g0/v1/frozen_codec.pkl`

SHA-256: `0f4a54e221729b14c6e75d30bed84bd566182a490f975c76d574e540f936adab`

The selected optimizer checkpoint is `codec_00250.pkl`, SHA-256
`a6ef46283ac0eada44b9725daacb3d218ba2419a9c3606051f03fc4f8f39619e`.
The frozen parameter tree is bitwise equal to its selected parameters; the
standalone artifact has read-only filesystem permissions. Moments remain in
the resumable checkpoint, not in the frozen inference artifact.

Independent, hash-verified no-overwrite copies of the frozen codec, all three
optimizer checkpoints and matched initialization are also preserved under
`/Users/malachy/src/Adze-Torx/results/runs/phase_g/g0/frozen_v1_archive/`.
See `archive.json`. This ignored archive does not change tracked main or any
accepted F artifact; no symlink or active runner state is shared.

## Held-out compatibility

| Metric | Result | Required |
| --- | ---: | ---: |
| Aggregate valid-byte accuracy | 99.7837% | >=99% |
| Worst length bucket | 99.3448% | >=98% |
| Historical field-signature recovery | 99.3291% | >=99% |
| Exact sequence accuracy, known extents | 92.5987% | reported separately |
| Valid-byte NLL, known extents | 0.00840532 | reported separately |
| Numerical/capacity failures | 0 | 0 |

All 57 per-length rows, NLLs and partial-carrier supports are retained in
`heldout_compatibility.json`. No length buckets were averaged away for the
minimum-byte-accuracy gate. Deterministic versus zero-noise public-Torx parity
is exact for h0, byte logits and both structural heads.

## Important remaining length limitation

The historical signature metric is mean correctness over 31 nonterminal
boundary fields and 32 extent fields. It is NOT exact-structure accuracy and
is dominated by common/inactive fields. That definition was retained, not
changed retrospectively to force either pass or failure.

Post-freeze diagnostics expose:

- exact emitted-length recovery: **58.0044%**;
- exact structure recovery: **57.9496%**;
- final-partial-carrier extent recovery: **43.0060%**;
- byte accuracy when decoding with its OWN predicted extents: **98.0686%**;
- corresponding exact sequence accuracy: **53.2346%**;
- missing bytes: **248**; extra bytes: **611**;
- length-aware reconstruction loss: **0.108342**.

The own-extent accuracy averages per-example correct bytes divided by
max(true length, emitted length). The known-extent gate uses valid-byte
weighting. These have different denominators and are not presented as a
like-for-like paired accuracy delta.

For example length 9 has 100% clean byte accuracy under known extents but only
37.5% exact emitted-length recovery. These are clean-latent reconstruction
diagnostics, NOT prompt-only or unconditional generation. Length-aware loss is
not described as ordinary probabilistic NLL.

The aggregate gate passes, but unconditional claims of reliable length
representation/readout or readiness for generated-length arms are unwarranted.
No extra training or criterion adjustment was performed after this diagnostic.
Any additional partial-length selection criterion or remedial study needs a
prospective version and independent held-out data, not reuse of this test for
tuning. The frozen candidate and its limitation remain available for review.

## Capacity policy and matched controls

`G0_CAPACITY_POLICY.md` specifies full-site SEMANTIC commitment followed by
deterministic dynamic routing cuts. All 108 semantic actions remain in the
distribution; no sample rejection or fixed-position masking occurs. Semantic
and capacity boundaries are different fields. Capacity cuts are not ground
truth semantic labels. The opted-in real model consumes the routed partition.

Exact enumeration and production-sampler MC agree for all 21 parameter/loss
fixtures, using 16 independent roots, 4096 samples each, seed 975 and the fixed
componentwise four-SE rule (zero-SE absolute tolerance 1e-6). Direct continuous
dependencies and the actual generated-length objective are included.

`postfreeze_verification.json` records exact zero-noise mapping parity, no
split intersections, and one common new F1-plus-G.0 initialization for every
future arm. Complete tree SHA-256:
`0f398f0d08d9d78bc0543cc65106e95f3b5981ecf144e3d88b0708a960390f05`.
All noncodec means and rho are unchanged. The shared frontend is deliberately
part of the codec overlay, so all arms must use the same new frontend as well.

## Resume decision

**Do not resume the unchanged original runner or any policy arm.** Its codec
identity, capacity policy, initialization and gate manifests refer to the old
experiment. The old failed gate remains authoritative for that version.

A new versioned engineering preflight can use this frozen candidate and the
new explicit dynamic policy, but must retain the partial-length limitation,
settle its treatment prospectively, and repeat its model gradient/update,
noisy-latent identifiability, structural necessity, no-leakage and
checkpoint/provenance gates. The changed teacher latent geometry has not been
shown compatible with the F1-initialized denoiser beyond the tests reported
here. G.0 does not
clear those downstream gates or establish scientific trainability. All three
arms must use the same new codec, complete initialization, fresh Adam and the
unchanged approved G data (1024/256, seeds 950/951/952). Cross-codec results
are not matched structural-policy comparisons.

No G campaign, H training, language training, S4, merge or push was performed.

## Source and evidence lineage

- Start: `aa02aaff22d27b9d0b654d2fdd47b30e6a05e37d`.
- Prospective protocol, diagnosis, training, selection, test and capacity MC:
  `085d2e1d5b1de7e02b47e84c892ca258c7f4e328`.
- Post-freeze verifier: `289e00fd755856310e7ec481161101f5d7b4aced`.
- Counterfactual verifier and final test source: commit `94a2416` (full identity
  recorded in the machine-readable counterfactual/validation evidence).
- Original failed gate SHA-256 remains
  `c99ca707c3ad3b7e7df057bd4fab93e3f293a0446efbf076fea889680ded448b`.

Reproduction commands (from the G worktree, at the recorded source versions):

```sh
PYTHONPATH=src python scripts/run_phase_g0.py --stage diagnose
PYTHONPATH=src python scripts/run_phase_g0.py --stage train
PYTHONPATH=src python scripts/run_phase_g0.py --stage oracle
PYTHONPATH=src python scripts/run_phase_g0.py --stage verify-frozen
PYTHONPATH=src python scripts/run_phase_g0.py --stage counterfactual
```

Do not rerun into existing evidence paths to overwrite this record. Training
resume rejects a changed protocol and refuses to overwrite durable checkpoints;
a validation-selected endpoint cannot accidentally continue optimization.

Validation results are recorded separately in `VALIDATION.md`.
