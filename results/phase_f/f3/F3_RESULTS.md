# Phase F.3 — self-generated S=2 rollout training

## Decision

**PHASE_F_3_S2_TRAINING_NEUTRAL**

Training was stable and converged under the frozen policy, and F3 preserved the
accepted F1 model's one-step capability. The same trained model at S=2 did not
meet the material-benefit gate in two informative buckets. It produced one
material degradation at `nu=0.25` and neutral effects at `nu=0.50` and
`nu=0.75`; consequently it also did not meet the two-bucket NEGATIVE gate.
The 16-root finite-Torx result reproduced the same practical-effect pattern.

This closes F3 only. It is not a final Phase-F S verdict and does not authorize
S4.

## Protocol and provenance

The accepted F1 parent is
`results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl`, SHA-256
`520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57`.
F3 continued its model means, frozen rho, Adam moments, and absolute optimizer
step 5000. It retained AdamW at learning rate `3e-4`, weight decay `0.01`, clip
norm `1.0`, and batch size 32; `L=4`, `Q=3`, `S_train=2`, `R=0`;
`lambda_op=1`, `sigma_op=1e-3`; deterministic inter-step re-corruption;
native conditioning `[0,1]`; shared parameters; and full pathwise gradients.

The immutable committed baseline is
`cf129ef87fa9b12bf3249dc2d528b57c3813af64`. The implementation files used by
the pre-run provenance gate, gradient gate, training, selection, deterministic
evaluation, and operator evaluation have the common source digest
`62545ce627447f509e6f6abd573c3667fa03a3a2540b18d3eacedcc06e426d4c`.
The complete working implementation diff against that baseline has SHA-256
`4a603f4395a5a6119958a8404f48d7151e77d708620855ee0ad2a856ab3f3ba3`.

The baseline source contained an older escalation rule. The hardened source
was completed before any F3 gate or training update, and its identity is bound
into all run artifacts. It is therefore the launched and governing rule:

- classify the full-test lambda-zero informative-region pattern as benefit,
  neutral, or degradation with the frozen practical gates;
- require the stochastic pattern to match that classification symmetrically;
- at 512 examples per `nu`, escalate to 1024 across the same 16 roots when the
  classifications conflict or a root-level Student-t accuracy interval spans
  both `-1` and `+1` percentage points, or a root-level NLL interval spans both
  `-2%` and `+2%` of S1 NLL.

The precise timestamps, source paths and line references, committed predicate,
launched predicate, and launch/snapshot evidence are in
`protocol_provenance.json`. Deterministic and operator evaluation protocol
digests are respectively
`f51e287ad27096cab62f0cc759cafdb04bdb856bf83367493ce40ebded67cdb9` and
`e8c65f826e80edb36ae2bbcff470daa46edf90065acf30b53260ace995e27d99`.

## Training endpoint and numerical gates

Training completed 10,000 F3 rollout updates, ending at absolute Adam step
15,000. The endpoint is
`results/runs/phase_f/f3/s2/init0_stoch0/rollout_10000.pkl`:

- checkpoint SHA-256: `af08d590fa338301c840e4fc2d795a75f9012c806296e2fb2a9c38003837a6c3`;
- parameter-tree SHA-256: `0c18d71d78bd918437cdbfd95c582fc05f3b2af0ec2380540465075e92701157`;
- Adam-moment-tree SHA-256: `191e0ca9de5df59945a5b475de54f6de89ad4c0d1c2cf2265d1c7fe6228d58b9`.

The 100, 250, 500, 1k, 2k, 5k, and 10k checkpoint manifests, update identities,
protocol/configuration hashes, dataset identities, and F1 parent identity all
match. Training records are unique and finite. `latest.pkl` is byte-identical
to the 10k endpoint. Frozen teacher and rho parameters and moments are bitwise
unchanged at every checkpoint.

The first-gradient gate passed:

| diagnostic | value |
|---|---:|
| `||g0||` | 2.561976 |
| `||g1||` | 19.136314 |
| `||g||` | 9.957397 |
| clipped `||g||` | 1.000000 |
| `cos(g0,g1)` | 0.243100 |
| `cos(g0,g)` | 0.362244 |
| `cos(g1,g)` | 0.992183 |
| connected-minus-detached step-1 gradient norm | 5.327756 |

The prospective full-model update had finite losses, gradients, parameters,
and moments; 165 enabled parameter leaves and 280 moment leaves changed; all
disabled leaves and frozen teacher/rho state remained unchanged. The 10-update
runtime smoke also passed.

Checkpoint-batch training observations are retained as observations rather
than validation scores:

| F3 updates | absolute step | step-0 loss | step-1 loss | rollout loss | step-0 acc | step-1 acc |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 5100 | 2.13981 | 2.15180 | 2.14581 | 0.378906 | 0.382812 |
| 250 | 5250 | 1.68772 | 1.82675 | 1.75724 | 0.511719 | 0.492188 |
| 500 | 5500 | 1.90338 | 1.98243 | 1.94291 | 0.476562 | 0.445312 |
| 1000 | 6000 | 1.70749 | 1.74323 | 1.72536 | 0.539062 | 0.503906 |
| 2000 | 7000 | 2.14794 | 2.18615 | 2.16705 | 0.417969 | 0.363281 |
| 5000 | 10000 | 2.52605 | 2.51921 | 2.52263 | 0.308594 | 0.308594 |
| 10000 | 15000 | 2.23053 | 2.24422 | 2.23737 | 0.402344 | 0.371094 |

## Validation-only checkpoint selection

The selection record was written before final-test metrics were exposed.
Candidates were excluded only for nonfinite informative validation values or
material step-0 forgetting in at least two informative buckets, then ranked
lexicographically by benefit-bucket count, negative degradation-bucket count,
mean informative S2 accuracy, negative mean informative S2 NLL, and update.
All seven candidates were eligible. The 10k checkpoint ranked first with
informative S2 validation accuracy `0.384521` and NLL `2.160811`, versus
`0.374919` and `2.193383` at 5k. No test or operator result entered selection.
The frozen record is `checkpoint_selection.json`.

## Frozen full-test result

The test set contains all 4,096 DENOISE_V1 examples at seed 942 with ordered
global-ID digest
`6b0751ba747e83253565921fb425c077c237092f764056bba34370d2b623e44f`.
Every comparison uses paired examples, corruption, masks, and identities. The
evaluation path has no optimizer and audits parameter, optimizer, teacher, and
rho immutability.

| nu | F1 S1 acc | F3 S1 acc | F3 S2 acc | S2-S1 pp | F3 S1 exact | F3 S2 exact | F3 S1 NLL | F3 S2 NLL | F3 S1 MSE | F3 S2 MSE | effect | forgetting |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| .10 | .995972 | .998627 | .995819 | -0.281 | .989014 | .966553 | .004742 | .014934 | .009716 | .010782 | neutral | no |
| .25 | .789307 | .806274 | .780182 | -2.609 | .166260 | .130615 | .601010 | .690691 | .014394 | .012478 | degradation | no |
| .50 | .272675 | .283081 | .271942 | -1.114 | .000000 | .000000 | 2.478818 | 2.515936 | .026124 | .015817 | neutral | no |
| .75 | .089691 | .093719 | .090088 | -0.363 | .000000 | .000000 | 3.277009 | 3.284255 | .035038 | .016129 | neutral | no |
| .90 | .045593 | .048126 | .046021 | -0.211 | .000000 | .000000 | 3.446512 | 3.444547 | .039535 | .016252 | neutral | no |

Informative-region paired 2,000-resample bootstrap intervals are:

| nu | accuracy change pp [95% CI] | NLL change [95% CI] | relative NLL | MSE change [95% CI] |
|---:|:---|:---|---:|:---|
| .25 | -2.609 [-2.863, -2.362] | +.089681 [.086610, .092736] | +14.92% | -.001916 [-.001940, -.001895] |
| .50 | -1.114 [-1.382, -.851] | +.037118 [.034512, .039542] | +1.50% | -.010307 [-.010345, -.010270] |
| .75 | -.363 [-.565, -.153] | +.007246 [.005619, .008724] | +0.22% | -.018909 [-.018967, -.018852] |

F3 S1 improves over F1 S1 in accuracy and NLL at every `nu`; no bucket meets
the frozen step-0-forgetting rule. S2 nevertheless meets both degradation
thresholds only at `.25`, not in two informative buckets.

## Prescribed diagnostics

Native trained conditioning `[0,1]` is authoritative. Reusing step-zero
conditioning `[0,0]` is substantially worse: at `nu=.25/.50/.75`, S2 accuracy
is `.710724/.254028/.087036` and NLL is `1.05173/2.65444/3.29924`, compared
with native values `.780182/.271942/.090088` and `.690691/2.515936/3.284255`.
This shows that the trained step-1 conditioning contributes materially.

The eta-one diagnostic is also worse. At `nu=.25/.50/.75`, S2 accuracy is
`.612335/.148865/.050354` and NLL is `1.32624/3.52007/3.98236`. It is not part
of the eta-zero primary decision.

Latent geometry improves while decoding often worsens. At `.25`, 3,401 of
4,096 examples have improved latent MSE but worse decoder NLL; at `.50` the
count is 2,753 and at `.75` it is 2,274. Delta-MSE/delta-NLL correlations are
`.09395`, `.08820`, and `.07607` respectively. Distance to clean falls from
`.119826` to `.111542` at `.25`, `.161446` to `.125567` at `.50`, and
`.187019` to `.126827` at `.75`, while cosine-to-clean rises. Thus Euclidean
carrier improvement is not sufficient for improved decoded bytes.

## Finite-Torx confirmation

The selected checkpoint was evaluated sequentially at `lambda_op=1`,
`sigma_op=1e-3`, eta zero, and native `[0,1]` conditioning on the fixed 512
examples per `nu` for all 16 independent operator roots. All shards match the
checkpoint, dataset, protocol, and root derivation; all are finite; no shard
failed or was retried.

| nu | root-mean S1 acc | root-mean S2 acc | delta pp | Student-t 95% CI pp | S1 NLL | S2 NLL | NLL delta | Student-t 95% CI | effect |
|---:|---:|---:|---:|:---|---:|---:|---:|:---|:---|
| .25 | .811523 | .778870 | -3.265 | [-3.323, -3.208] | .598773 | .694454 | +.095681 | [.095592, .095770] | degradation |
| .50 | .279968 | .273361 | -.661 | [-.715, -.607] | 2.466047 | 2.503504 | +.037457 | [.037385, .037529] | neutral |
| .75 | .097321 | .092346 | -.497 | [-.530, -.465] | 3.271199 | 3.277543 | +.006344 | [.006303, .006384] | neutral |

The stochastic pattern exactly matches the lambda-zero pattern and no
root-level interval satisfies the preregistered high-variance predicate.
Therefore `symmetric_practical_effect_agreement=true`, `high_variance=false`,
and `escalation_required=false`; the supported 1,024-example escalation was
not run.

## Rule-based decision

BENEFIT requires material benefit in at least two informative buckets, no
material degradation, paired uncertainty support, preserved S1, and agreeing
finite-Torx evidence. It fails because there are zero benefit buckets and one
degradation bucket. NEGATIVE requires material degradation in at least two
informative buckets with no compensating benefit; it fails because there is
only one. Training is healthy and converged, there is no major forgetting or
gradient pathology, and the practical pattern is not mixed or stochastically
reversed. The frozen rule therefore yields exactly
`PHASE_F_3_S2_TRAINING_NEUTRAL`.

## Validation and environment

Formatting (125 files), lint, type checking (zero diagnostics), dependency
boundaries, and the private-Torx scan all pass. The source-identical accepted
pre-execution gate passed 176 regular and 9 slow tests. At closure, all 176
regular and all 9 slow tests passed in isolated execution. Aggregate-suite
wrappers twice disappeared without returning an exit status; the initially
unreported Phase-D tests subsequently passed one at a time. Full details are
recorded in `validation.json`.

Evaluation environment: macOS 27.0.1 arm64; Python 3.11.8; JAX 0.8.1 on CPU
(`TFRT_CPU_0`); NumPy 2.4.6; SciPy 1.16.3; Torx installed from its public API
at pinned revision `f1fc858ed950ecd41935d15c06d0ec7c5e0674ae`.

Reproduction entry points are:

```text
python scripts/run_phase_f_3.py --stage select
python scripts/run_phase_f_3.py --stage final
python scripts/run_phase_f_3.py --stage operator-shard --operator-root <root>
python scripts/run_phase_f_3.py --stage operator-aggregate
python scripts/run_phase_f_3.py --stage aggregate
```

`artifact_hashes.json` binds the committed evidence package. Large resumable
checkpoints and raw stochastic shards remain under ignored `results/runs/` and
are identified by their manifests and digests.

## Roadmap boundary

The predeclared roadmap forbids beginning S4 until S2 is understood, and says
that if S2 cannot be trained to compose usefully, immediately increasing depth
would add trajectory depth to an unresolved optimization problem. F3 is now
understood as a healthy neutral result, not a useful S2 composition result.
Accordingly S4 is **not automatically eligible** and requires a new explicit
scientific authorization. No S4, G, H, or I work was started during closure.
