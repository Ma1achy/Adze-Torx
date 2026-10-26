# Phase G.0.1 — length-information investigation

## Decision

The new, separately frozen G.0.1 codec passes every prospectively declared
clean structural, byte-compatibility and noisy-input readiness criterion.
**The codec/length-information blocker is cleared for the declared synthetic
domain.** This is not a Phase-G scientific result and does not establish that
the F1-initialized G predictor already generates correct structure.

The unchanged original G smoke runner must NOT be launched. An explicitly
resumed, versioned G preflight can use this candidate after the amendments
below and must still pass its original full-model optimization and structural
necessity gates. No teacher/argmax/sampled smoke arm was run here.

## Prospective protocol and provenance

Start branch: `1b4eaf57b93691952c4046eddfaeb1ceddd64ec6`.
`PROTOCOL.md`, candidate criteria, probe budgets, selection rule and all code
were committed before diagnosis/candidate execution at
`c80920efd4eb8cdd8c15e684c341ab990370aa3f`. No criterion was weakened after outcomes.

Parent: frozen G.0 codec, SHA-256
`0f4a54e221729b14c6e75d30bed84bd566182a490f975c76d574e540f936adab`.
It remains unchanged and read-only. F1/F3 checkpoints and optimizer states
were not modified. No existing encoder, decoder, sampling or score-bridge
source was changed: this increment adds isolated modules/scripts/evidence.

New data are iid bytes 1..32, target lengths 8–64, width64/C32/Lmax4:

| Split | Seed | Examples/length | Total |
| --- | ---: | ---: | ---: |
| Probe fitting | 1010 | 128 | 7296 |
| Validation | 1011 | 32 | 1824 |
| Codec training | 1012 | 256 | 14592 |
| Protected test | 1013 | 64 | 3648 |

All four splits have zero exact sequence intersections. Supplemental checks
also found zero intersections between G.0 train/test and G.0.1 train/test.
Ordered dataset identities are in `postselection_audit.json`. Test was not
exposed until validation-only checkpoint selection.

## Frozen-representation diagnosis

Probes fit TRAIN ONLY, including standardization. They receive h and fixed
site identity where declared, never target mask, teacher b/ell or true length.
Global diagnostic heads also predict total length, emitting count and final
partial class directly. These global heads underperformed totals derived from
the better local extent readouts; all results, not just the best, are retained.

The local linear probe is compatible with the production linear extent head.
The diagnostic MLP has 64 hidden GELU units and constant site identity. It is
an instrument, NOT an architectural replacement or installed candidate head.
Each probe uses the preregistered fixed 800 updates; no checkpoint search.

Validation results below use totals derived from predicted per-site extents:

| Frozen G.0 input | Existing head total / partial | Linear total / partial | MLP total / partial |
| --- | ---: | ---: | ---: |
| clean h0 | 58.607% / 43.824% | 85.252% / 97.247% | 99.123% / 99.926% |
| nu=.025 | 58.279% / 43.378% | 83.717% / 96.652% | 99.232% / 99.926% |
| nu=.10 | 56.908% / 41.518% | 75.932% / 92.336% | 97.807% / 98.586% |
| nu=.25 | 50.713% / 33.259% | 54.057% / 78.497% | 78.728% / 84.747% |

Thus clean h0 retains substantial usable length information: the old head's
failure is not evidence of irreversible encoder loss. But the unchanged
linear readout misses the declared criteria, and even the tested nonlinear
readout is unreliable at the upper corruption level. The prospective rule
therefore selected **joint target-codec representation/readout remediation**.
This is a conservative decision under the finite probe family and required
linear-head topology, not a proof that no conceivable frozen-h readout works.

Controls were fitted from training labels only. Global majority total-length
accuracy is 1.754%, emitting-count prior 7.018%, partial-class prior 26.316%.
The local position/length-prior baseline has 1.754% exact total length; the
global per-site majority predicts all-zero and has zero exact total length.
The explicitly labelled teacher-structure upper bound is 100%.

All lengths, partial classes, class0..4 confusion matrices, supports and
zero/nonzero statistics are in `diagnosis.json`. `per_site_class_audit.json`
adds every carrier's supported-class recall/confusion and reproduces the saved
probe outcomes exactly. Absent classes are null, not invented perfect scores.

## Actual G input and existing G head

The actual G explicit carrier input is bitwise h_nu. This was verified through
the real shared frontend/context/proposal/pack/DiT/unpool/residual graph at each
noise level, using a fixed228-example validation subset (four per length).
Observed structure and routing are fixed neutral constants, prompt is 0xF1,
and the forward target object contains only inert sanitized content. No true
length/mask/boundary is supplied to this inference graph.

Accordingly actual-input probe results alias the corresponding noisy-stage
results; they are NOT portrayed as independent evidence. The existing G
prediction head was evaluated separately. Its exact-length accuracy is only
about 4–5% with G.0 and 8–10% with G.0.1. Those noncodec F1 means have not been
trained for this task. A good target-codec readout does not mean that this
untrained G predictor already commits correct structure.

## Remediation and selection

The original Mamba frontend/target branch/slot pooling/h64/linear b and ell
heads/Mamba decoder topology is unchanged. No diagnostic MLP is installed.
The candidate starts from an exact copy of G.0 parameters and fresh Adam
moments/step0, not a continuation of its optimizer.

Loss, frozen before training: clean byte CE + clean semantic-boundary CE +
class-balanced clean extent CE + class-balanced noisy extent CE, weights1.
Equal class mass prevents the many inactive sites from hiding partial-extent
errors. Noisy h uses the unchanged cos/sin corruption and fresh unit Gaussian
epsilon, nu uniform(.025,.25). AdamW: lr3e-4/wd.01/clip1/batch32.
Only the original target-codec subtree is optimized; no rho, context, proposal,
DiT, outer refinement or accepted model optimizer state is updated.

Step0 failed. **Update100** was the first scheduled validation pass for BOTH
clean criteria and all noisy-input criteria; it was selected before test.
Training stopped there. Test was evaluated once; no later checkpoint was
selected or trained after seeing test outcomes. Checkpoints0/100 and optimizer
state are preserved; no numerical update or persistence failure occurred.

Frozen candidate:

`results/runs/phase_g/g0_1/v1/frozen_codec.pkl`

SHA-256: `213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7`

The parameter tree hash is
`e65b5d6dc6e1300dd80c6bd8046fb9df469cab78842009d4b15db9369fa4ad15`.
The artifact has read-only permissions. Independent no-overwrite copies of
both checkpoints, frozen codec, optimizer state, matched initialization and
all eight probes are archived under the durable ignored directory:

`/Users/malachy/src/Adze-Torx/results/runs/phase_g/g0_1/frozen_v1_archive/`

All13 copied files were independently hash-verified against their sources.

## Protected held-out structural and byte results

These are empirical results on 3648 examples, not guarantees outside this
domain. Every one of 57 target lengths and all three partial classes is scored.

| Metric | Result | Prospective criterion |
| --- | ---: | ---: |
| Exact total emitted length | 100% | >=99%, each length>=95% |
| Emitting carrier count | 100% | reported |
| Per-carrier extent class | 100% | >=99.5% |
| Exact extent vector | 100% | >=99% |
| Final partial extent | 100% | >=99%, each partial class>=98% |
| Zero recall / nonzero recall | 100% /100% | both>=99% |
| Exact complete b/ell structure | 99.589% | >=99% |
| Historical field-signature | 99.965% | >=99% |
| Known-extent byte accuracy | 99.469% | >=99% |
| Worst known-extent length bucket | 98.684% | >=98% |
| Inferred-extent byte accuracy | 99.476% | >=99% |
| Worst inferred-extent length bucket | 98.684% | >=98% |
| Known/inferred exact byte sequence | 83.169% /83.169% | reported |
| Known-extent byte NLL | 0.0194321 | reported |
| Inferred length-aware loss | 0.0190104 | reported |
| Missing / extra bytes | 0 /0 | reported |
| Numerical / capacity failures | 0 /0 | zero |

Class supports: zero=82560, class1/2/3 each=896, class4=31488. All five class
recalls are 100%; partial support=2688. The complete structure misses a small
number of b predictions, not extents. Historical field averaging is no longer
used as a substitute for exact structure.

No teacher length enters `decode_inferred_extents`: its only inputs are h0,
the learned head and decoder parameters. Oracle length reconstruction is a
separately labelled upper-bound path. Both paths flatten/align global emitted
byte order and retain missing/extra penalties. The penalized loss is NOT
called conventional byte NLL. Known-extent accuracy is valid-byte weighted;
inferred accuracy averages correct/max(true length, emitted length) per
example. Their different denominators are explicit.

Noisy frozen-head test results:

| nu | Exact total / extent vector | Partial | Per-site extent |
| --- | ---: | ---: | ---: |
| .025 | 99.973% | 100% | 99.999% |
| .10 | 99.973% | 100% | 99.999% |
| .25 | 99.753% | 100% | 99.992% |

Every noise stage passes its declared95% exact/vector/partial and99% per-site/
zero/nonzero thresholds; each partial class passes. Actual G input aliases
these carriers, but the G model's own untrained prediction head remains a
separate baseline, not silently replaced by this codec head.

## Explicit byte tradeoff and oracle dependence

Paired test comparison at the exact same examples/corruptions, after selection:

| Metric | Frozen G.0 | Frozen G.0.1 |
| --- | ---: | ---: |
| Oracle-length byte accuracy | 99.775% | 99.469% |
| Oracle-length byte NLL | .00842007 | .0194321 |
| Oracle-length exact sequence | 92.270% | 83.169% |
| Inferred-length byte accuracy | 98.005% | 99.476% |
| Inferred-length exact sequence | 52.083% | 83.169% |
| Inferred missing / extra bytes | 521 /1267 | 0 /0 |
| Inferred length-aware loss | .110526 | .0190104 |

The old oracle-length success concealed a genuine inference limitation.
G.0.1 removes that oracle dependence for these extents, but costs .305pp in
oracle byte accuracy, 9.101pp in oracle exact sequence, and .011012 NLL.
The unchanged byte gates still pass; the tradeoff is NOT hidden, and no claim
of uniform quality improvement is made. No further test-driven tuning ran.

h0 RMS changes from .740888 to 1.082174. Noise coefficients and unit-epsilon
scale did not change. Improved length robustness includes the learned change
in latent geometry/margins; it is not an F1/F3 denoising-depth result. Future
arms must all use this same codec and geometry.

## Sampling preservation and validation

Semantic b versus capacity cuts and the action law are unchanged:
`PHASE_G0_SEMANTIC_PUSHFORWARD_DYNAMIC_CAPACITY_V1`. Git audit confirms no
changes to generated_structure, G.0 routing, either exact oracle or the local
score bridge. All 84 relevant tests passed in 63.86 seconds, including the 21
Monte Carlo tests and exact finite-state conformance. The MLP probe is never
part of that action distribution or estimator.
Those MC tests are the existing local-bridge fixture grid. The separately
recorded production capacity-pushforward MC evidence from G.0 is preserved,
not claimed as rerun here; its implementation and distribution law did not
change. The exact dynamic-capacity regressions were rerun in the 84-test set.

Format143 files, lint, Pyright (zero errors), boundaries, diff-check and the
explicit private-Torx scan pass. See `VALIDATION.md` for exact commands, source
identities and the full-suite scope limitation.

## G-preflight readiness and required amendments

**Codec/length gate: ready. Unchanged old runner: NOT ready.** On a subsequent
explicit resume, version the G experiment rather than overwrite the old gate:

1. Bind this new frozen codec SHA to teacher construction, initialization,
   checkpoints and every result. Preserve the old G.0/F/B records unchanged.
2. All teacher/argmax/sampled arms start from the same prepared complete tree
   (hash `84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb`),
   fresh Adam/step0 and the approved seeds950/951/952/counts1024/256. No mixing
   codec versions, and no transferring candidate codec moments into G.
3. Wire the validated full-site/dynamic-capacity policy through the new G
   optimizer wrapper; do not accidentally invoke its legacy default fixed-edge
   path. Keep semantic labels, capacity cuts and their score ownership separate.
4. Keep G's approved common/descendant loss weights unchanged. The class-balanced
   noisy codec loss here does NOT silently become a new G arm objective.
5. Carry exact length/vector/partial/zero/complete-structure metrics and the new
   oracle/inferred reconstruction baseline into preflight reporting. Document
   the changed byte ceiling and latent scale before comparing arms.
6. Repeat the original finite full-model gradient/update, frozen teacher/rho,
   fail-closed optimizer, leakage, packing/reactivation and resumability gates
   with these weights. Keep S=1/R=0 and the same-input predictor/committed replay;
   this is not S2 or S4.
7. Preserve the post-teacher structural-necessity gate before argmax/sample;
   codec accuracy does not establish useful downstream structural dependence.

This task stops here: no G smoke arm/campaign, H/I training, natural-language
training or S4 ran. No G scientific benefit/negative label, merge or push.

## Evidence and reproduction

- Diagnosis/probe/candidate source: `c80920e` (full immutable SHA in JSON).
- Post-selection auditor: `c59581cdb749e5d4c1a90af152fe07286cdce220`.
- Per-site auditor: `26f2d0f707dcfcf5e91726b1611b05e4f2b34dce`.
- Raw per-length/class/site results: `diagnosis.json`, `heldout.json`,
  `postselection_audit.json`, `per_site_class_audit.json`.
- Selection and checkpoint identities: `selection.json`, `frozen_codec.json`.

Commands at the recorded source versions, in a fresh isolated output directory:

```sh
PYTHONPATH=src python scripts/run_phase_g01.py --stage diagnose
PYTHONPATH=src python scripts/run_phase_g01.py --stage train
PYTHONPATH=src python scripts/audit_phase_g01.py
PYTHONPATH=src python scripts/audit_phase_g01_sites.py
```

Existing scientific outputs refuse overwrite; selection blocks subsequent
training/reselection. Inputs remain target-latent-conditional, not prompt-only
generation or full256-byte codec coverage.
