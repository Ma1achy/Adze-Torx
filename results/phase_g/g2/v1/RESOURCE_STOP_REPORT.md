# G23 V1 — durable resource stop, not scientific closure

G21 adjudication: **G21_GRADIENT_ESTIMATOR_CONFIRMED**. Original G23 V0 remains
**PHASE_G_BLOCKED**, permanently unchanged. The new V1 continuation is
**PHASE_G_UNRESOLVED** because the prescribed multi-seed training/evaluation is
incomplete. This is not a negative generated-structure result or a new estimator
failure. No final Phase G scientific closure or main integration is claimed.

## Stop and exact resumption

The reduced-priority official campaign exited 1 on 2026-10-27 at 17:06:05.
The unchanged guard at `scripts/run_phase_g23_v1.py:649-650` raised
`critical disk capacity; exact durable state preserved` BEFORE optimizer update
138 of seed1/B. Its frozen predicate is available disk `< 2 * 1024**3` bytes.
The exact byte reading was not saved; it must not be invented. A post-exit `df`
read 2.5 GiB available, later approximately 5 GiB: recovery after process exit
does not erase the observed failure or provide sustained capacity for this run.
No unrelated data/process was removed or signaled. No automatic restart was made
against oscillating disk headroom. Further heavy work is blocked on sufficient
disk capacity; normal CPU contention/swap alone is NOT the blocker.

| Official arm | Completed updates | Status |
|---|---:|---|
| Seed0 teacher A | 1000 | Initial cap complete |
| Seed0 argmax B | 1000 | Initial cap complete |
| Seed0 sampled C | 1000 | Initial cap complete |
| Seed1 teacher A | 1000 | Initial cap complete |
| Seed1 argmax B | 137 | Durable resource stop; next update138 |
| Seed1 sampled C; seed2 A/B/C | 0 | Not started |

Total **4137** new counted optimizer updates; no prospective trial or inherited
teacher100 update is counted. Four completed arms have lossless JSONL curves.
All five receipt chains (4142 receipts INCLUDING five uncounted initialization
records), all milestone payloads, both ring slots per arm and derived LATEST
pointers verify. Every counted receipt records finite metrics and unchanged
frozen-state fingerprint. The payload contains params, Adam moments, step,
campaign identity, seed and arm; checksum/serialized-field inspection does NOT
constitute a new numerical tensor restoration or optimizer trial.

Completed arms retain full runtime-segment records. Seed1/B did not reach the
normal-return segment writer: its partial segment wall-time record is missing.
Per-update durations/peak RSS, launch records, complete stdout and failure time
are preserved instead; no unmeasured arm wall time is inferred from them.

Current seed1/B full-state SHA-256:
`f82269e87457c71f92117102ed6a7d0eb76a827476b9532f18f9dc32dda9af0d`.
An additional immutable, fsynced copy is recorded in
`resource_stop_20261027T170605/RESUME_SNAPSHOT.json`; the original journal was not
modified. Raw completed receipts, failure, stdout and all endpoint identities
are in that directory. Binary snapshots remain in ignored `results/runs/`, not
in Git; retain these local artifacts. Ring slots are transient after future
resumption; the separately archived update137 copy preserves this exact stop.

Once capacity is genuinely available, resume the SAME frozen command:

```sh
nice -n 10 env PYTHONPATH=src:scripts python scripts/run_phase_g23_v1.py --stage campaign
```

It must reuse existing G2/extension decisions, skip completed optimizer budgets,
and continue seed1/B from update138 with its existing Adam moments. No freeze,
fresh initialization, alternate seed, changed cap or new loss is authorized by
this report. All remaining scientific gates and protected-test restrictions
remain in force. After all nine arms/validation-only selections are durable,
and only if every selection is eligible and source/state identity verifies,
run the frozen `--stage science`, then `--stage close` and full validation.

## G21 result and prospective V1 identity

See `../../g2_1/G21_RESULTS.md` and the immutable G21 manifest. The old selected
peaked extent-logit coordinate remains **4.450393401 empirical SE**, above4.
Independent finite-distribution audit, all42 affected coordinates, 64 unused
roots x4096 draws per fixture and the prospectively frozen simultaneous
Bernstein procedure passed. Known-correct original-procedure calibration
rejected140/2048 complete-family experiments (6.83594%). Evidence supports an
expected statistical false rejection under sparse, non-Gaussian studentization
and multiple comparisons, not demonstrated estimator bias or a variance-formula
coding defect. The selected confirmation interval includes zero; no claim of
precise sign or relative accuracy for that near-zero coordinate is made.

No estimator, baseline, task, architecture, loss, scientific effect threshold,
optimizer or sampling policy was changed. V1 replaces only the adjudicated
oracle prerequisite, with separate protocol/evidence/state paths. V1 source and
protocol were committed at `ddb0c5e`; identity at
`9b42d57f208f598e7e4c929d424250847ae6a9b1` before full-model preflight.
Preflight passed and was committed at `3c594f5` BEFORE counted training.
The bound execution identity is
`5cb5e7363b14239b4c65a7dda1fb70b569e0b92a935e1f2c67311496a15f1d99`.
New pause-audit/export utilities are unimported by the model/runner; bound source
and protocols still match exactly. Evidence-only Git commits did not change
the launched implementation.

## Interim validation — NOT protected-test findings

Every row uses912 validation examples; C sampled is the preregistered four-root
mean of metrics, not an ensemble decoder. Query accuracy is over the three
correlated payload bytes; all bootstrap resampling uses complete examples.

| Seed/arm at1000 | Query accuracy | Query overlap NLL | Matched-cut accuracy | Emitted-length accuracy |
|---|---:|---:|---:|---:|
| 0/A teacher | 99.525% | .012282 | 3.253% | 100% |
| 0/B argmax | 99.452% | .019840 | 2.997% | 100% |
| 0/C sampled | 99.050% | .037512 | 3.600% | 99.753% |
| 1/A teacher | 99.306% | .029892 | 3.253% | 100% |

Seed0 B-argmax, C-argmax AND C-sampled pass every frozen G2 functionality check,
including structure, validity/reachability, length, query/cut paired uncertainty
and both nonquery guards. G2 selected update1000 for each arm using validation
only. B versus cut accuracy difference is+96.455pp (95% interval95.760–97.113pp);
C sampled versus cut is+95.450pp (94.682–96.181pp). Large cut NLLs reflect
confident wrong predictions under the intervention, not optimal conditional
chance NLL. These results permit G3, not a final multi-seed functionality claim.

The frozen extension predicate returned cap1000, NOT2000: frontier=true but
material=false. Average B-argmax/C-sampled query accuracy moved98.9583% to
99.2507% from500 to1000 (+.2924pp, below required1pp), despite loss reduction
.0435371 to.0286756. No extension or threshold adjustment occurred.

G2 selected payloads (not final G3 checkpoint selection):

- A1000: `6a82112b480840ffe2f759fd5180d8e733c6ee02e7cd89543e310babbde9b9d0`
- B1000: `ff344d72489c0923cc014e40b2d5b647f04c8c6e4af928713019838da07a6446`
- C1000: `b4896a4d44bc74928c68824bd2be2c52f726eac1f0675a14f77b54d123a7abc8`

Full-model preflight raw gradient norms: A75.2228, B75.2228, C102.5066;
C connected-versus-detached score-gradient difference57.070751. All prospective
AdamW states were finite and discarded; teacher/rho preservation and actual
invalid-step no-update checks passed. The fixed16-root initial C gradient
variance diagnostic is retained separately. Clipping/Adam are unchanged;
oracle correctness is not a claim that clipped updates are unbiased.

The final three scientific axes remain unresolved pending their prescribed
evidence. No fixed-neutral/shifted/neutral-discordant or protected-test result
is available; teacher/cut validation is NOT evidence of superiority to neutral
routing or of sampling-training value. Seeds1/2 remain incomplete; no final
G3 selection exists, and no new protected-test shard exists. Do not select
checkpoints, stop seeds or infer comparative equivalence from this partial run.

## Validation and preservation ledger

- Earlier G21/structural focused run:30 passed. V1 source conformance:3 passed.
  Initial receipt-export regression:5 passed (not full-model proof).
- Post-stop pure-I/O regression:11 passed (six pause checks plus the same five
  export checks; not11 new full-model tests), latest0.46s. Tip/previous-slot,
  pointer, receipt corruption and gaps are rejected. No model import/execution.
- `PYTHONPATH=src:scripts make format-check lint typecheck boundaries`: PASS;
  180 files formatted, lint passed, zero type errors, public boundaries passed.
  Initial invocation without the scripts search path failed with nine missing
  script-import resolutions and one new fixture's possibly-unbound variable.
  The fixture initialization was corrected; the prescribed source search path
  resolves the imports. Raw failed and successful logs are BOTH preserved;
  no error suppression or previous test weakening was used.
- Full private-Torx AST import scan:180 Python files, zero private imports;
  boundary scan also passes. The earlier128-file src/scripts/tests-only scan is
  retained but is not substituted for the complete scan including experiments.
- Full regular, slow and post-integration suites NOT rerun: real disk failure
  halted dependent heavy work; no complete scientific closure/main merge.
  Earlier full-model preflight remains evidence, not a substitute claim that
  those outstanding suites ran. Resume/test gates remain outstanding.
- All731 baseline members, G1(13), G11(22), teacher100(270), G23 V0(37),
  G21(254) manifest members and42 G21-bound originals verify (overlapping
  counts, NOT a summed unique total). See `PRESERVATION_CHECK.json`.
- Accepted F1/F3/G0/G01/G11 states/evidence remain unchanged. Main is clean at
  `2a0cab0c9c647832596013d56037c842ff3844ea`; H/I branches are clean at their
  recorded SHAs. Thermo worktree was not accessed. No S4/H/I/LM work occurred.

Validated engineering/evidence is committed only on isolated G. No main merge
or push is appropriate while multi-seed science and required validation remain
incomplete. Phase H cannot advance through the G integration prerequisite yet.
This is a documented resource blocker, not a finished Phase G milestone.
