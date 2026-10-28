# G.0.1 prospective length investigation and remediation V1

Frozen before probe/candidate execution. Parent is G.0 codec SHA-256
`0f4a54e221729b14c6e75d30bed84bd566182a490f975c76d574e540f936adab`.
G.0/F1/F3 and the earlier compatibility record remain unchanged.

## Data and input stages

Fresh byte-domain 1..32 targets, lengths 8..64, static width64/C32/Lmax4.
Probe train:128 examples/length seed1010; validation:32/length seed1011.
Candidate train:256/length seed1012. Protected test:64/length seed1013;
test is not exposed until validation-only selection. Report every length and
partial class. Check exact example disjointness and hash ordered splits.

Stages: clean h0; h_nu at .025/.10/.25 using the frozen F corruption contract;
and actual G predictor explicit carrier input. The latter is expected to be
BITWISE IDENTICAL to h_nu, not a third independent signal. Verify that through
the real sanitized model graph at each nu; then report alias-stage results
explicitly rather than manufacture independent evidence. Observed structure,
prompt and routing are target-independent neutral constants. Actual learned
G head baseline uses a fixed four-example-per-length validation subset, not
teacher routing. Report its limited sample size. Corpus/noise roots are
explicit JAX keys; each split uses independent noise identities.

## Frozen probes and controls

One production-compatible local linear extent readout (h only), plus global
linear heads on flattened h0 for total length, emitting count and partial
class. One small-MLP instrument:64 hidden units/GELU for local/global heads;
local diagnostic MLP also sees constant carrier identity. Never install this
MLP in Adze. Standardization fits TRAIN ONLY. Fixed800 AdamW updates per
probe, batch128, lr1e-3/wd.01/clip1. Local loss class-balanced extent CE;
global total/count/partial CEs weights one. Seeds1015 plus stage/kind index.
No probe hyperparameter tuning or validation checkpoint search.

Controls: training-only global majority, training-only per-position majority
(length prior), existing codec linear structural heads, existing actual G
heads, and explicitly labelled teacher-structure upper bound. True lengths,
target masks, teacher b/ell and target-dependent metadata are not probe or
predicted-length decoder inputs. They are fitting labels and metric support
only. Oracle reconstruction is reported separately from inferred-length
reconstruction, with global alignment/missing/extra penalties.

## Frozen diagnostic decision and candidate training

Readout-only is permitted only if the production-compatible local LINEAR
probe passes the clean structural criteria below and the noisy diagnostic
criteria at all three nu. A diagnostic MLP/global-head pass alone does not
justify changing the architecture or installing those probes. Otherwise
jointly shape the faithful encoder/linear structural heads/Mamba decoder.
Finite probe failure does not prove information-theoretic loss.

Candidate starts from an exact COPY of frozen G.0; fresh AdamW, lr3e-4,
wd.01/clip1/batch32. Loss = clean byte CE + clean semantic-boundary CE +
class-balanced clean extent CE + class-balanced noisy extent CE, weights1.
Noisy h uses fresh Gaussian epsilon and nu uniform(.025,.25). Readout-only
updates target_b/target_l only; joint updates only the original target-codec
subtree. No other model/teacher/rho/optimizer state is modified.

Maximum5000 updates, validation at 0/100/250/500/1000/2000/5000.
Select FIRST scheduled validation checkpoint satisfying every clean candidate
criterion AND all three noisy-input readiness criteria, then evaluate protected
test once. Freeze only if both sets pass on test. No test-based reselection.
Stop nonfinite updates before persistence; retain prior durable checkpoint.

## Candidate acceptance, declared before training

- Retain known-extent byte accuracy>=99%, EVERY length>=98%, historical
  field-signature>=99%, zero numerical/capacity failures.
- Exact total length>=99%; every length bucket exact-total>=95%.
- Per-carrier extent>=99.5%; exact extent vector>=99%.
- Partial extent>=99%, each partial class>=98%.
- Zero recall and nonzero recall both>=99%.
- EXACT complete structure (b and ell, not field average)>=99%.
- Inferred-length globally aligned byte accuracy>=99%, every length>=98%.
- Report oracle/inferred exact-sequence accuracy, NLL, missing/extra counts and
  length-aware reconstruction loss; never call length-penalized loss NLL.

Additional G-input readiness diagnostic at EACH nu: exact total/vector/partial
>=95%, each partial class>=95%, per-site>=99%, zero/nonzero recalls>=99%.
Failure of this diagnostic blocks G-smoke readiness, even if clean codec
acceptance passes. It is not evidence against generated structure or S.

## Boundaries and stop

Semantic boundaries vs capacity-forced cuts, categorical action law and local
score bridge are UNCHANGED. Preserve exact/MC oracles; rerun focused conformance
checks but no new sampling distribution is introduced. Freeze/hash successful
G.0.1 candidate separately and prepare matched controls only after acceptance.
STOP at diagnosis, remediation, held-out compatibility and readiness report.
No G smoke arm/campaign, H/I training, LM training or S4.
