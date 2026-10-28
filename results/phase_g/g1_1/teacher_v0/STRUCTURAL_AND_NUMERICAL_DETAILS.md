# Endpoint100 — structural readout and numerical details

These are auxiliary predictions from the neutral predictor, NOT measurements of
learned structural commitment. Replay used explicit teacher b/ell throughout.
Source:validation_100_A.json; no additional model execution or fitted readout.

| Predictor metric | Endpoint100 |
|---|---:|
| Boundary accuracy (31 nonterminal edges) | 99.922186% |
| Boundary F1 (pooled counts) | 99.755664% |
| Exact predicted boundary vector (terminal forced) | 97.587719% |
| Per-carrier predicted extent accuracy | 99.897206% |
| Exact predicted extent vector | 96.710526% |
| Exact predicted total emitted length | 96.710526% |
| Complete predicted b/ell structure | 94.298246% |
| Final partial extent | 642/672 =95.535714% |

| True extent class | Support | Correct | Recall |
|---:|---:|---:|---:|
| 0 | 20640 | 20640 | 100% |
| 1 | 224 | 224 | 100% |
| 2 | 224 | 223 | 99.553571% |
| 3 | 224 | 195 | 87.053571% |
| 4 | 7872 | 7872 | 100% |

The lower class3 recall must not be hidden behind many inactive sites. These
heads differ from the accepted frozen codec's target_l readout. No generated
argmax/sample replay was evaluated or trained, and these metrics do not establish
its validity, capacity behavior or useful downstream effect.

Teacher committed lengths match truth by construction:missing/extra bytes0,
actual emitted-length accuracy100%. Global length-aware reconstruction loss
equals overlap-only conventional byte NLL here:0.420410961. Replay latent MSE
is0.289025486. This equality does not redefine length-penalized loss as ordinary
NLL for a future mismatched-length arm.

Nonquery guard uses prospective equal-example means outside the query carrier:
accuracy91.371542%, NLL0.332729131 at100. Separate valid-byte-weighted values are
90.539337% and0.355376306; complementary means excluding only the3 query payload
bytes are91.676307% and0.320786595. None substitutes for the declared guard.

The recorded structural_churn0.862664461 is neutral→teacher changed-site fraction
for the two-application replay, not temporal churn of a learned policy.
Committed structure is constant for a fixed example under this teacher control.
Queries have full extent4; partial-query support is0, not a quality score.

First raw full-model gradient norm2201.717285156; official masked raw norm range
10.172092438..2201.673583984, clipped at1. All parameter/Adam leaves remain finite;
all frozen leaves and frozen moments are bitwise identical in all101 states.
The output decoder is intentionally trainable; the external codec/target encoder
and every rho leaf are not. Ordinary AdamW decay may change enabled unused
means even where the data gradient is0; this is the unchanged accepted optimizer.

No sampled action or score-estimator ownership changed. Historical exact/MC
score oracles are preserved, not claimed as newly trained sampled-arm evidence.
