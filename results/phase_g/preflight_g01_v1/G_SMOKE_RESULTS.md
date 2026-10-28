# Bounded G smoke — teacher completed, generated arms blocked

This is engineering evidence, not a scientific G benefit/negative result.
Source:`f571ab40659da872777f4deccbfcd05531c4c576`.
Prospective settings:[PROTOCOL.md](PROTOCOL.md).

| Arm | Training updates | Status |
| --- | ---: | --- |
| A:teacher committed |100|completed exactly once|
| B:stopped argmax |0|blocked by post-teacher necessity|
| C:temperature1 sampled |0|blocked by post-teacher necessity|

All declared arms bind the same initial mean-tree hash
`84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb`,
frozen G.0.1 codec, fresh Adam0, data and loss weights. Only Arm A was
instantiated/trained. No comparison of independently trained three arms exists.

## Teacher curve

Training entries are the corresponding checkpoint batch, not epoch averages.
Validation is the same fixed256 examples at every checkpoint.

| Update | Training loss | Training byte accuracy | Raw gradient norm | Teacher-commit validation byte accuracy | Length-aware loss |
| --- | ---: | ---: | ---: | ---: | ---: |
|0|—|—|—|14.521%|9.51552|
|10|2.87472|40.268%|19.1539|43.809%|1.92602|
|25|1.02777|77.739%|9.36359|80.037%|0.626008|
|50|0.451250|92.537%|9.95313|93.261%|0.212031|
|100|0.229369|96.071%|6.49824|95.205%|0.141290|

All scheduled0/10/25/50/100 checkpoint payloads and immutable JSON receipts
are retained. Final teacher checkpoint:
`results/runs/phase_g/preflight_g01_v1/teacher/update_100.pkl`.
SHA-256:`203926286c4f991b316e9f9d8cb8bee529861da5124f374279b82d143decd220`.
Adam absolute step100, not F1 step5100. Frozen target/rho state checks pass.

## Inference policies on the SAME teacher-trained update100 weights

These are not B/C training results. Accuracy/NLL/loss are equal-example means;
overlap byte/example support is retained explicitly in each JSON receipt.

| Commitment | Byte accuracy | Exact byte sequence | Overlap NLL | Length-aware loss | Exact total length | Complete committed structure | Missing/extra bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
|Teacher|95.205%|39.844%|0.141290|0.141290|100%|100%|0/0|
|Argmax|94.848%|37.109%|0.181210|0.192970|94.531%|88.281%|5/9|
|Sample|94.053%|30.859%|0.231809|0.281738|76.563%|56.250%|30/40|

The learned predictor (all three rows share it) has boundary accuracy 99.786%
and extent accuracy 99.805%. Teacher committed accuracy 100% is an oracle
control, not its learned prediction score. Sampled committed boundary F1 is
97.932%; argmax F1 is 99.544% (both are means of the eight batch-level F1s,
not pooled micro-F1). No invalid/overflow validation occurrence was
reported. Neutral→committed structural churn and all per-example predicted/
committed vectors are retained; this churn is not a temporal convergence metric.
Deterministic structure-root gradient variance is exactly zero by policy,
not evidence that minibatch/continuous gradients are noiseless.

## Mandatory necessity gate: FAILED

Correct must beat EACH control by>=1pp accuracy AND>=2% mean per-example
relative length-loss reduction, with both paired2000-bootstrap95% lower
endpoints positive. No controls/examples were filtered or pooled.

| Control | Correct minus control accuracy, pp (95% CI) | Mean per-example relative loss reduction (95% CI) | Pass |
| --- | --- | --- | --- |
|Shifted cuts|−0.0113 (−0.2705,+0.2384)|−5.429% (−9.660%,−1.754%)|no|
|Permuted extents|+92.077 (+91.296,+92.861)|+99.330% (+99.220%,+99.425%)|yes|
|Joint|+92.084 (+91.313,+92.882)|+99.329% (+99.230%,+99.428%)|yes|
|Fixed neutral|+69.711 (+68.185,+71.268)|+98.968% (+98.804%,+99.127%)|yes|

Correct/shifted aggregate losses are0.141290/0.138299. The reported −5.429%
gate statistic is the mean paired relative change, NOT the ratio of these
aggregate means. Shifted cuts genuinely alter packing, attention and byte
outputs but do not produce the required performance penalty.

Evidence:[necessity.json](f571ab40659d/necessity.json) retains all paired
examples, bootstrap settings, final checkpoint hash and real-model traces.
The runtime exits with `STRUCTURAL_NECESSITY_GATE_FAILED`; that is the expected
fail-closed policy outcome, not a test-suite exception to hide or retry.

No B/C training, longer teacher run, task change or threshold relaxation was
attempted. The original G preflight cannot continue without an explicitly
approved necessity-blocker resolution.
