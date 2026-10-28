# Phase G codec compatibility

Status: `G_CODEC_TASK_COMPATIBILITY_BLOCKED` at implementation
`1a96675cab9bdc2d7c9c377b4f732d03c3124f1d`.

The gate covers lengths 8..64 and requires aggregate byte accuracy at least
99%, each length at least 98%, actual task structural signature at least 99%,
and no capacity or numerical failure.

Clean reconstruction and task signatures are evaluated with the exact
deterministic CODEC artifact, never the subsequently trained F1 output decoder.
Before corruption, the deterministic codec's clean target-analysis path is
compared with the F1 public-Torx zero-noise mean path for frontend, h0,
structural logits, and teacher structure. Any mismatch blocks integration. The
trainable model output decoder is deliberately outside this frozen-teacher
parity comparison.

Because the predictor observes corrupted carrier state rather than clean
decoded bytes, the gate also evaluates an accepted-F1 denoise/decoder witness
at `nu=.025,.10,.25`. For eligible content-dependent cuts, the Wilson 95% lower
bound must exceed chance at every noise level. The direct untrained boundary
head is reported diagnostically. The task is conditional on a noisy observation
of the clean teacher latent.

Failure label: `G_CODEC_TASK_COMPATIBILITY_BLOCKED`.

## Result

The correctly wired gate failed the frozen practical thresholds:

- aggregate clean byte accuracy: 69.74% (required at least 99%);
- minimum per-length accuracy: 51.02% (required at least 98%);
- aggregate task-signature accuracy: 92.40% (required at least 99%);
- numerical failures: zero;
- capacity failures: zero.

The deterministic CODEC and F1 Torx-mean frozen teacher paths matched exactly
for frontend, h0, boundary/extent logits, and teacher structure. Thus the block
is not an F1 integration mismatch. The noisy witness remained measurably above
chance, with content-cut accuracy/Wilson-95 lower bounds of 76.16%/74.31%,
72.73%/70.80%, and 64.17%/62.11% at nu .025, .10, and .25 respectively, but
that diagnostic cannot waive the clean codec thresholds.

Historical attempts 1--3 are retained in `codec_attempts.jsonl`; each was
superseded for a documented audit-wiring implementation bug. Attempt 4 is the
first valid gate result. Per protocol, engineering preflight and all smoke arms
were not started.
