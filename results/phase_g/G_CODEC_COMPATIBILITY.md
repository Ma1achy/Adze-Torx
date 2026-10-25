# Phase G codec compatibility

Status: pending resource-coordinated runtime gate.

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
