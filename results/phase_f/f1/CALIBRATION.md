# Phase F.1 — DENOISE_V0 calibration

Status: **PHASE_F_1_ONE_STEP_BLOCKED**.

The explicit corrupted-carrier path, qualified parity, target-leakage checks,
fixed-structure content invariance, validation-noise pairing, and training-key
occurrence semantics passed their focused regressions.

Calibration did not start because the mandatory frozen-codec suitability gate
failed. On 4,096 examples whose eight target bytes were uniform over
`0..255`, the accepted `target_codec_b1.pkl` reconstructed 12.4512% of bytes
and 0% exact sequences. The same checkpoint still reproduces its historical
accepted-domain result on values `1..32`: 99.4141% byte accuracy and 95.3125%
exact accuracy. This isolates the blocker as codec domain coverage rather than
an F1 wiring regression.

The uniform-target latents were distinct but highly clustered: global `h0`
RMS was 0.664845, mean coordinate variance was 0.00285248, and mean pairwise
cosine similarity was 0.993610. No normalization or corruption-contract change
was made.

No first-gradient gate, overfit run, one-step calibration, operator-noise
sanity evaluation, or `S>1` experiment was run. Consequently there is no
evidence for or against denoising depth S.

Repository validation passed with 155 regular tests and 9 slow tests. Formatting,
lint, type checking, dependency boundaries, and the private-Torx import scan
also passed.
