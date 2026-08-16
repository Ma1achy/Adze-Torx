# Phase E — Q-recurrence science

Decision: **PHASE_E_RECURRENCE_NEUTRAL** (primary seed 0).

## E0 control audit

| Configuration | Parameters | Physical blocks | Q | Applications |
|---|---:|---:|---:|---:|
| E_REF (also E_REF_DEPTHCOND) | 2,287,146 | 4 | 3 | 12 |
| E_Q1 (also E_PARAMMATCH_Q1) | 2,287,146 | 4 | 1 | 4 |
| E_UNSHARED12 | 4,415,018 | 12 | 1 | 12 |
| E_REF_NODEPTHCOND | 2,287,146 | 4 | 3 | 12 |

E0 passed: E_REF/E_Q1 parameter mismatch is exactly zero, the unshared control has twelve block entries, the depth-condition toggle exists, and all four lambda-zero smoke forwards were finite.

## Training and diagnostics

The Phase-E runner, controls, corrected stable-example Monte Carlo evaluation path, cycle truncation, delta suppression, and depth-code interventions are implemented.

## E1 COPY smoke

All four distinct configurations reached step 500 with finite lambda-zero and lambda-one evaluation, finite gradients, and zero applied rho gradient. Final 32-root lambda-one byte accuracy was E_REF `0.192978`, E_Q1 `0.200897`, E_UNSHARED12 `0.202362`, and E_REF_NODEPTHCOND `0.200745`. These are smoke-test observations only, not comparative scientific evidence.

## E2/E3 final 20k primary matrix

Values are corrected 32-root lambda-one MC byte/exact-sequence means. Lambda-zero values were evaluated at the same final checkpoint.

| Task | Config | Lambda 0 byte | Lambda 1 byte | Lambda 1 exact |
|---|---|---:|---:|---:|
| COPY | E_REF | 0.995117 | 0.995316 | 0.962524 |
| COPY | E_Q1 / E_PARAMMATCH_Q1 | 0.998047 | 0.997833 | 0.982666 |
| COPY | E_UNSHARED12 | 0.996582 | 0.996582 | 0.972656 |
| COPY | E_REF_NODEPTHCOND | 0.997070 | 0.997162 | 0.977295 |
| REVERSE | E_REF | 0.995117 | 0.995041 | 0.960327 |
| REVERSE | E_Q1 / E_PARAMMATCH_Q1 | 0.993652 | 0.994019 | 0.952148 |
| REVERSE | E_UNSHARED12 | 0.992188 | 0.991684 | 0.938354 |
| REVERSE | E_REF_NODEPTHCOND | 0.999023 | 0.998611 | 0.988892 |

The lambda-zero/lambda-one gap is small in every final model; finite stochastic execution did not materially alter the ranking.

## E4 causal diagnostics

Cycle truncation (lambda-one byte accuracy): COPY Q=1/2/3 = `0.993179` / `0.996536` / `0.995316`; REVERSE = `0.994461` / `0.995300` / `0.995041`. This is non-monotonic and does not establish a causal benefit for the final third cycle.

Suppressing the q=1 recurrent delta produced COPY `0.995300` versus identity `0.995316`, and REVERSE `0.995178` versus identity `0.995041`: no meaningful degradation. Depth-code interventions were similarly small; the strongest REVERSE decrease was all-q0 `0.993637` versus correct `0.995041`.

Per-cycle activations and update summaries are in `diagnostics/copy.json` and `diagnostics/reverse.json`. JVP/local-contraction and linear probes remain explicitly deferred secondary diagnostics.

## Runtime and seed coverage

All primary runs use `init_seed=0` and `stochastic_training_seed=0`; no multi-seed claim is made. E5 repeats and E_Q1_COMPUTEMATCH were not run because the primary controlled and causal evidence was neutral. Checkpoint hashes and steady-state runtime metrics are retained in each task/config summary JSON.

JVP contraction and linear probes are explicitly deferred secondary diagnostics.
