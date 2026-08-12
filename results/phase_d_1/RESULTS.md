# Phase D.1 — corrected MC evaluation and D0 gradient hardening

Decision: **PHASE_D_1_PASS**.

## Evaluation correction

Every validation sample now receives a stable integer `global_example_id`, separate from static scope/module identities. Evaluation runs each logical sample as a B=1 execution under `vmap`, then aggregates objective numerators and denominators using the original loss normalization. Training was not changed: its factor calls already sample batch-shaped epsilon tensors under a fresh optimizer-step key.

## Corrected final evidence

- COPY stochastic-training seed D2: lambda-zero `0.962891`, lambda-one `0.963577` (SD `0.001022`, 95% CI `[0.963209, 0.963946]`).
- REVERSE stochastic-training seed D2: lambda-zero `0.965820`, lambda-one `0.965469` (SD `0.000922`, 95% CI `[0.965137, 0.965802]`).
- COPY stochastic-training seed 0: lambda-zero `0.954590`, lambda-one `0.954605` (SD `0.001180`, 95% CI `[0.954180, 0.955030]`).
- COPY stochastic-training seed 1: lambda-zero `0.970215`, lambda-one `0.969574` (SD `0.000845`, 95% CI `[0.969269, 0.969879]`).
- COPY stochastic-training seed 2: lambda-zero `0.946289`, lambda-one `0.946457` (SD `0.001167`, 95% CI `[0.946036, 0.946878]`).
- REVERSE stochastic-training seed 0: lambda-zero `0.935547`, lambda-one `0.934906` (SD `0.001166`, 95% CI `[0.934486, 0.935327]`).
- REVERSE stochastic-training seed 1: lambda-zero `0.951660`, lambda-one `0.950638` (SD `0.001236`, 95% CI `[0.950192, 0.951084]`).
- REVERSE stochastic-training seed 2: lambda-zero `0.949707`, lambda-one `0.947540` (SD `0.001240`, 95% CI `[0.947093, 0.947987]`).

D0 fixed-key parity and MC expected-gradient evidence cover affine, categorical-logit, embedding, and depthwise-convolution factors. New records correctly call D3 trials independent stochastic-training seeds; their non-codec initialization is fixed at seed 700.
