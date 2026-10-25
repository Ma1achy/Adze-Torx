# Phase F.3 engineering checkpoint

Status: **implementation complete; runtime validation pending**.

This checkpoint contains the code, resumable runner, decision policy, and
regression-test definitions for `PHASE F.3 — SELF-GENERATED S=2 ROLLOUT
TRAINING`.

It is not scientific evidence and issues no Phase F.3 result label. No gradient
gate, optimizer update, training run, deterministic evaluation, Monte Carlo
operator evaluation, or S4 work is represented by this checkpoint. The accepted
F1 checkpoint remains the required immutable parent:

`results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl`

Required parent SHA-256:

`520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57`

## Runtime resumption order

When the machine is available:

1. Run the focused F3 tests and the complete validation suite.
2. Run the F3 provenance audit and first-gradient/gradient-conflict gate,
   including the connected-versus-detached full-model regression.
3. Run one non-persistent prospective optimizer update and the 10-update smoke.
4. Stop on any correctness, leakage, provenance, nonfinite, frozen-state, or
   gradient-pathology failure.
5. Otherwise train through 5,000 new rollout updates from absolute optimizer
   step 5,000, retaining the accepted F1 Adam moments.
6. Apply the predeclared 5k-to-10k extension rule.
7. Run full-test, diagnostic, and finite-Torx evaluations with exact protocol
   manifests on every deterministic result and stochastic shard.
8. Only then write the F3 evidence, issue exactly one scientific F3 label, and
   update the Phase F history.
