# Corrected Phase B decision

Decision: **ADZE_D_CORE_PASS**.

Phase B.1 terminology: **ADZE_D_CORE_TRAINABILITY_PASS** remains the substantive
decision; deterministic correctness hardening is recorded separately as
`PHASE_B_1_PASS` in `results/phase_b_1/DECISION.md`.

- `B0_INTERFACE_PASS`
- `B1_DIT_PASS`
- `B2_MODEL_PASS`
- single-example memorization passed;
- 16-example overfit passed;
- held-out COPY passed;
- held-out REVERSE passed;
- no Phase C or Torx stochastic implementation was started.

The model contains 2,268,245 parameters for `adze_reference_small_v0`, of
which 1,126,528 belong to the four-physical-block looped DiT parameter tree.
