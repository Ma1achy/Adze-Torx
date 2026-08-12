# Phase D decision

Decision: **TORX_STOCHASTIC_TRAINABILITY_PASS**.

The finite-noise contract (D0), zero-shot portability (D1), stochastic continuation
(D2), and stochastic scratch trainability (D3) gates all passed. COPY and REVERSE
passed the D3 zero-noise and 32-root finite-noise criteria for training seeds 0, 1,
and 2. No evaluated root or training checkpoint contained a nonfinite value.

The milestone remains within `PHASE_D_NOISE_POLICY_V0`: sigma is fixed at `1e-3`,
rho and its optimizer moments are frozen, direct SSM coefficients are trainable
means without rho, and `L=4,Q=3,S=1,R=0` is unchanged.
