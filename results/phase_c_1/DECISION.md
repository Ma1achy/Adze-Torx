# Phase C.1 / D0 decision

**PHASE_C_1_PASS**

Prompt and target uses of the shared frontend retain one mean parameter set but
now derive distinct, deterministic, reproducible occurrence keys. Tied DiT
blocks and modulation heads retain one mean set while all three Q cycles derive
distinct keys.

With stochasticity enabled and `lambda_op=0`, full forward, Phase-B losses, and
raw mean gradients remain exact against Adze-D; rho gradients remain exactly
zero; key/rho invariance and public-factor invocation coverage pass.

`TORX_PARITY_PASS` remains accepted. Phase D was not started.
