# Phase F.1 — DENOISE_V1 calibration

Status: **DENOISE_V1_CALIBRATION_PASS**.

DENOISE_V1 changes only the target distribution from the blocked DENOISE_V0:
all eight target bytes are independently uniform on `1..32`, matching the
accepted frozen target-codec domain. The ordinary 256-way decoder is unchanged
and its logits are not masked. Byte chance is 3.125%; exact-sequence chance is
`(1/32)^8 = 9.094947e-13`.

The 4,096-example codec gate passed with 99.4568% byte reconstruction and
95.7275% exact reconstruction. Clean latents were finite and noncollapsed, with
global RMS 0.671449, mean coordinate variance 0.0102975, and no duplicate latent
states.

The real `lambda_op=1` first-gradient gate passed. The raw global gradient norm
was `1.4641897472e11`, the permitted norm was `1.46418548736e11`, and clipping
produced applied norm 1.0. DiT QKVO, DiT FFN, conditioning, output heads,
decoder, and direct SSM gradients were finite and nonzero. Proposal gradient
was exactly zero as expected from the benchmark-local `proposal_weight=0`;
raw rho gradient was nonzero and applied rho update was exactly zero. The frozen
target teacher was bitwise unchanged.

Fixed-corruption overfit at `nu=0.5` passed all gates:

| examples | step | byte accuracy | exact accuracy | byte NLL |
|---:|---:|---:|---:|---:|
| 1 | 25 | 100.0000% | 100.0000% | 0.002957 |
| 8 | 100 | 100.0000% | 100.0000% | 0.000183 |
| 256 | 250 | 97.3145% | 80.4688% | 0.112626 |

F_ONE_STEP_V1 trained for 5,000 steps on 16,384 examples with
`nu ~ Uniform(0.025, 0.9)`, `L=4`, `Q=3`, `S=1`, `R=0`, and Torx operator
stochasticity enabled at `sigma_op=1e-3`. Validation uses 512 examples per
level, with one fixed epsilon direction per logical example shared across all
corruption levels and model checkpoints.

Final `lambda_op=0`, Q=3 calibration metrics:

| nu | h0 MSE | byte NLL | byte accuracy | exact accuracy | boundary loss | extent loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 diagnostic | 0.017827 | 0.000214 | 100.0000% | 100.0000% | 0.000019 | 0.000039 |
| 0.10 | 0.016936 | 0.013032 | 99.7070% | 97.6562% | 0.000024 | 0.000043 |
| 0.25 | 0.020173 | 0.673138 | 78.4668% | 13.0859% | 0.000035 | 0.000054 |
| 0.50 | 0.030384 | 2.524078 | 27.5879% | 0.0000% | 0.000045 | 0.000066 |
| 0.75 | 0.040172 | 3.299378 | 9.4971% | 0.0000% | 0.000034 | 0.000077 |
| 0.90 | 0.045908 | 3.460845 | 5.0293% | 0.0000% | 0.000027 | 0.000111 |

This is a learnable, non-ceiling denoising range with a strong corruption
gradient. Training stopped at the predeclared 5k endpoint: the benchmark gate
was resolved, while changes from 2k to 5k in the hard buckets were modest.

The trained zero-physical-block / DiT-shell diagnostic remains distinct from
the full Q=3 denoiser. At `nu=0.25`, Q0 versus Q3 byte accuracy was 73.5840%
versus 78.4668%, and h0 MSE was 0.170683 versus 0.020173. At `nu=0.75`, byte
accuracy was similar (9.2773% versus 9.4971%), while h0 MSE was 0.405036 versus
0.040172. This baseline is diagnostic only and is not a complete DiT bypass.

The finite operator-noise sanity check used four roots on 64 examples per
level. All nonfinite rates were zero and results closely tracked `lambda_op=0`;
at `nu=0.25`, lambda-zero byte accuracy was 81.2500% and the four lambda-one
roots ranged from 81.0547% to 81.4453%.

No S>1 execution, re-corruption rollout, eta experiment, or Phase-F S decision
was performed.

Repository validation passed: formatting (120 files), lint, type checking,
dependency boundaries, 156 regular tests, 9 slow tests, all 9 focused F1 tests,
and an explicit private-Torx scan with zero matches.
