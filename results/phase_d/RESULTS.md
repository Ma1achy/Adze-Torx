# Phase D — finite stochastic operator execution and trainability

## D0 — finite-noise contract

Decision: **D0_FINITE_NOISE_CONTRACT_PASS**.

- All four 4,096-sample factor oracles passed five-standard-error mean and unbiased
  variance bounds. Observed variances were `9.9291e-7` (affine), `1.0162e-6`
  (categorical logits), `9.9166e-7` (embedding), and `1.0288e-6` (depthwise
  convolution), against analytic variance `1e-6`.
- Fixed-key input/weight/bias gradients matched the explicit identical-epsilon JAX
  reference exactly; the maximum rho difference was `5.82e-11`.
- Every coordinate of the 4,096-key pathwise-gradient mean passed its five-SEM plus
  numerical-floor bound. Exact lambda-zero outputs and zero rho gradients passed.
- Fixed-epsilon scaling passed for every factor at lambdas `0.1`, `0.25`, `0.5`, and
  `1.0`. Sigma initialization and both clamps passed.
- Q-occurrence variances were `1.0151e-6`, `1.0273e-6`, and `1.0453e-6`, confirming
  equal local variance without Q/effective-depth normalization.
- Prompt/target frontend occurrences and all block-0 Q occurrences shared means while
  producing distinct, reproducible finite-noise residuals. Observed scope/module IDs
  were collision-free.

## D1 — zero-shot portability

Decision: **D1_PORTABILITY_PASS**. Evaluations used paired nested root sets, 32-example
chunks, validation seeds 821/831, and root bases 4100/4200.

| Task | Lambda-1 byte accuracy, mean (95% CI) | Loss, mean (95% CI) | Exact sequence, mean (95% CI) | Nonfinite |
| --- | --- | --- | --- | --- |
| COPY | `0.958618` (`0.958346`, `0.958890`) | `0.140500` (`0.140247`, `0.140753`) | `0.707397` (`0.705426`, `0.709368`) | `0` |
| REVERSE | `0.959305` (`0.959013`, `0.959597`) | `0.137804` (`0.137526`, `0.138082`) | `0.729126` (`0.727213`, `0.731039`) | `0` |

The task JSON and root-distribution files contain the complete lambda `0`, `0.1`,
`0.25`, `0.5`, and `1.0` statistics and stage-by-stage signal/perturbation RMS for
frontend, proposal, pack, every physical block and Q cycle, unpool, `h_hat`, carrier,
and decoder logits.

## D2 — stochastic continuation

Decision: **D2_STOCHASTIC_CONTINUATION_PASS**. Both tasks passed the 32-root candidate
gate at step 100.

| Task | Byte accuracy, mean (95% CI) | Loss, mean (95% CI) | Exact sequence, mean (95% CI) | Working-checkpoint SHA-256 |
| --- | --- | --- | --- | --- |
| COPY | `0.963425` (`0.963191`, `0.963659`) | `0.137685` (`0.137429`, `0.137941`) | `0.730835` (`0.728964`, `0.732706`) | `f32514ad2c9b595d55942519565be0146d1acaaab9f2779e7d179b1ff2f2df55` |
| REVERSE | `0.965698` (`0.965364`, `0.966033`) | `0.129348` (`0.128971`, `0.129724`) | `0.776855` (`0.774436`, `0.779275`) | `fe5e29188fe4fb03150d6f1d8a491735d0376237956c4f33eac3494e3bd4278b` |

Checkpoint mapping, rho parameters, rho moments, and clean-teacher freezing were
bitwise verified. Permitted means changed, raw rho connectivity was nonzero, applied
rho gradients were zero, and all metrics were finite.

## D3 — stochastic scratch trainability

Decision: **D3_STOCHASTIC_SCRATCH_PASS**. Every run used the accepted 65,536/256
corpus, fixed task seeds/order, batch size 32, and the clarified B3 initialization:
accepted `target_codec_b1.pkl` leaves plus fresh seed-700 non-codec generative means,
then deterministic-to-Torx mapping and fixed sigma `1e-3`. No task-trained generative
means were loaded. All six runs passed at step 5,000 and exceeded baseline by more
than 20 points at lambda zero and lambda one.

| Task / training seed | Lambda 0 byte / exact | Lambda 1 byte, mean (95% CI) | Lambda 1 exact, mean (95% CI) | Lambda 1 loss, mean (95% CI) |
| --- | --- | --- | --- | --- |
| COPY / 0 | `0.954590` / `0.683594` | `0.954910` (`0.954511`, `0.955309`) | `0.686523` (`0.683639`, `0.689407`) | `0.152695` (`0.152328`, `0.153062`) |
| REVERSE / 0 | `0.935547` / `0.566406` | `0.934708` (`0.934169`, `0.935247`) | `0.562256` (`0.558644`, `0.565868`) | `0.209455` (`0.209067`, `0.209843`) |
| COPY / 1 | `0.970215` / `0.777344` | `0.969406` (`0.969073`, `0.969740`) | `0.774780` (`0.772418`, `0.777143`) | `0.107156` (`0.106795`, `0.107517`) |
| REVERSE / 1 | `0.951660` / `0.664062` | `0.950348` (`0.949851`, `0.950845`) | `0.657593` (`0.654740`, `0.660446`) | `0.173231` (`0.172877`, `0.173586`) |
| COPY / 2 | `0.946289` / `0.648438` | `0.946304` (`0.945896`, `0.946713`) | `0.646851` (`0.644679`, `0.649022`) | `0.184026` (`0.183670`, `0.184383`) |
| REVERSE / 2 | `0.949707` / `0.652344` | `0.947479` (`0.947103`, `0.947856`) | `0.639404` (`0.636839`, `0.641970`) | `0.184600` (`0.184239`, `0.184960`) |

All lambda-one results use 32 roots and have zero nonfinite rate. Per-root final
distributions, checkpoint-by-checkpoint JSONL curves, initialization hashes, and
working-checkpoint SHA-256 hashes are committed alongside this report; binary working
states remain ignored under `results/runs/phase_d/`.

## Verification

- `make format-check`: passed
- `make lint`: passed
- `make typecheck`: passed with 0 errors and 0 warnings
- `make test`: 104 passed, 8 deselected
- `make boundaries`: passed
- `make test-slow`: 8 passed, 104 deselected
- explicit `torx._` source/test/script scan: no matches
- regenerated C.1 evidence: all 238 gradient records passed, rho gradient maximum
  was zero, and observed scope/module IDs were collision-free
