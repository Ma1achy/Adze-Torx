# FIXED_STATE_TRANSITION_V0 calibration

Calibration diagnosis: **FIXED_TRANSITION_V0_TOO_HARD**.

This is a benchmark diagnosis, not a recurrence result. No authoritative
training, stochastic MC evaluation, probes, interventions, replication, or
strong controls were run.

## Setup

- Rule: periodic 64-bit elementary cellular automaton Rule 30.
- Depths: `1, 2, 3, 4, 6, 8, 12, 16`.
- Train: 8,192 deterministic examples from seed 930.
- Validation: 512 deterministic examples from seed 931, exactly 64 per depth.
- Model seeds: initialization 0, stochastic training 0.
- Training: lambda 1, sigma `1e-3`, frozen rho, batch 32.
- Evaluation: lambda 0 at steps 100, 250, 500, 1k, 2k, and 5k.
- Chance: 50% bit, 0.390625% byte, and `2^-64` exact-state accuracy.

The model-independent audit passed before training: the full generated splits
had no duplicates or intersections, output entropy remained near maximal, the
mean state change was approximately 32/64 bits at every depth, and no sampled
fixed-point or short-cycle collapse was found.

## Step-5k overall metrics

| Arm / execution | Bit accuracy | Byte accuracy | Exact state | Byte NLL |
|---|---:|---:|---:|---:|
| T_Q1 Q1 | 49.768% | 0.293% | 0% | 6.0974 |
| T_REF Q0 | 49.966% | 0.391% | 0% | 6.0099 |
| T_REF Q1 | 49.594% | 0.317% | 0% | 6.0698 |
| T_REF Q2 | 49.600% | 0.317% | 0% | 6.1647 |
| T_REF Q3 | 49.475% | 0.269% | 0% | 6.4157 |

All logit nonfinite rates were zero. The final T_Q1/T_REF training-batch byte
accuracies were 8.98%/7.42%, while held-out performance remained at chance.
Raw gradient norms were 5.93/5.87 and were clipped to the configured norm 1.0.

## T_REF step-5k metrics by transition depth

Every exact-state accuracy below is zero.

| d | Q0 bit | Q1 bit | Q2 bit | Q3 bit | Q0 byte | Q1 byte | Q2 byte | Q3 byte | Q0 NLL | Q1 NLL | Q2 NLL | Q3 NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.05% | 48.78% | 47.85% | 48.88% | 0.20% | 0.39% | 0.39% | 0.00% | 5.952 | 6.012 | 6.084 | 6.231 |
| 2 | 49.71% | 49.58% | 48.73% | 48.56% | 0.20% | 0.39% | 0.20% | 0.39% | 6.187 | 6.211 | 6.285 | 6.495 |
| 3 | 50.00% | 50.00% | 49.39% | 50.34% | 0.78% | 0.98% | 0.59% | 0.59% | 5.949 | 5.996 | 6.075 | 6.275 |
| 4 | 50.37% | 50.27% | 50.46% | 48.80% | 0.59% | 0.20% | 0.20% | 0.20% | 6.061 | 6.237 | 6.418 | 7.036 |
| 6 | 50.17% | 49.41% | 48.88% | 49.63% | 0.59% | 0.39% | 0.00% | 0.20% | 5.905 | 5.937 | 5.996 | 6.210 |
| 8 | 50.39% | 50.24% | 50.27% | 49.51% | 0.20% | 0.00% | 0.20% | 0.00% | 6.096 | 6.148 | 6.244 | 6.494 |
| 12 | 50.37% | 49.80% | 50.85% | 49.58% | 0.59% | 0.20% | 0.98% | 0.59% | 5.975 | 6.028 | 6.146 | 6.350 |
| 16 | 49.68% | 48.66% | 50.37% | 50.49% | 0.00% | 0.00% | 0.00% | 0.20% | 5.954 | 5.989 | 6.069 | 6.234 |

There is no meaningful depth gradient and no above-chance prediction at any Q.
Q0 is the zero-physical-block / DiT-shell baseline. Because no Q predicts the
task, Q0 versus Q1 does not establish task-useful physical-block localization.
The progressive Q-dependent NLL degradation at 5k is not a recurrence result.

## Tiny fixed-set overfit gate

Fresh T_REF models used separate checkpoint namespaces for each case.

| Fixed set | Depth balance | Step passed | Byte accuracy | Exact state | Byte NLL |
|---|---|---:|---:|---:|---:|
| 1 | one depth-1 example | 25 | 100.00% | 100.00% | 0.000013 |
| 8 | one per depth | 50 | 96.88% | 75.00% | 0.1412 |
| 256 | 32 per depth | 1000 | 99.51% | 96.48% | 0.0217 |

The accepted pass criterion was byte accuracy at least 95% and byte NLL at
most 0.25. These results rule out a gross target-encoding, decoder-mask, or
representation-capacity failure. They do not show systematic Rule-30
generalization.

## Gate decision

- Learnability on held-out examples: fail.
- Difficulty gradient: fail because all depths remain at chance.
- Non-ceiling: technically satisfied but scientifically uninformative.
- Physical-block localization: not established because no Q solves the task.
- Tiny-set overfit: pass.

Therefore `FIXED_TRANSITION_V0_TOO_HARD` is the only benchmark-local label.
The authoritative Phase E.1 experiment is gated off, and no claim about Q is
drawn.

The overfit JSONL rows retain their execution-time parent HEAD. Commit
`c8b30a2` records the exact tested overfit implementation; the rows are
preserved rather than rewritten after execution.
