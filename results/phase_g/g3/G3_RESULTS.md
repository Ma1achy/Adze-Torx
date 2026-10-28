# G3 — scientific evaluation: not executed, upstream gate blocked

Overall programme status: **PHASE_G_BLOCKED** under frozen G23 V0.

| Axis | Outcome | Reason |
|---|---|---|
| Generated structure functionality | BLOCKED | G2 required MC preflight nonpass; no learned-commit training/evaluation |
| Semantic routing value | BLOCKED | No selected generated model or new paired neutral/incorrect/cut comparisons |
| Sampling-estimator training value | BLOCKED | No matched A/B/C adaptation runs or training-seed replication |

These statuses mean not estimable, not evidence that the architecture fails or
that sampling is inferior. No G2/G3 selected checkpoints, performance table,
test confidence interval or across-training-seed variance can honestly be issued.

## What is historical evidence, and what is not

Accepted G11 teacher100: correct-routing query accuracy 59.831870%, valid-cut
accuracy 3.691521%; NLL 1.242239118 vs 11.162091255. Accuracy delta 56.140352 pp,
paired 95% CI54.422516..57.895652 pp; NLL relative reduction 88.870911%.
Both nonquery guards and negative-label invariance passed in THAT experiment.
See ../g1_1/teacher_v0/TEACHER100_RESULTS.md and its immutable manifest.
Those numbers are teacher-committed retrieval, NOT G2 generated commitment.

The frozen validation metadata contains780 neutral-K8-reachable and132
neutral-discordant examples. This membership was verified before the new MC
oracle. No new policy accuracy/NLL on either stratum was evaluated. It is not
valid to infer semantic advantage from the historical teacher-versus-cut gap.

## Frozen but unexecuted science

G3_PROTOCOL.md specifies seeds 0/1/2, exact G2 reuse, conditional common1000/2000
caps, validation-only eligibility/ties, protected-test opening after durable
selection, deterministic and16-root sampled readouts, same-weight matched
teacher/neutral/shifted/cut/q-anchored/no-attention/direct-decoder interventions,
length/gap/competitor/block/extent stratification, paired 2000-replicate example
bootstrap and separate n3 Student-t seed uncertainty. None was substituted with
a smaller test set, single favorable root or unregistered checkpoint.

No test-opening/selection record exists. No protected-test body or model metrics
were constructed in this campaign. Dataset identity metadata from the accepted
preregistration remains part of the frozen protocol only.

## Blocker and integration policy

See ../g2/ORACLE_NONPASS_DIAGNOSIS.md. Exact gradients and key identities pass
diagnosis, but the separate required finite-MC criterion remains failed. No
criteria-preserving implementation repair was identified. Changing its SE,
sample count, root or baseline after seeing the failure would require explicit,
separately versioned prospective validation; no such gate change was executed.

Preserve the isolated G branch. Do not merge into main to imply scientific
completion. Phase H remains gated on actual generated-commitment correctness/
trainability and later G integration. H/I/Thermo are unchanged. No S4 or LM work.
