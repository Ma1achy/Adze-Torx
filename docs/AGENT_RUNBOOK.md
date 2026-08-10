# Coding-agent runbook

The repo is designed for repeated short, reviewable agent runs rather than one
open-ended "build Adze-T" instruction.

## Sequence

1. Read `AGENTS.md`.
2. Read the current milestone under `docs/milestones/`.
3. Read prior `RESULTS.md` / `DECISION.md` records.
4. Use the matching prompt under `docs/prompts/`.
5. Implement only the current milestone.
6. Run the milestone's tests and repo quality gates.
7. Write numerical evidence, not just PASS/FAIL.
8. Write the gate decision.
9. Stop for review.

## Review rule

A human review should answer:

```text
Did the evidence establish the milestone claim?
Are the oracles independent?
Were public Torx APIs sufficient?
Did any assumption become an architecture decision?
Does the next milestone still make sense?
```

The roadmap is allowed to change after any gate.

## Results location

M1 has a pre-created experiment record under:

```text
experiments/m1_trainability/
```

For M2+, create a milestone-specific experiment/results directory only when the
milestone begins. Use `docs/templates/RESULTS_TEMPLATE.md` and
`docs/templates/DECISION_TEMPLATE.md`.

## Commit discipline

Prefer one reviewable commit per completed sub-stage when practical:

```text
M1A public API baseline
M1B discrete recurrence
M1C continuous
M1D mixed
M1E sharing/composites
M1 decision
```

Do not combine a failed experiment with a large unrelated refactor.

## When upstream Torx changes

Do not silently update code until it passes again.

Treat a Torx bump as an integration event:

1. record old/new commit;
2. run public contract tests;
3. run M1 regression/oracle suite;
4. inspect failures;
5. update the dependency record;
6. only then continue model work.

## When to open a new ADR

Create `docs/decisions/NNNN-*.md` for choices that constrain later work, such as:

- adding Temper or GenJAX;
- moving model semantics into THRML;
- changing fixed-capacity carrier semantics;
- changing byte representation;
- adopting a biased gradient estimator;
- changing the Q/S/R interpretation;
- introducing dynamic graph/ragged execution;
- changing Torx's role from semantic source of truth.
