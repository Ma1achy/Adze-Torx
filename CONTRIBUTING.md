# Contributing

Adze-T is an experiment-driven research codebase.

Before opening a change:

1. read `AGENTS.md`;
2. identify the active roadmap milestone;
3. add or update the oracle/control for any new stochastic mechanism;
4. keep Torx usage on documented public APIs;
5. include tests and a results record for behaviour-changing experiments.

Do not add broad abstractions ahead of evidence. Small duplicated experimental
code is preferable to a premature generic framework; promote code into
`src/adze_t/` only after its semantics have been validated.

For stochastic changes, include the estimator/sample count/seeds and comparison
to an independent oracle or control in the pull request description.
