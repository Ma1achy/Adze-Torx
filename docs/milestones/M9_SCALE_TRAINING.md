# M9 — Scale and training systems

Add only now:

- real byte/text dataset pipeline;
- run manifests;
- checkpoint/resume;
- multi-seed experiment launcher;
- profiling;
- compile cache strategy;
- memory budgets;
- larger batch/device execution;
- failure recovery.

## Reproducibility contract

Every run records:

```text
git commit
Torx commit
config
seed tree/root
dataset revision
precision
device/backend
```

## Gate

Repeated training runs are reproducible enough to compare architecture changes and can be resumed without changing model semantics.
