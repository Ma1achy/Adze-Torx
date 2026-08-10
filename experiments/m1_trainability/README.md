# M1 trainability experiment

This directory owns the first architecture spike.

## Objective

Determine whether Adze-T can be trained directly with Torx + JAX.

The experiment progresses:

```text
public API baseline
→ discrete recurrence
→ continuous transition
→ mixed transition
→ tied parameters
→ ChainFactor
→ TiledFactor
→ decision
→ M1.5 mixed-gradient bridge
```

See `docs/milestones/M1_TORX_TRAINABILITY.md`.

## Files

- `discrete.py` — finite-state Torx experiment
- `continuous.py` — continuous/Gaussian experiment
- `mixed.py` — coupled mixed-state experiment
- `M1_5_RESULTS.md` — mixed-gradient bridge numerical record
- `M1_5_DECISION.md` — M1.5 review-gate decision
- `oracles.py` — experiment-local independent oracles
- `RESULTS.md` — numerical record
- `DECISION.md` — final M1 gate

Do not move reusable code into `src/adze_t/` until it has passed its oracle tests.
