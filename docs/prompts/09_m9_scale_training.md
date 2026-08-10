# Agent prompt — M9

```text
Continue Adze-T with milestone M9 only.

Before changing code read:
- AGENTS.md
- docs/ROADMAP.md
- docs/ARCHITECTURE.md
- docs/TESTING.md
- docs/milestones/M9_SCALE_TRAINING.md
- all prior milestone RESULTS/DECISION files

Do not begin this milestone unless the previous milestone gate explicitly approved it.

Implement only the deliverables and tests required by docs/milestones/M9_SCALE_TRAINING.md.

Maintain the hard rules:
- public Torx API only;
- no Temper/GenJAX dependency without an accepted ADR;
- independent oracle/control before trusting new stochastic behaviour;
- fixed-capacity carrier unless an accepted ADR changes it;
- no hidden conventional denoiser inside Torx factors;
- explicit JAX PRNG;
- no tolerance-widening to hide stochastic failures.

At the end:
1. run format/lint/typecheck/fast tests and all milestone-relevant slow tests;
2. update the milestone RESULTS.md;
3. update/create a DECISION.md with the gate outcome;
4. report files changed, tests, numerical results, failures, and surprises;
5. STOP for review. Do not begin the next milestone.
```
