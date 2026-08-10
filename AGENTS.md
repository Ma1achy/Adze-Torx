# AGENTS.md — Adze-T coding-agent contract

Read this file, `docs/ROADMAP.md`, and the active milestone document before changing code.

## Hard rules

### 1. Torx is an external public dependency

Adze-T is built **around Torx's documented public API**.

Forbidden:

```python
from torx._anything import ...
```

Also forbidden:

- relying on undocumented Torx object layout;
- monkey-patching Torx;
- modifying a vendored Torx checkout;
- copying Torx private implementations into this repo;
- reaching into private fields because a public API is inconvenient.

Reading upstream source to understand documented semantics is allowed.

If the public API is insufficient, stop, document the missing capability, and surface it as a design decision. Do not smuggle an internal dependency into the codebase.

### 2. Do not introduce Temper speculatively

Temper is **deferred**.

Do not add a Temper dependency, GenJAX/ADEV dependency, or generic stochastic-AD compiler unless the current Torx-native experiment demonstrates a concrete gap.

A Temper extraction requires a written decision record answering:

- what exact Torx-native operation is missing?
- why is it reusable beyond Adze-T?
- why can it not be expressed through Torx's public interfaces?
- what exact mathematical contract would Temper own?
- what oracle proves the proposed estimator correct?

### 3. Evidence before architecture

Never build the next abstraction because the roadmap says it may eventually be useful.

For stochastic code:

1. write an independent oracle;
2. implement the smallest Torx program;
3. validate forward semantics;
4. validate gradients statistically/exactly;
5. only then generalise.

### 4. Forward correctness != gradient correctness

Always test separately:

```text
Torx forward program  <-> expected forward law
gradient estimator    <-> exact / analytic gradient
```

A correct sampler can have a wrong derivative, and a mathematically correct score rule can differentiate a `log_probability` implementation that does not match `sample()`.

### 5. Do not relax statistical tests until they pass

Statistical assertions must be predeclared and uncertainty-aware.

Do not respond to a failing stochastic test by repeatedly widening a tolerance. Diagnose bias, PRNG reuse, estimator variance, oracle error, sample/log-prob mismatch, parameter-sharing error, and numerical instability.

### 6. Explicit PRNG only

No Python `random` and no NumPy RNG in model execution. Use JAX keys explicitly. Any deliberate common-random-number coupling must be documented as estimator policy, not accidental key reuse.

### 7. No hidden deterministic neural model in stochastic factors

Torx-native means the heavy iterative computational core is represented by stochastic Torx factors/composites.

Deterministic JAX code is allowed for parameter transforms, lightweight conditioning/projections, losses, oracle calculations, bookkeeping, and debug baselines. It must not conceal an ordinary Transformer/DiT/Mamba denoiser inside a factor.

### 8. Fixed-capacity carrier first

Do not introduce ragged/dynamic-shape carrier allocation in early milestones.

Use fixed carrier capacity and explicit state:

- `h`: continuous content
- `b`: boundary state/probability
- `length`: byte expansion length, including zero/non-emitting
- activity/routing state

### 9. Stop at milestone gates

Do not automatically begin the next milestone.

Every milestone ends with tests run, numerical results, `RESULTS.md` update, unresolved issues, gate decision, and a concise implementation report. Then stop for human review.

### 10. Unknown semantics stay explicit

Use `NotImplementedError`, TODOs tied to milestone IDs, or an open decision record. Do not invent plausible behaviour merely to complete an interface.

### 11. Commit and push completed work

After completing every milestone or requested task, commit all related changes
with a focused message and push the commit to the configured upstream branch.
Verify that the worktree is clean and the local branch is synchronized with its
upstream before reporting completion. Do not push incomplete or unreviewed
work between milestone gates unless explicitly requested.

## Quality requirements

Before declaring a milestone complete, run as applicable:

```bash
make format-check
make lint
make typecheck
make test
make test-slow
```

Search for private dependency use:

```bash
python scripts/check_public_boundaries.py
```

Keep the default test suite deterministic and fast. Put high-sample statistical conformance tests behind `@pytest.mark.slow`.

## What to update when code changes

At minimum:

- active milestone `RESULTS.md`
- active milestone `DECISION.md` if the gate decision changes
- `docs/DEPENDENCIES.md` when dependency pins change
- architecture decision record for a new irreversible design choice
- tests accompanying any new stochastic rule
