# Adze-T

**Adze-T** is the Torx-native implementation track for Adze: a persistent-carrier, draft-then-refine language model whose stochastic computational core is expressed through Torx rather than hidden inside a conventional Transformer/DiT.

This repository is organised as a sequence of **evidence-gated milestones**. The first task is not to build a language model. It is to establish that Torx + JAX can train the mixed continuous/discrete, weight-shared, recurrent stochastic computations that Adze-T actually needs.

## Current architectural decision

Build Adze-T directly on Torx first.

**Temper is not a dependency.** Temper should only be revived/extracted if the Torx-native trainability spike exposes a concrete, reusable stochastic-gradient gap that cannot be handled cleanly through Torx's public API.

See:

- `AGENTS.md` — non-negotiable rules for coding agents
- `NEXT_STEPS.md` — exact first task/prompt
- `docs/ROADMAP.md` — milestone sequence from spike to endpoint
- `docs/ARCHITECTURE.md` — intended Adze-T architecture
- `docs/TORX_INTEGRATION.md` — Torx boundary and dependency policy
- `docs/TESTING.md` — oracle, contract, statistical, integration, and e2e strategy
- `docs/decisions/0001-build-directly-on-torx.md` — why Temper is deferred

## Repository shape

```text
src/adze_t/
    model/          carrier, corruption, routing, core, denoise, refinement, emission
    torx_api/       public Torx boundary only
    train/          objectives and training loop
    oracle/         exact/analytic validation machinery
    eval/           metrics and diagnostics

experiments/
    m1_trainability/    first Torx-native trainability spike

tests/
    unit/
    contracts/
    oracle/
    integration/
    statistical/
    metamorphic/
    e2e/

docs/
    milestones/
    decisions/
    prompts/
```

Most model modules intentionally contain `NotImplementedError`. This scaffold must not pretend unresolved semantics are implemented.

## Quick start

Python 3.11+ is expected because the current Torx line requires it.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torx]"
make check
```

If the pinned Torx commit has moved or its dependency constraints have changed, do **not** work around the public API. Update the pin deliberately and record the change in `docs/DEPENDENCIES.md` and the active milestone `RESULTS.md`.

## Development philosophy

1. **Known answer before unknown answer.** Every new stochastic-gradient route is validated against an analytic or exact oracle on a tiny system before it is used in a model whose correct answer is unknown.
2. **Public Torx API only.** No `torx._...`, undocumented object-layout assumptions, monkey-patches, or modifications to Torx.
3. **One milestone at a time.** Coding agents stop at review gates.
4. **Forward semantics and gradient correctness are separate claims.**
5. **No silent estimator substitutions.** If a gradient route is approximate, biased, unsupported, or empirically suspect, say so.
6. **No hidden conventional denoiser.** A custom Torx factor may use normal JAX arithmetic required to parameterise a stochastic kernel, but it must not hide a Transformer/DiT/Mamba-style model in `Factor.sample()` and call that "Torx-native".
7. **Keep shapes static initially.** Adze-T uses a fixed-capacity persistent carrier; deletion/non-emission is represented in state rather than by dynamically changing JAX array shapes.

## License

Choose and add the project licence before public release.
