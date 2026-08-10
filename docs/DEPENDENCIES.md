# Dependencies

## Torx

Intended upstream:

- repository: `https://github.com/extropic-ai/torx`
- distribution: `extro-torx`
- import namespace: `torx`

Initial reproducibility pin used by the scaffold:

```text
f1fc858ed950ecd41935d15c06d0ec7c5e0674ae
```

The pin is a **starting point, not a permanent compatibility promise**. Before changing it, run the active milestone's public-API contract tests and record the reason/version change.

Torx is intentionally isolated as an optional dependency in `pyproject.toml` while the earliest scaffold/unit tests remain runnable without it.

## THRML

THRML is not an Adze-T dependency at scaffold time.

It is relevant prior art/reference for:

- discrete probabilistic graphical models;
- Gibbs/block-Gibbs sampling;
- discrete EBM utilities;
- exact-vs-stochastic gradient testing patterns.

Only add THRML if a later milestone makes an explicit decision that a THRML primitive is genuinely part of the desired Adze-T execution/training path.

## GenJAX/ADEV / Temper

Neither is a dependency.

Temper was originally considered as a generic stochastic-gradient compiler. Current policy is to first establish what Torx already supports natively.

If M1 exposes a reusable gap, create a decision record before adding any stochastic-AD backend.

## Dependency policy

- Pin unstable research dependencies.
- Record upstream commit hashes in milestone results.
- Prefer public, documented extension surfaces.
- Never pin a dependency by copying its private implementation into this repo.
