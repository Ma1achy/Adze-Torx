# Adze-T roadmap

The roadmap is evidence-gated. A milestone may change later milestones.

```text
M0  repo + dependency/public-API baseline
 ↓
M1  Torx-native trainability spike
 ↓
M2  direct-carrier reconstruction model
 ↓
M3  full carrier channels + corruption
 ↓
M4  Torx-native recurrent core + Q
 ↓
M5  denoising S + structure commit/routing
 ↓
M6  byte emission
 ↓
M7  outer refinement R
 ↓
M8  adaptive selection + stopping
 ↓
M9  scale/training systems
 ↓
M10 endpoint evaluation + ablations
 ↓
M11 optional thermal/hardware integration
```

## Global gate rule

No milestone advances because "the code runs".

Each milestone must establish:

1. semantic correctness;
2. gradient correctness where trainable;
3. measurement/diagnostics;
4. regression tests;
5. a written decision.

## M0 — baseline

Freeze repo conventions, dependency pin, public-API policy, CI, and test taxonomy.

**Gate:** clean scaffold and reproducible environment.

## M1 — Torx-native trainability

Prove mixed continuous/discrete, tied, recurrent Torx computation can be trained with Torx + JAX alone, or identify the exact missing capability.

**Gate:** `GO_DIRECT`, `TORX_GAP_LOCAL`, `TEMPER_CANDIDATE`, or `BLOCKED`.

## M2 — direct-carrier reconstruction

Build the smallest fixed-capacity carrier model with no learned hierarchy.

Start with content channel and fixed structure so the stochastic core can be debugged without routing confounds.

**Gate:** trainable synthetic reconstruction + tiny byte task; stochastic model beats trivial/no-update baselines; deterministic/debug equivalents agree where expected.

## M3 — carrier structure and corruption

Add boundary and byte-length channels, UNKNOWN corruption state, and deletion via `length=0`.

Keep routing direct/fixed.

**Gate:** each channel learns independently; joint model preserves content while recovering structure; corruption/recovery tests pass.

## M4 — recurrent Torx core Q

Introduce the actual Torx-native stochastic transition architecture and weight-tied repeated cycles.

Ablate:

```text
L,Q with equal effective depth
tied vs untied
cycle conditioning off/on
local vs multiscale coupling
```

**Gate:** recurrence is stable/trainable and parameter sharing is verified.

## M5 — denoising S and structure routing

Add S denoising-time transitions, predicted structural state, once-per-outer commit, hysteresis, monotone activity, and fixed-shape multiscale routing.

Maintain direct-carrier debug mode.

**Gate:** routing changes improve the intended metric without corrupting non-target state; direct mode remains a reproducible control.

## M6 — byte emission

Implement fixed-slot bytes controlled by `length`.

Compare 8-pbit bytes against categorical bytes if useful.

**Gate:** exact packing/emission invariants + successful reconstruction on tiny byte sequences.

## M7 — outer refinement R

Iteration 0 causal/draft. Later passes global/refinement.

Implement:

```text
selector
erase/corrupt
regenerate
commit
```

Start with externally specified/random selectors before learned selection.

**Gate:** controlled global refinement causally improves selected reconstruction without unexplained non-target damage.

## M8 — adaptive compute

Add uncertainty/disagreement selection, S convergence criteria, R stopping, and optional post-hoc batch scheduling predictor.

**Gate:** adaptive compute trades quality for work monotonically and does not silently cap useful refinement.

## M9 — scale/training systems

Add checkpointing, resumable runs, experiment manifests, deterministic seed accounting, larger datasets, profiling, compilation/memory budgets, and distributed/batched execution only when required.

**Gate:** reproducible multi-run training at the first meaningful scale.

## M10 — endpoint evaluation

Run full benchmark and ablation suite.

At minimum isolate:

```text
causal draft vs global refinement
Q recurrence
S denoising
R refinement
boundary channel
length channel
routing commits
selector choice
weight tying
cycle conditioning
direct-carrier vs hierarchical routing
```

**Gate:** every claimed architectural benefit is supported by a controlled comparison.

## M11 — optional Thermalizers/hardware path

Only after software semantics/training are stable.

Investigate Extropic Thermalizers/THRML/hardware lowering as separate execution backends without changing the model's mathematical contract.

**Gate:** backend equivalence tests against the software Torx reference.
