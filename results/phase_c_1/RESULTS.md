# Phase C.1 / D0 — stochastic occurrence and zero-noise hardening

Base Phase-C commit: `ae19b50dce9bef04cf483314169533c0b7ef5961`.

Frozen deterministic reference: `03d4677dca646b89719284775b035113f6fca6e8`.

Pinned Torx revision: `f1fc858ed950ecd41935d15c06d0ec7c5e0674ae`.

Decision: **PHASE_C_1_PASS**.

The accepted `TORX_PARITY_PASS` decision remains unchanged. This milestone fixes
a finite-noise occurrence-identity blocker without reopening Phase C or
executing a finite-noise model trajectory.

## Alias finding and correction

At the Phase-C base, prompt and target executions of the tied shared frontend
used the same module path and other occurrence coordinates, so they derived the
same stochastic subkey. Mean sharing was correct, but distinct executions could
not receive independent future draws.

The immutable occurrence context now owns an ordered tuple of explicitly
declared static scopes. The full graph executes the tied frontend under:

```text
prompt/frontend.byte_embed
target/frontend.byte_embed
```

Both occurrences still consume the single Torx mean destination
`encoder/byte_embed/mean`. No scope appears in a parameter path and no parameter
was duplicated.

## Exact key derivation

Keys use repeated `jax.random.fold_in` in this frozen order:

```text
root key
evaluation/sample identity
optimizer/training step
ordered static scopes
learned module path
denoise step s
refinement iteration r
recurrence cycle q
physical layer ell
explicit site/tile coordinate
```

Static scope and module strings use BLAKE2s with a four-byte digest interpreted
as little-endian uint32. Python `hash()`, object identity, mutable counters,
global state, execution order, and Python/NumPy RNG are not used.

For root key `[0, 201]`, the actual full-model `frontend.byte_embed` occurrences
derived:

- prompt: `[2277131107, 3505768866]`;
- target: `[3477246704, 179510340]`.

Repeating the construction reproduces both keys exactly. Changing only the root
key changes both derived keys.

## Q recurrence and reuse audit

For physical block 0, the three tied Q-projection occurrences derived:

```text
[528601506, 2007423278]
[1320497939, 1630946592]
[618080976, 4143185596]
```

Its tied block-specific modulation head likewise derived three distinct keys:

```text
[1177672475, 2444389144]
[521164284, 3134868051]
[4250133796, 3670753601]
```

Changing Q does not change parameter topology or counts. The frontend is the
only low-level parameter stack reused across prompt and target in the current
full forward. DiT blocks and modulation heads reuse means across Q. Draft/refine
executions have explicit `mode:draft` / `mode:refine` scopes. The context already
carries `s`, `r`, and site/tile coordinates, so future loops can distinguish
occurrences without another identity redesign; `S=1,R=0` remain unchanged.
Target-codec decoding and final output decoding also have distinct scopes.

## Parameters and zero-noise evidence

- deterministic parameters: `2,268,245`;
- Torx mean parameters: `2,268,245`;
- Torx stochastic-only rho parameters: `18,901`;
- total Torx parameters: `2,287,146`.

With `operator_stochasticity=True` and `lambda_op=0`:

- full-model forward maximum absolute error: `0`;
- every Phase-B loss component error: `0`;
- all 237 mapped raw mean-gradient leaf errors: `0`;
- worst mapped path: `['b_head']['bias']` (all paths tie at zero error);
- maximum absolute rho gradient: `0`;
- different-root-key plus extreme-rho invariance error: `0`;
- public Torx factor occurrences: `237`.

The occurrence count is unchanged. The graph executes project-local learned
stochastic operators through public Torx factor `sample()` methods at zero
noise. This does **not** claim that the complete model has been lowered into a
Torx DFG or physical stochastic execution substrate.

Machine evidence: `occurrence_keys.json` and `full_gradient_parity.json`.

## Remaining Phase-D obligation

`PHASE_D_DIRECT_PARAMETER_NOISE_POLICY_TBD`

Direct learned SSM coefficients such as `a_log`, `d_skip`, `delta_bias`, and
`layer_scale` remain mapped/accessed through public mean factors without a
finite-noise law. No such law was invented here.

## Validation

- `python scripts/run_phase_c_1_evidence.py`: passed; evidence regenerated;
- targeted Phase-C.1 tests: 4 passed across non-slow and slow invocations;
- `make format-check`: passed, 96 files already formatted;
- `make lint`: passed;
- `make typecheck`: 0 errors, 0 warnings;
- `make test`: 76 passed, 2 deselected in 497.26 seconds;
- `make boundaries`: public dependency-boundary check passed;
- `make test-slow`: 2 passed, 76 deselected in 36.93 seconds;
- explicit private-Torx import search: no matches;
- `git diff --check`: passed.

No architecture, parameter sharing, Torx revision, attention, Mamba, pack,
codec, teacher, decoder, mask, residual, `L/Q/S/R`, or accepted Phase-C record
was changed. No finite-noise scientific execution or stochastic training was
performed.
