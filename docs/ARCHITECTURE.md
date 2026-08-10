# Adze-T architecture

This is the target architecture, not permission to implement every component at once. Each section becomes executable only after its roadmap gate passes.

## 1. Persistent fixed-capacity carrier

Adze-T maintains `C` carrier sites.

Each site contains three principal channels:

```text
h_i       continuous content latent
b_i       boundary state/probability
length_i  categorical byte expansion length, including 0
```

`length_i = 0` means non-emitting/deleted at output. It does **not** imply that the site disappears from the fixed carrier array.

Static carrier capacity is intentional: JAX/Torx compilation should not depend on dynamic ragged allocation during the early architecture.

## 2. Structural state layers

Keep these concepts separate:

```text
observed corrupted state:
    s_b, s_length
    may include UNKNOWN

predicted state:
    p_b, p_length

committed routing state:
    c_b, activity
```

Predictions may change during denoising. Routing is committed only at explicit outer-iteration boundaries.

This prevents rapidly changing predicted boundaries from changing graph connectivity at every inner step.

## 3. Encoders

Two logical encoders exist:

```text
context encoder
    prompt -> conditioning

carrier encoder (training only)
    target bytes -> clean carrier target
```

The heavy iterative denoiser must not be hidden in either encoder.

The exact degree to which the encoders themselves should use Torx stochastic primitives remains milestone-gated.

## 4. Corruption

Content `h` uses a continuous corruption/flow process.

Boundary and length state use discrete corruption. The intended early design is absorbing corruption with an UNKNOWN state.

Corruption must be independently testable and reversible only to the extent the training objective requires.

## 5. Torx-native stochastic core

The heavy core is a stochastic transition kernel rather than a conventional attention/MLP block hidden inside a Torx wrapper.

Candidate mechanisms include:

- local stochastic mixing;
- multiscale interactions at fixed distances;
- Gaussian/continuous stochastic maps;
- binary/categorical switching;
- boundary-controlled couplings;
- repeated weight-tied stochastic transitions.

The exact kernel family is an experimental decision, not fixed by the scaffold.

## 6. Recurrent compute Q

If there are `L` distinct physical stochastic blocks and `Q` cycles:

```text
(B_L o ... o B_1)^Q
```

is one denoiser evaluation.

A useful initial comparison keeps approximately 12 effective block applications, for example:

```text
L=4, Q=3
L=3, Q=4
```

Cycle-index conditioning is an ablation and is **off by default**.

Parameter sharing across repeated applications must be tested explicitly.

## 7. Inner denoising S

The stochastic core is applied over `S` denoising-time transitions.

Keep the conceptual axes separate:

```text
Q = repeated compute inside one denoiser transition
S = denoising-time transitions
R = outer draft/refinement iterations
```

Conflating them will make training/evaluation results uninterpretable.

## 8. Routing and hierarchy

Avoid dynamic hard packing initially.

Use fixed-capacity sites with multiscale interactions whose activity is controlled by committed boundary/activity state.

Potential interaction scales:

```text
1, 2, 4, 8, ...
```

Boundary state acts as a stochastic/committed coupler rather than merely an attention-mask entry.

A direct-carrier debug mode must exist before hierarchical routing is trusted.

## 9. Byte emission

Use fixed output capacity.

`length_i` controls how many byte slots site `i` emits.

Initial representation candidate:

```text
max_bytes_per_site fixed slots
8 pbits per byte slot
```

A 256-way categorical byte representation is an allowed ablation if Torx makes it materially cleaner.

Do not dynamically allocate output arrays from sampled lengths.

## 10. Draft then refine

Outer iteration 0 is the draft:

```text
causal / predecessor-only structure
```

Later iterations may use global/bidirectional stochastic connectivity.

Refinement:

```text
select region
erase/corrupt target (+ optional neighbour)
regenerate with later/global context available
commit structure
```

The selector and regeneration operator are separate mechanisms and must be ablated separately.

## 11. Adaptive compute

Only after fixed-compute correctness.

### Inner S

Stop based on stochastic convergence diagnostics over content/boundary/length state.

### Selection

Select uncertain or causally-disagreeing regions.

### Outer R

Stop when expected utility of another refinement iteration is non-positive.

A post-hoc survival predictor may schedule batches by likely required R, but must not silently cap the model's permitted refinement depth.

## 12. Final training shape

Conceptually:

```text
prompt
  ↓
context conditioning
  ↓
persistent carrier
  ↓
[ stochastic core × Q ] × S
  ↓
draft state
  ↓
(select → erase → [core × Q] × S → commit) × R
  ↓
fixed-slot byte emission
  ↓
loss
```

Every stochastic component must have an identified training-gradient route and an oracle-backed tiny test before scale-up.
