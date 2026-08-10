# Testing strategy

Adze-T should be harder to fool than a normal ML prototype.

## Oracle hierarchy

Use multiple independent routes whenever possible.

### A. Analytic closed form

For tiny Bernoulli, categorical, Gaussian, and linear/quadratic cases.

### B. Exact finite-state enumeration

For tiny discrete Torx graphs:

```text
J(theta) = sum_trajectory p_theta(trajectory) L(trajectory)
```

Implement the sum independently and differentiate it with ordinary JAX.

### C. Exact finite-state propagation

For recurrent Markov systems:

```text
pi_T = pi_0 K_theta^T
J = pi_T @ cost
```

This scales to long recurrence while the state space remains tiny.

### D. Conditional-moment dynamic programming

For mixed discrete/linear-Gaussian systems, propagate finite-state probabilities and conditional Gaussian moments. This provides an exact mixed oracle without enumerating continuous samples.

### E. Finite differences

Secondary numerical oracle only. Sweep step size rather than trusting one `h`.

### F. Deterministic limits

Useful sanity check, never the only gradient oracle.

## Required test families

### Unit

Pure transforms, state invariants, config, corruption, routing, metrics.

### Contract/interface

- public Torx imports;
- factor capability assumptions;
- `sample`/`log_probability` consistency;
- parameter-sharing identity;
- composite public semantics.

### Oracle

Known expected objectives and exact gradients.

### Statistical conformance

For stochastic estimators compute:

```text
mean
variance
standard deviation
standard error
z/error in standard-error units
RMSE across seed blocks
```

Tests must account for Monte-Carlo uncertainty.

### Metamorphic

Examples:

```text
tied gradient = sum of untied occurrence gradients
ChainFactor(depth=1) = base factor
manual recurrence = ChainFactor
TiledFactor(n=1) = base factor
manual tiles = TiledFactor
adding disconnected randomness leaves objective/gradient unchanged
fixed seed reproduces
```

### Integration

Actual Torx DFG/composite + Adze-T modules.

### E2E

Small trainable tasks first, then tiny language modelling.

## Statistical-test policy

Do not assert:

```python
abs(stochastic_gradient - exact_gradient) < arbitrary_tolerance
```

Prefer a predeclared confidence rule based on measured standard error.

Long statistical tests use `@pytest.mark.slow`.

Fast PR tests should use deterministic or low-cost fixed-seed checks.

## Numerical policy

Oracles should prefer float64 where available.

Guard:

- log probabilities near 0/1;
- underflow in trajectory probabilities;
- long-chain matrix powers;
- accidental NaN/Inf;
- probability normalization;
- invalid categorical states.

## Mutation mentality

For critical contracts, deliberately break the implementation once while writing the test.

Examples:

- treat shared parameters as untied;
- swap a Bernoulli log-probability branch;
- reuse one PRNG key;
- invert a boundary mask.

If the test still passes, the test is not strong enough.
