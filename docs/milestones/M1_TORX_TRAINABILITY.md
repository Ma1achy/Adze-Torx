# M1 — Torx-native trainability spike

## Question

Can Torx + JAX train the mixed continuous/discrete, shared-parameter, recurrent stochastic computation required by Adze-T?

## Stage A — dependency/public API

Record exact:

```text
Python
JAX
Torx distribution
Torx commit
platform
```

Verify documented public interfaces actually used.

## Stage B — discrete recurrence

Use a tiny binary Markov transition with shared parameters.

For transition matrix `K_theta` and initial distribution `pi_0`:

```text
pi_T = pi_0 K_theta^T
J(theta) = pi_T @ cost
```

Differentiate this exact deterministic expression with JAX.

Compare the Torx stochastic gradient at depths:

```text
1, 2, 4, 8, 16, 32
```

Also test sample/log-probability consistency if the gradient route uses `log_probability`.

## Stage C — continuous transition

Use the simplest public Torx continuous/Gaussian transition.

Choose linear/quadratic loss so expected objective and derivative are analytic.

Do not assume pathwise autodiff exists. Determine what the public Torx/JAX path actually does.

## Stage D — mixed state

Preferred toy:

```text
X_t       binary finite state
H_t       affine Gaussian state

X_{t+1} ~ Bernoulli(sigmoid(theta0 + theta1 X_t))
H_{t+1}  = alpha H_t + beta X_t + sigma epsilon_t
```

Use a terminal objective such as:

```text
L = c_x X_T + c_h H_T + q H_T^2
```

Build an independent exact oracle by propagating:

```text
P(X_t)
E[H_t | X_t]
E[H_t^2 | X_t]
```

or another mathematically equivalent finite-state/conditional-moment recursion.

Compare every parameter gradient.

## Stage E — shared parameters

Create tied and untied equivalent programs.

At equal occurrence parameter values:

```text
g_tied = sum_i g_untied_i
```

must hold.

## Stage F — composites

Validate public:

```text
ChainFactor(..., weight_tied=True)
TiledFactor(..., weight_tied=True)
```

against manually constructed equivalents.

## Stage G — gradient-route inventory

For each experiment record the actual public mechanism:

```text
ordinary JAX/pathwise?
score/REINFORCE?
Torx simulator custom VJP?
parameter shift?
exact log-probability surrogate?
other?
```

Do not label a mechanism based on assumptions.

## Required tests

- public API imports;
- sample/log-prob match;
- exact discrete oracle;
- analytic continuous oracle;
- mixed conditional-moment oracle;
- recurrence depth sweep;
- tied vs untied;
- ChainFactor identity/equivalence;
- TiledFactor identity/equivalence;
- fixed-seed reproducibility;
- disconnected randomness zero-gradient test;
- finite values at numerically difficult parameter values.

## Decision

Choose exactly one:

### GO_DIRECT

Torx is sufficient. Proceed to M2.

### TORX_GAP_LOCAL

One small Adze-T-local helper is required but no generic compiler is justified.

### TEMPER_CANDIDATE

A reusable stochastic-gradient/compiler capability is missing.

Must include a minimal Torx counterexample + oracle.

### BLOCKED

Required model semantics cannot currently be expressed through public Torx APIs.

Then stop.
