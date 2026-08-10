# Next steps

The repo starts at **M1: Torx-native trainability**.

Do not start the language model yet.

The first task is to determine whether Torx + JAX already provide enough differentiation machinery for the computational motifs Adze-T needs.

## First agent prompt

Copy/paste the block below into the coding agent.

```text
You are implementing milestone M1 of Adze-T: the Torx-native trainability spike.

Before changing code, read:
- AGENTS.md
- README.md
- docs/ROADMAP.md
- docs/TORX_INTEGRATION.md
- docs/TESTING.md
- docs/milestones/M1_TORX_TRAINABILITY.md
- experiments/m1_trainability/README.md
- docs/decisions/0001-build-directly-on-torx.md

The purpose of this milestone is NOT to build Adze-T and NOT to build Temper.

The question is:

Can Torx + JAX, using only Torx's public API, correctly train the mixed
continuous/discrete, shared-parameter, recurrent stochastic computations that
Adze-T needs?

Hard constraints:
1. Public Torx API only. No torx._* imports, private-field reliance,
   monkey-patching, or Torx modifications.
2. Do not add GenJAX/ADEV or Temper.
3. Build an independent exact/analytic oracle before trusting any stochastic
   gradient.
4. Forward-law validation and gradient validation are separate tests.
5. Do not move to M2 automatically. Finish M1, write results, then stop.

Implement M1 in the stages defined by docs/milestones/M1_TORX_TRAINABILITY.md:

A. Dependency/API baseline
   - inspect the pinned Extropic Torx public API;
   - record Python/JAX/Torx versions and commit;
   - verify the required public factor/capability/composite interfaces;
   - update experiments/m1_trainability/RESULTS.md.

B. Discrete recurrent case
   - construct the smallest public-Torx finite-state stochastic transition;
   - use a shared parameter across repeated applications;
   - create an exact finite-state Markov oracle;
   - validate sampling/log_probability consistency where applicable;
   - validate the stochastic gradient against the exact JAX gradient;
   - test recurrence depths 1,2,4,8,16,32.

C. Continuous case
   - construct the smallest affine Gaussian/continuous Torx case available
     through the public API;
   - choose an objective with analytic moments/gradients;
   - determine what Torx/JAX gradient route is actually supported rather than
     assuming pathwise differentiation;
   - compare against the analytic oracle.

D. Mixed case
   - combine a small discrete state and a continuous state;
   - use an objective whose expected value/gradient is still exactly
     computable, preferably by finite-state dynamic programming plus Gaussian
     conditional moments;
   - validate the whole mixed gradient.

E. Parameter sharing
   - compare tied parameters against an equivalent untied construction
     evaluated at equal parameter values;
   - verify tied gradient = sum of untied occurrence gradients.

F. Torx composites
   - compare manual recurrence against public ChainFactor semantics;
   - compare manual tiling against public TiledFactor semantics;
   - test weight_tied=True;
   - use exact/statistical forward and gradient checks.

G. Native Torx differentiation routes
   - inventory the public training/differentiation route actually used by each
     experiment;
   - if a Torx estimator is biased or unsupported for this use case, reproduce
     the failure with the smallest oracle-backed test and record it;
   - do not work around it with private APIs.

For every stochastic-gradient experiment record:
- exact/analytic objective;
- exact/analytic gradient;
- estimated gradient mean;
- standard deviation;
- standard error;
- error in standard-error units;
- seeds/sample count;
- recurrence depth;
- parameter-sharing mode;
- dtype;
- compile/runtime notes.

Add or complete tests under:
- tests/contracts/
- tests/oracle/
- tests/statistical/
- tests/metamorphic/
- tests/integration/

Required metamorphic tests include:
- tied gradient equals sum of untied gradients;
- ChainFactor depth 1 equals base factor;
- manual recurrence agrees with ChainFactor;
- TiledFactor n=1 equals base factor;
- disconnected stochastic state has zero gradient;
- fixed seed reproduces exactly where the public RNG contract permits.

Run all quality gates:
- formatting
- Ruff
- Pyright
- fast pytest
- slow statistical pytest
- python scripts/check_public_boundaries.py

At the end:
1. update experiments/m1_trainability/RESULTS.md with numerical evidence;
2. fill experiments/m1_trainability/DECISION.md;
3. explicitly choose one:
   GO_DIRECT: Torx is sufficient; proceed to M2.
   TORX_GAP_LOCAL: a small Adze-T-local helper is needed.
   TEMPER_CANDIDATE: a concrete reusable stochastic-gradient gap exists and
   deserves a separate Temper design review.
   BLOCKED: Torx public API cannot express a required primitive.
4. report files changed, tests run, numerical results, and architectural
   surprises.
5. STOP. Do not begin M2.
```

## After M1

Use the prompt corresponding to the next approved milestone under `docs/prompts/`.

Never skip a gate because the next stage looks straightforward.
