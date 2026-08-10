# M1.5 decision

**MIXED_BRIDGE_PASS**

## Evidence

The narrow Adze-T-local score correction restored the missing mixed discrete
gradient while preserving the existing continuous JAX gradient:

- the previously-zero discrete theta gradient now agrees with the exact mixed
  conditional-moment oracle within measured Monte Carlo uncertainty;
- continuous alpha, beta, and log-variance gradients are unchanged from the
  already-correct pathwise route;
- recurrence depths 1, 2, 4, 8, 16, and 32 pass the 4-standard-error criterion;
- tied shared-parameter semantics remain correct;
- the discrete-only bridge agrees with both the exact Markov oracle and the
  native Torx discrete route;
- per-trajectory batching and complete recurrent log-probability accumulation
  are regression-tested.

## Scope boundary

The implementation is only:

```text
loss + stop_gradient(loss)
     * (log_prob_sum - stop_gradient(log_prob_sum))
```

with strict scalar-or-equal-shape validation. It adds no generic estimator
compiler, registry, planner, baseline, control variate, optimizer, or new
stochastic IR.

Native Torx gradient routes are explicitly excluded from score correction to
avoid double counting. The helper cannot discover route ownership automatically;
that limitation is documented as a caller contract.

## Why this is not TEMPER_CANDIDATE

Torx already supplies validated native routes for the discrete-only and
continuous-only fragments. M1.5 needed only a small composition helper around
public Torx sampling and log-probability calls. No reusable generic compiler
capability was demonstrated to be necessary.

## Next step

M2 may begin only after this M1.5 result is reviewed. This milestone does not
implement M2.

