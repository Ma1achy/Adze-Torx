# M4.1 decision

**M4_1_Q_NEUTRAL**

The corrected experiment resolves most of the original M4 initialization and
optimization confound. Both new residual families begin with an exact identity
mean map for Q=1,2,4, and Q-normalized residual keeps core gradient scales much
closer across Q while preserving fixed total nominal variance.

The solvable `T=expm(G)` toy confirms that tied Q-step solutions exist for every
tested Q. On the M3 carrier task, however, neither corrected family gives a
convincing endpoint advantage for Q>1 after the same 240 optimizer steps:

```text
identity residual:       Q1 .429693 +/- .003406, Q2 .431213 +/- .002111,
                          Q4 .443488 +/- .001770
Q-normalized residual:    Q1 .429693 +/- .003406, Q2 .431924 +/- .003794,
                          Q4 .432582 +/- .003517
```

Q-normalization changes Q=4 from the severe original M4 degradation to a much
closer result, and both corrected models show progressive per-cycle refinement
on most validation examples. That is evidence that the previous M4 result was
materially parameterization-confounded, but it is not evidence that extra Q
improves endpoint quality for this affine-Gaussian surrogate.

This conclusion is deliberately narrow: it applies to the tested affine
recurrent surrogate and frozen synthetic task, not to recurrence generally or
to a future multi-block architecture. Carry the corrected Q=1 core forward;
do not begin M5 automatically.
