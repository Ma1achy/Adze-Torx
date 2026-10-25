# Phase G sampled-gradient oracle

Corrected production-loss oracle status: PASS in focused development checks.

- Exact action space: `2^2 * 3^3 = 108`.
- Exact local bridge: 23/23 tests passed.
- Monte Carlo: 21/21 parameter/loss combinations passed using the prospective
  seed 8100, 16 roots, and 4096 samples per root.
- Acceptance: every component within four root-level standard errors; a
  zero-standard-error component uses absolute tolerance `1e-6`.
- The corrected oracle evaluates production global-compaction byte loss and a
  direct continuous parameter path, so its exact gradient is score plus direct.

The earlier count-surrogate run is preliminary algebra evidence only. Its three
peaked failures are preserved:

- missing boundary means approximately `[3.1625e-7,-3.1632e-7]` and
  `[-1.4170e-6,1.4171e-6]`, SE approximately `[4.050e-8,4.051e-8]` and
  `[1.8148e-7,1.8148e-7]`, absolute z about 7.81;
- extra boundary means approximately `[2.2967e-7,-2.2971e-7]` and
  `[-1.0290e-6,1.0291e-6]`, SE approximately `[2.0185e-8,2.0185e-8]` and
  `[9.0426e-8,9.0435e-8]`, absolute z about 11.38;
- direct-dependencies extent rows had component z values approximately
  6.56 and 6.14--6.98.

Those failures arose in the rejected surrogate near-zero-gradient fixtures and
do not override the corrected production-loss gate. Durable JSON evidence is
written only by the exact-protocol oracle runner; no sampled smoke arm may start
without it.
