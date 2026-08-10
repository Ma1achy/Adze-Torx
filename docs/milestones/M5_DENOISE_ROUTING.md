# M5 — Inner denoising S + committed routing

Add `S` denoising-time transitions.

Maintain three structural layers:

```text
observed corrupted
predicted
committed routing
```

Commit routing once per outer iteration using hysteresis.

Add monotone activity within an inner trajectory if supported by the selected state representation.

## Requirements

- direct-carrier debug mode remains available;
- fixed carrier shape;
- fixed multiscale interaction graph;
- committed boundaries gate interactions;
- no per-step dynamic graph rebuild.

## Gate

Routing improves target metrics in controlled comparisons and does not create unexplained non-target corruption.
