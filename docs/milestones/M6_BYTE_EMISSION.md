# M6 — Fixed-slot byte emission

Implement final output packing without ragged JAX allocation.

## Initial design

Each carrier site owns a fixed maximum number of byte slots.

`length_i` determines how many are observable.

Primary candidate:

```text
8 pbits / byte slot
```

Optional ablation:

```text
one categorical byte state
```

## Required invariants

- length zero emits nothing;
- emitted ordering is deterministic given committed lengths;
- inactive/non-emitting sites do not condition emission incorrectly;
- max-capacity overflow is explicit, never silent.

## Gate

Exact packing tests plus successful tiny byte-sequence reconstruction.
