# THRML notes

THRML is relevant context, not a default dependency.

It is an Extropic JAX library focused on:

- probabilistic graphical models;
- blocked Gibbs sampling;
- discrete energy-based models;
- sparse/heterogeneous graph sampling.

Potential future uses:

1. reference implementation for tiny EBM gradient/oracle tests;
2. specialised EBM training machinery if Adze-T eventually uses an energy-based factor for which THRML is the natural public abstraction;
3. comparison point for block-sampling performance.

Do not route Adze-T through THRML merely because it is adjacent to Torx. Adze-T's model semantics should stay defined in Torx unless an explicit ADR changes that.
