# B3 — deterministic trainability ladder

Base commit: `293ed67` plus the Phase B working tree.

Status: **ADZE_D_CORE_TRAINABILITY_FAILURE** (sanity-task gate unresolved).

The real deterministic architecture can optimize a single fixed example:
after 100 compiled AdamW steps, total loss was approximately `0.138` and byte
loss approximately `0.0068`, with finite component losses and gradients.

A two-example short reverse probe reached approximately 50% emitted-byte
accuracy after 100 compiled steps, below the predeclared 90% sanity-task
threshold. The attempted larger-batch probe was stopped after excessive CPU
compile time; no result was treated as evidence.

Therefore the required copy/reverse ladder is not passed and `ADZE_D_CORE_PASS`
is not issued. The implementation stops at B3 for diagnosis before any Torx,
Q-scaling, S>1, or R>0 work.
