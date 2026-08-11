# Phase B.1 — deterministic correctness hardening

Base commit: `7c5411c01185eb485157662a1773d2d4fda10d37`.

Decision: **PHASE_B_1_PASS**.

The substantive Phase B trainability result remains accepted. This milestone
corrects execution semantics and evidence claims before Phase C; it does not
change the Adze topology or add a Torx implementation.

## Corrections

- AdamW now accepts an update-mask pytree. Disabled leaves are excluded from
  clipping and retain parameters, first moments, and second moments bitwise;
  weight decay is not applied. Codec pretraining and model training use
  explicit complementary masks at the target-codec boundary.
- Empty-K/V attention rows use a guarded masked softmax. The complete attention
  contribution is forced to zero after the O projection, including when its
  bias is non-zero.
- Masked Mamba positions are zero inputs/outputs and selective-scan state
  no-ops. Their delta/B/C values cannot affect later active positions.
- Fixed-structure batches accept explicit byte masks. Omitted masks mean all
  positions are valid, so byte `0x00` is no longer implicit padding.
- Teacher construction has a pure-JAX core with capacity-overflow and
  prefix-mask-valid flags plus an eager wrapper that raises. Oversized targets
  are never silently accepted or truncated by the training path.

Raw subsystem gradient metrics remain connectivity diagnostics computed before
the optimizer update mask. The optimizer's `grad_norm` is now the norm of the
actually trainable gradient leaves.

## Authoritative target-codec rerun

The codec was retrained from the original deterministic seed 700, with the same
1,024-example training set, 256-example validation set, batch size 32, and
reference configuration. No old checkpoint or optimizer state was loaded.

| Step | Validation byte | Validation exact | Validation byte CE | Wall time |
|---:|---:|---:|---:|---:|
| 100 | 21.3867% | 0.0000% | 2.811694 | 51.3 s |
| 250 | 56.9824% | 0.0000% | 1.539491 | 125.8 s |
| 500 | 84.6680% | 26.5625% | 0.575242 | 270.8 s |
| 1,000 | 92.8223% | 55.4688% | 0.311564 | 538.8 s |
| 2,000 | 97.9980% | 85.9375% | 0.079285 | 1,017.9 s |
| 5,000 | 99.4141% | 95.3125% | 0.023071 | 2,511.2 s |

The declared `>=99%` byte and `>=95%` exact-sequence validation gate passed at
step 5,000: **TARGET_CODEC_PRETRAIN_PASS**. The superseded 2,000-step run is
retained unchanged under `results/phase_b/runs/target_codec.jsonl`; the new
machine-readable curve is `target_codec_b1.jsonl`.

A checkpoint-specific model step reported finite loss `10.659152`, applied
gradient norm `37.692989`, bitwise-identical frozen target-codec parameters,
and changed active context-encoder parameters.

## Regression coverage

New deterministic tests cover masked AdamW parameters/moments under non-zero
weight decay, exact codec/model masks, real codec/model steps, empty-K/V
attention with deliberately non-zero O bias, prefix/tail/internal-hole Mamba
masks, masked-site perturbation invariance, all-valid/JIT equivalence, embedded
zero bytes, explicit padding masks, prefix-mask errors, and JIT-visible teacher
capacity overflow.

Validation:

- `make format-check`: passed, 90 files formatted;
- `make lint`: passed;
- `make typecheck`: 0 errors, 0 warnings;
- `make test`: 50 passed in 297.29 seconds;
- `make boundaries`: public dependency-boundary check passed;
- `make test-slow`: passed, 50 tests deselected.

No Phase C code, stochastic Torx operator, Q experiment, `S>1`, or `R>0` work
was implemented.
