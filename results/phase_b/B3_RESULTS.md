# B3 — deterministic trainability ladder

Base commit: `293ed67` plus the Phase B working tree.

## Historical first attempt

Historical status at `c723f42`: **ADZE_D_CORE_TRAINABILITY_FAILURE** (withdrawn).

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

That interpretation is withdrawn. The two-example reverse smoke probe used
only 100 steps, not the approved B3 budget, and a subsequent audit found B0,
B1, and B2 deviations that had to be corrected before a valid experiment. The
measurement above is retained unchanged as historical evidence.

## Corrected re-audit result

Status: **ADZE_D_CORE_PASS**.

All accepted runs use `adze_reference_small_v0`, deterministic operators,
`L=4,Q=3,S=1,R=0`, batch 32 for generated tasks, fixed 8-byte sequences with
values 1..32, zero reserved, fixed K-bucket routing, and deterministic disjoint
datasets. The generated-task training set contains 65,536 examples and the
validation set 256 examples.

| Gate/run | Step | Train byte | Validation byte | Validation exact | Wall time | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| single example | 100 | 100.00% | 100.00% | 100.00% | 15.7 s | 6.37 step/s |
| 16-example overfit | 100 | 99.22% | 99.22% | 93.75% | 165.3 s | 0.61 step/s |
| COPY | 5,000 | 95.51% | 95.75% | 70.31% | 3,852.2 s | 1.30 step/s |
| REVERSE | 5,000 | 96.29% | 96.00% | 73.05% | 3,880.4 s | 1.29 step/s |

COPY validation checkpoints were 17.87%, 19.48%, 19.97%, 26.37%, 45.26%,
and 95.75% at steps 100, 250, 500, 1k, 2k, and 5k. REVERSE was 17.24%,
19.24%, 20.36%, 26.37%, 43.99%, and 96.00%. Majority-byte baselines were
3.16% in both accepted runs, so both exceed baseline by more than 92 points.

Single-example h loss fell from `0.804393` to `0.021122` (97.37% reduction),
with perfect boundary and extent accuracy. The full-batch 16-example gate used
the configured batch size and passed at its first 100-step checkpoint.

At the corrected first-step gradient audit, subsystem norms were: frontend
5.303, context encoder 6.942, proposal 10.061, DiT Q/K/V/O 13.173, DiT FFN
7.982, conditioning/modulation 23.285, output heads 5.704, decoder 2.898, and
global 34.545. Target analysis is deliberately frozen after codec pretraining
and therefore reports zero during B3. Unit tests separately require finite,
non-zero first-step gradients in every physical DiT block and clean head.

Phase B.1 correction: the pre-mask frontend/target norms above remain valid
connectivity diagnostics, but the original zero-gradient freezing mechanism
still allowed AdamW weight decay. Phase B.1 replaces it with update masks that
keep frozen parameters and optimizer moments bitwise unchanged. The accepted
COPY/REVERSE measurements are preserved and are not reinterpreted as having
used the later optimizer implementation.

Machine-readable complete curves and activation RMS trajectories live in
`results/phase_b/runs/`. Diagnostic curves preserve the superseded scalar-pool,
batch-one, 1,024-example, and full-capacity-padding investigations; none is
used for the final decision.

Addition was not run. It remains diagnostic-only and is not a corrected Phase
B pass requirement. No Torx backend, Q science experiment, S>1, or R>0 work was
started.
