# B2 re-audit — deterministic surrounding topology

Decision: **B2_MODEL_PASS**.

The first-attempt gated tanh recurrences were not faithful Mamba and are no
longer described as such. Frontend, context, target analysis, proposal, and
decoder now reuse a local explicit Mamba-1-style selective SSM block:

```text
input projection/split
 -> causal depthwise convolution
 -> input-dependent delta/B/C
 -> stable diagonal selective scan
 -> SiLU gate
 -> output projection
 -> residual
```

All configured layer counts are active. The proposal accepts an explicit
carrier prior. The decoder uses carrier-major/slot-minor ordering and consumes
the committed clean prediction `h_hat`, so byte loss reaches the clean content
head.

Provisional reference micro-choices are: expansion 2, state width 16, causal
kernel width 3, `A=-exp(a_log)`, softplus delta with initial bias -2, diagonal
scan, learned D skip, and residual layer scale. These are
**PROVISIONAL**, not recovered original-Adze facts.

The target codec uses `PROVISIONAL_PHASE_B_TEACHER`: for any
`N <= C*L_max`, bytes map monotonically to `(t//L_max,t%L_max)`, extent is
`clamp(N-i*L_max,0,L_max)`, activity derives from positive extent, and logical
boundaries independently cut fixed K buckets plus the terminal sentinel. An
ordered within-carrier slot concatenation/projection creates site-distinct h0.

The shared target codec/decoder was pretrained on deterministic 8-byte data.
At step 2,000 it reached 100% train byte/exact accuracy and 97.27% held-out byte
accuracy (80.47% exact sequence); held-out boundary/extent losses were
`0.000339`/`0.000486`. Target-side h0 is then stop-gradient/frozen for the
denoising loss, while the normal decoder remains trainable. Changing target
bytes changes h0, and tests cover slot ordering, proposal prior influence,
configured depth, eager/JIT full forward, and decoder-path gradients.
