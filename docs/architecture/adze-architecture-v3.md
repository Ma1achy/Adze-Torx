# Adze Architecture Reset
## One Architecture, Two Computational Backends

**Status:** architecture/design specification — audited and implementation-gated baseline  
**Revision:** 2026-08-10 v3 — Adze conformance, pinned-Torx feasibility, and reference build freeze added  
**Purpose:** define the canonical Adze architecture to be implemented in two forms:

- **Adze-D** — deterministic reference implementation
- **Adze-T** — Torx-native stochastic implementation

The central rule is:

> **There is one Adze architecture. The deterministic and Torx variants differ in computational substrate, not in model topology.**

The original non-Torx Adze endpoint is the architectural source of truth. Torx should implement the same information-processing graph as faithfully as possible. Architectural substitutions made purely because they appear more “Torx-like” are out of scope for the baseline and must instead be treated as later ablations.

---

# 1. Motivation for the reset

The earlier Adze-T design gradually moved away from the original Adze endpoint in several important ways. In particular, it considered replacing the packed block stream, full attention, Transformer/DiT MLPs, and parts of the original encoder/decoder stack with local or multiscale stochastic kernels chosen partly for hardware-nearness.

That produced a valid new stochastic architecture, but it was no longer a faithful test of Adze.

This matters because the original Adze hypothesis depends heavily on:

1. a persistent latent carrier,
2. generated block structure,
3. hard packing,
4. a real Transformer/DiT heavy core,
5. weight-tied recurrent depth \(Q\),
6. denoising depth \(S\),
7. draft-then-global-refine depth \(R\),
8. encoder/proposal/decoder components surrounding the heavy core.

If those are replaced before the baseline exists, then later experiments cannot cleanly answer whether the original Adze architecture works when implemented stochastically.

The project therefore resets around:

\[
\boxed{\text{Adze defines the architecture. Torx defines the computational substrate.}}
\]

---

# 2. Two backends, one architecture

The deterministic and stochastic implementations must share:

- the same byte-level interface,
- the same encoder topology,
- the same training-only clean target encoder,
- the same persistent carrier layout,
- the same \(h,b,\ell\) semantics,
- the same generated block structure,
- the same hard pack and unpool operations,
- the same \(M\times K\) transient block representation,
- the same Transformer/DiT attention topology,
- the same FFN/SwiGLU topology,
- the same \(L\) physical blocks,
- the same recurrent reuse count \(Q\),
- the same draft/refine masks,
- the same denoising loop \(S\),
- the same outer refinement loop \(R\),
- the same proposal path,
- the same decoder topology,
- the same losses,
- the same data,
- the same tensor shapes wherever possible,
- the same parameter naming scheme.

The difference is the learned operator backend.

For a deterministic learned affine map,

\[
y=Wx+b,
\]

Adze-T instead uses an explicit stochastic Torx operator with the same mean parameters, e.g.

\[
y\sim\mathcal N(Wx+b,\Sigma).
\]

Likewise, deterministic categorical logits in Adze-D correspond to explicit Torx categorical/pdit factors in Adze-T.

The comparison is therefore:

\[
\boxed{\text{same model architecture, different computational substrate}.}
\]

---

# 3. Canonical end-to-end topology

```text
                           PROMPT BYTES
                               |
                               v
                    SHARED BYTE / ROUTER-1
                           FRONTEND
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
         CONTEXT ENCODER              TARGET ENCODER
        inference-visible             training-only
                 |                           |
                 v                           v
            context c                 clean carrier X0
                                             |
                                       h0, b0, l0
                 |                           |
                 +-------------+-------------+
                               |
                               v
                    corruption / prior
                               |
                               v
                 persistent C-site carrier
                         X = (h,b,l)
                               |
                               | committed structure
                               v
                         HARD PACK
                               |
                               v
                    M x K BLOCK STREAM
                               |
                               v
                 +-------------------------+
                 | LOOPED DiT CORE         |
                 |                         |
                 | B1 -> B2 -> ... -> BL   |
                 |  ^                |     |
                 |  +------ x Q -----+     |
                 +-------------------------+
                               |
                               v
                            UNPOOL
                               |
                               v
                   residual carrier update
                               |
                               v
                     clean-state heads
                        h_hat, b_hat, l_hat
                               |
                         x S denoise
                               |
                               v
                             DRAFT
                               |
                       select / erase
                               |
                               v
                    GLOBAL REFINEMENT
                         same weights
                               |
                         x R iterations
                               |
                               v
                     committed carrier
                               |
                               v
                       MAMBA DECODER
                               |
                               v
                         OUTPUT BYTES
```

The source topology is intentionally the original Adze endpoint:

```text
bytes
  -> Mamba byte frontend
  -> SSM proposal
  -> hard pack
  -> looped DiT
  -> unpool
  -> h/b/l heads
  -> Mamba decoder
  -> bytes
```

---

# 4. Byte-level interface

Adze remains tokeniser-free. The architectural baseline uses raw bytes:

\[
x_j\in\{0,\ldots,255\}.
\]

For the first faithful implementation, a byte should remain a 256-way categorical symbol rather than immediately being decomposed into eight binary variables.

- **Adze-D:** ordinary 256-way embedding / categorical output.
- **Adze-T:** 256-state pdit or equivalent Torx categorical representation.

A later hardware-oriented ablation may compare a 256-state pdit against 8 pbits per byte. That is not part of the faithful baseline.

---

# 5. Shared byte frontend and dual encoders

The original Adze design shares the low-level byte/frontend machinery between:

1. the inference-visible context encoder,
2. the training-only clean target/carrier encoder.

```text
bytes
  |
  v
shared byte / Router-1 frontend
  |
  +----------------------+
  |                      |
  v                      v
context branch      target-analysis branch
```

The shared frontend learns low-level byte and local structure. After this shared portion, the two branches diverge because they serve different semantic roles.

The deterministic and Torx backends must use the same topology and hidden sizes.

## 5.1 Context encoder

The context encoder transforms prompt bytes into the conditioning state used by the proposal network and DiT core.

If the original endpoint uses a Mamba/SSM-style encoder, that topology is preserved. Do not replace it with a context lattice, local propagation network, Transformer, or unrelated pooled representation merely for Torx convenience.

A schematic deterministic transition,

\[
s_{t+1}=A_t s_t+B_t x_t,
\]

becomes in Adze-T:

\[
s_{t+1}\sim\mathcal N(A_t s_t+B_t x_t,\Sigma_t).
\]

Learned projections that produce selective SSM parameters, gates, or readouts should likewise have explicit Torx equivalents. Parameter-free algebra may remain deterministic JAX.

## 5.2 Clean target encoder

The training-only target encoder maps target bytes to:

\[
X_0=\{h_0,b_0,\ell_0\}.
\]

It shares the low-level byte frontend with the context encoder but has a separate analysis path afterwards.

---

# 6. Persistent carrier

The persistent output representation is a fixed-capacity carrier of \(C\) sites:

\[
X=\{X_i\}_{i=1}^{C}.
\]

At minimum:

\[
X_i=(h_i,b_i,\ell_i),
\]

where:

- \(h_i\in\mathbb R^{d_h}\): continuous semantic/reasoning state,
- \(b_i\): boundary state,
- \(\ell_i\): byte extent / emission length.

For implementation clarity, structural state is split into:

### 6.1 Observed/corrupted state

\[
s_{b,i},\qquad s_{\ell,i}.
\]

These may include an `UNKNOWN` state.

### 6.2 Current predictions

\[
p_{b,i},\qquad p_{\ell,i}.
\]

These may evolve during denoising.

### 6.3 Committed routing state

\[
c_{b,i},\qquad a_i.
\]

These determine block construction, packing, activity, key/value participation, and emission semantics.

The invariant is:

> **Current structural predictions do not immediately rewrite the computational topology.**

During one inner denoising trajectory, predictions may change while committed routing remains stable. Commit occurs only at controlled boundaries.

---

# 7. Extent and inactive-site semantics

\[
\ell_i=0
\]

means that carrier site \(i\):

- emits no output bytes,
- contributes no key/value content to shared communication,
- contributes no pooling content,
- remains able to receive information,
- remains able to reactivate later.

The invariant is:

> **query-active, key/value/pool-inactive**

for a non-emitting site.

Internal deletion must not reindex later carrier identities.

---

# 8. Generated structure and block construction

Generated structure remains load-bearing. The heavy Transformer does not operate directly on the persistent \(C\)-site carrier. Committed structural state generates a transient block representation:

\[
(c_b,a)\longrightarrow\mathcal P,
\]

where \(\mathcal P\) is the packing map.

The packer constructs a padded/static representation such as

\[
H_{\mathrm{block}}\in\mathbb R^{B\times M_{\max}\times K\times d},
\]

with:

- \(C\): persistent carrier capacity,
- \(M\): generated number of blocks,
- \(M_{\max}\): padded maximum number of blocks,
- \(K\): fixed block capacity,
- \(d\): DiT hidden width.

The packer should return at least:

- packed block tensor,
- valid block mask,
- valid slot mask,
- carrier-to-packed map,
- packed-to-carrier inverse map,
- block IDs,
- carrier IDs,
- within-block position IDs,
- attention masks,
- key/value activity masks,
- pool activity masks.

---

# 9. Hard pack and unpool are deterministic

Hard packing remains deterministic in both backends.

Valid deterministic operations include:

- prefix sums,
- gather/scatter,
- bucket/pad,
- masking,
- inverse indexing,
- block-ID construction.

This is not a compromise in stochastic purity: pack/unpool are indexing and geometry operations, not learned reasoning.

A fixed \(M_{\max}\) with masks is acceptable for static-shape/JAX convenience. Removing the block stream is not.

The core dataflow is:

```text
persistent C carrier
      |
      v
hard pack
      |
      v
transient M x K block stream
      |
      v
looped Transformer / DiT
      |
      v
unpool
      |
      v
residual update to persistent carrier
```

---

# 10. DiT block: preserve a real Transformer

A physical block \(B_\ell\) must remain a genuine Transformer/DiT block.

Do not replace it with affine+tanh recurrence, local pairwise kernels, generic dense recurrent maps, or nearest-neighbour stochastic couplers.

A canonical block is:

\[
x'=x+\operatorname{MHA}(\operatorname{AdaNorm}(x,c_{\mathrm{cond}})),
\]

followed by

\[
x''=x'+\operatorname{FFN}(\operatorname{AdaNorm}(x',c_{\mathrm{cond}})).
\]

The exact normalization and FFN type should follow the original endpoint. If the endpoint uses SwiGLU, preserve SwiGLU. If it uses GELU, preserve GELU.

---

# 11. Multi-head attention

For each physical DiT block:

\[
Q=XW_Q+b_Q,
\qquad
K=XW_K+b_K,
\qquad
V=XW_V+b_V.
\]

Then:

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),
\]

\[
H=AV,
\qquad
Y=HW_O+b_O.
\]

The attention mask \(M\) depends on draft/refine mode and generated block structure.

## 11.1 Adze-D attention

All learned projections are ordinary deterministic maps.

## 11.2 Adze-T attention

The topology is identical, but learned projections are explicit Torx stochastic maps:

\[
Q\sim\mathcal N(XW_Q+b_Q,\Sigma_Q),
\]

\[
K\sim\mathcal N(XW_K+b_K,\Sigma_K),
\]

\[
V\sim\mathcal N(XW_V+b_V,\Sigma_V).
\]

JAX then computes the exact attention algebra:

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),
\qquad
H=AV,
\]

followed by a stochastic Torx output projection.

Attention therefore remains attention.

---

# 12. FFN / SwiGLU

For a SwiGLU block:

\[
u=W_u x+b_u,
\qquad
g=W_g x+b_g,
\]

\[
z=\operatorname{SiLU}(g)\odot u,
\]

\[
y=W_d z+b_d.
\]

Adze-D uses deterministic affine maps.

Adze-T uses stochastic Torx affine maps for the learned projections while keeping SiLU and the elementwise product deterministic.

This preserves the actual Transformer FFN.

---

# 13. What “pure Torx” means

“Pure Torx” does **not** mean every elementary mathematical operation must become stochastic.

The following may remain deterministic JAX operations:

- reshape,
- transpose,
- gather/scatter,
- masking,
- softmax,
- normalization,
- positional transforms / RoPE,
- activation functions,
- elementwise products,
- residual addition,
- prefix sums,
- packing/unpacking,
- fixed corruption equations,
- scheduling/bookkeeping.

The requirement is:

> **Large learned transformations that move model state should be explicit stochastic Torx operations.**

This includes:

- byte-to-latent lifts,
- Mamba input/output projections,
- Mamba state transitions,
- proposal transformations,
- DiT Q/K/V/O projections,
- DiT FFN projections,
- conditioning projections,
- content transition heads,
- categorical structure factors,
- decoder state transitions,
- decoder output factors.

Forbidden:

```text
one custom Torx factor
    |
    +-- internally runs a giant deterministic Adze model
    |
    +-- samples only at the end
```

The stochastic sites must remain explicit and compositional.

---

# 14. Looped DiT core

The original recurrence is restored exactly.

There are \(L\) distinct physical blocks:

\[
B_1,\ldots,B_L.
\]

The physical stack is

\[
B_L\circ\cdots\circ B_1,
\]

and is repeated \(Q\) times:

\[
F_\theta=(B_L\circ\cdots\circ B_1)^Q.
\]

Parameters are:

- distinct across physical blocks \(\ell=1,\ldots,L\),
- tied across recurrence cycles \(q=1,\ldots,Q\).

For example,

\[
L=4,\quad Q=3
\]

gives twelve effective Transformer block applications using four parameter sets.

---

# 15. Q is compute depth, not numerical integration

No \(1/Q\) normalization is applied to ordinary Transformer residual updates.

Increasing \(Q\) means:

> **run another complete pass of the shared physical Transformer stack.**

It does not mean splitting one update into smaller Euler steps or preserving a fixed ODE horizon.

Useful matched-effective-depth comparisons include:

\[
(L,Q)\in\{(12,1),(6,2),(4,3),(3,4),(2,6),(1,12)\}.
\]

Separate experiments should test the same trained model under varying \(Q_{\mathrm{test}}\).

---

# 16. DiT conditioning

Define effective depth index:

\[
d_{\mathrm{eff}}=qL+\ell.
\]

The DiT conditioning vector may contain:

\[
e=e_\nu+e_m+e_r+e_s+e_{d_{\mathrm{eff}}},
\]

where:

- \(e_\nu\): noise/corruption level,
- \(e_m\): draft/refine mode,
- \(e_r\): outer refinement iteration,
- \(e_s\): denoising step,
- \(e_{d_{\mathrm{eff}}}\): effective recurrent depth.

The same conditioning interface is used in Adze-D and Adze-T.

Cycle/effective-depth conditioning should be explicitly ablated rather than accidentally introduced.

---

# 17. Draft versus refine mode

Draft and refinement use the **same model weights**.

They differ in communication policy and conditioning.

## 17.1 Draft mode

Within one generated block:

\[
\text{bidirectional attention}.
\]

Across generated blocks:

\[
\text{causal attention only}.
\]

Earlier blocks must not receive information from later blocks.

## 17.2 Refine mode

Global bidirectional attention is enabled across all valid packed positions.

There is no separate refinement Transformer.

---

# 18. Inner denoising loop S

\(S\) is distinct from \(Q\).

At denoising step \(s\), the model receives a corrupted carrier \(X_s\), then:

1. hard-packs committed structure,
2. runs the looped DiT core,
3. unpools,
4. predicts clean state,
5. re-corrupts to the next requested noise level if another step remains.

```text
X_s
  |
  v
pack
  |
  v
(B_L ... B_1)^Q
  |
  v
unpool
  |
  v
predict clean h/b/l
  |
  v
re-corrupt to next noise level
  |
  v
X_{s+1}
```

---

# 19. Typed corruption

Adze retains three corruption channels.

## 19.1 Content

\[
h_\nu=\alpha(\nu)h_0+\sigma(\nu)\epsilon.
\]

## 19.2 Boundary

\[
b_\nu=
\begin{cases}
b_0,&1-\nu_b,\\
\mathrm{UNKNOWN},&\nu_b.
\end{cases}
\]

## 19.3 Extent

\[
\ell_\nu=
\begin{cases}
\ell_0,&1-\nu_\ell,\\
\mathrm{UNKNOWN},&\nu_\ell.
\end{cases}
\]

The deterministic and Torx implementations use the same corruption semantics.

---

# 20. Clean-state prediction and structure commitment

After DiT output is unpooled back into the carrier:

\[
H_{\mathrm{carrier}}^{\mathrm{new}}
=
H_{\mathrm{carrier}}^{\mathrm{old}}+U(H_{\mathrm{block}}),
\]

heads predict:

\[
\hat h_0,
\qquad
p(b_0),
\qquad
p(\ell_0).
\]

During one \(S\)-step trajectory:

- \(p_b\) may change,
- \(p_\ell\) may change,
- \(h\) may change,
- committed routing remains stable unless the original endpoint explicitly requires otherwise.

After the trajectory:

```text
current structural predictions
        |
        v
commit / hysteresis / structural decision
        |
        v
new committed boundaries and activity
        |
        v
next hard pack
```

---

# 21. Outer refinement loop R

Iteration \(r=0\) produces the draft.

Each refinement iteration performs:

\[
\text{select}
\rightarrow
\text{erase/reset}
\rightarrow
\text{global re-denoise}
\rightarrow
\text{commit}.
\]

Mathematically:

\[
X^{(r+1)}
=
\operatorname{Commit}
\circ
\operatorname{Denoise}_{\theta,\mathrm{global}}^S
\circ
\operatorname{Reset}_{e^{(r)}}
(X^{(r)}).
\]

The full process is:

\[
X_{\mathrm{final}}
=
\mathcal R_\theta^R
\left(
\operatorname{Denoise}_{\theta,\mathrm{draft}}^S
(X_{\mathrm{proposal}})
\right).
\]

---

# 22. Selector and reset

Candidate selector evidence may include:

- causal/global disagreement,
- structural entropy,
- repeated-sample variance,
- content uncertainty,
- snapshot confidence,
- learned expected-error estimates.

For first parity work, the selector may remain deterministic in both backends. It is control logic rather than the main heavy reasoning computation.

For selected regions:

\[
h_i\rightarrow\text{high-noise/prior state},
\]

\[
b_i\rightarrow\mathrm{UNKNOWN},
\qquad
\ell_i\rightarrow\mathrm{UNKNOWN}.
\]

Unselected state should remain pinned/preserved according to the original refinement semantics.

---

# 23. Proposal network

The original proposal path remains.

Generation begins without target bytes, so the model needs a prompt-conditioned initial proposal over the carrier, including enough information to seed:

- continuous carrier content,
- initial extent/activity,
- initial boundaries/blocks.

If the original proposal is SSM/Mamba-like, preserve that topology in both backends.

Adze-T stochasticises the learned projections/transitions; it does not replace the proposal with an unrelated factor graph.

---

# 24. Decoder

The original Mamba decoder is retained.

Carrier site \(i\) expands according to \(\ell_i\). A static implementation may allocate:

\[
C\times L_{\max}
\]

potential byte slots and mask according to extent.

### Adze-D

Deterministic Mamba/SSM transitions and final 256-way byte logits.

### Adze-T

Same Mamba/SSM topology, but learned state-changing transformations become Torx stochastic factors. Final byte output uses a 256-state categorical/pdit factor in the faithful baseline.

---

# 25. Stochastic Mamba / SSM

Mamba remains Mamba.

A schematic deterministic state transition:

\[
s_{t+1}=A_t s_t+B_t x_t
\]

becomes:

\[
s_{t+1}\sim\mathcal N(A_t s_t+B_t x_t,\Sigma_s).
\]

Keep:

- selective scan,
- gating,
- state-space recurrence,
- local convolution if present,
- residual structure,
- input-dependent SSM parameter generation.

Where those require learned projections, Adze-T uses Torx stochastic operators.

Do not replace Mamba with another recurrent model simply because it is easier to express.

---

# 26. Deterministic-vs-Torx parameter correspondence

Every learned deterministic operator gets a corresponding Torx operator with the same mean parameters.

For example:

```text
dit.block_2.attn.q.weight
dit.block_2.attn.q.bias

dit.block_2.attn.k.weight
dit.block_2.attn.k.bias

dit.block_2.ffn.up.weight
dit.block_2.ffn.gate.weight
dit.block_2.ffn.down.weight

encoder.ssm.in_proj.weight
decoder.ssm.out_proj.weight
...
```

Adze-T may add stochastic parameters such as variance or mixture parameters, but should not invent an unrelated mean parameterization.

---

# 27. Near-zero-noise parity limit

A hard design invariant is:

\[
\boxed{\text{Adze-T}(\sigma\rightarrow0)\approx\text{Adze-D}.}
\]

For a stochastic affine factor:

\[
y_T=Wx+b+\sigma\epsilon.
\]

Then:

\[
\lim_{\sigma\to0}y_T=Wx+b=y_D.
\]

This must be tested module-by-module and end-to-end.

---

# 28. Checkpoint transfer

A deterministic checkpoint should load directly into the stochastic model mean parameters:

\[
W_T\leftarrow W_D,
\qquad
b_T\leftarrow b_D.
\]

This enables:

1. train Adze-D,
2. verify it works,
3. load into Adze-T,
4. begin with \(\sigma\approx0\),
5. gradually introduce stochasticity,
6. fine-tune.

This directly asks whether a known-good Adze computation survives stochasticisation.

---

# 29. Three primary comparison arms

## 29.1 Adze-D

Train deterministic Adze from scratch.

Purpose:

- architectural reference,
- task solvability,
- known-good checkpoint.

## 29.2 Adze-T(port)

Initialize from Adze-D mean parameters, then gradually introduce stochasticity.

Question:

> Can a working Adze computation survive transition to Torx stochastic computation?

## 29.3 Adze-T(scratch)

Train the stochastic model directly from scratch.

Question:

> Can Torx learn the same architecture natively?

These are different scientific questions and must be reported separately.

---

# 30. Model stochasticity vs diffusion stochasticity

Keep two axes separate.

### Computational backend

- deterministic learned operators,
- stochastic Torx learned operators.

### Diffusion sampling

- ODE-like / deterministic path,
- SDE / stochastic re-noising path.

Useful comparisons include:

```text
Adze-D + ODE
Adze-D + SDE
Adze-T + ODE-like schedule
Adze-T + SDE
```

This prevents stochastic neural computation from being conflated with diffusion stochasticity.

---

# 31. Noise semantics under recurrence

The primary recurrent model uses fixed per-application semantics.

Do not automatically divide gate noise or residual strength by \(Q\).

Each physical block application is the same operation every time it is reused.

Therefore \(Q\uparrow\) means:

- more Transformer computation,
- more stochastic operations,
- greater recurrent trajectory length.

A matched-total-noise control may be run separately, but it should not replace the primary semantics.

---

# 32. Training strategy

The actual architecture must be shown to work before recurrence claims are made.

## Stage A — deterministic architecture

Establish that Adze-D works:

- encoder forward path,
- target carrier encoding,
- generated block structure,
- hard pack/unpool,
- single DiT pass,
- reconstruction/autoencoding,
- one-step denoising,
- stable optimization.

If this fails, do not interpret Torx results.

## Stage B — deterministic/Torx parity

Copy Adze-D weights into Adze-T and use near-zero variance.

Require:

- module-level parity,
- pack parity,
- attention parity,
- encoder parity,
- decoder parity,
- end-to-end parity.

## Stage C — finite stochasticity

Increase Torx variance gradually and measure degradation/recovery.

## Stage D — recurrent depth \(Q\)

Only after the actual Transformer architecture is trainable should \(Q\) experiments begin.

## Stage E — denoising depth \(S\)

Add full denoising/re-noising trajectories.

## Stage F — generated structure

Allow learned \(b,\ell\) to affect future committed packing.

## Stage G — outer refinement \(R\)

Activate the full draft/select/erase/global-refine/commit process.

---

# 33. Task strategy

Simple tasks are sanity tests, not the final evidence base.

Suggested progression:

1. copy / reverse,
2. addition,
3. long-carry addition,
4. multiplication,
5. expression evaluation,
6. program execution,
7. deeper algorithmic tasks,
8. code reasoning/execution,
9. chess tactics,
10. Go tactical/life-and-death tasks.

A later load-bearing experiment should examine:

\[
\text{task difficulty}\times Q_{\mathrm{test}}.
\]

---

# 34. Interpretation of M1–M4.5

Prior milestones remain in the scientific record, but their role changes.

They should be described as:

> **Torx substrate and simplified recurrent-proxy experiments.**

They established useful facts about:

- Torx gradient correctness,
- continuous stochastic recurrence,
- discrete score gradients,
- the local score bridge,
- tied recurrence,
- noise accounting,
- fixed-horizon vs compute-scaling semantics,
- optimization behavior of flat recurrent proxies.

They do **not** constitute decisive evidence for or against recurrence in the original Adze looped Transformer architecture.

Labels such as `M4_Q_NOT_USEFUL`, `M4_3_Q_NEGATIVE`, and `M4_5_CORE_REDESIGN` must be interpreted only within the exact proxy studied.

---

# 35. Future hardware-near redesigns

The following remain interesting, but they are future ablations rather than the faithful baseline:

- replacing attention with local stochastic couplers,
- removing hard packing,
- replacing the transient block stream with a fixed local carrier,
- pbit-only byte representations,
- replacing SwiGLU with mixture factors,
- local multiscale communication in place of global attention,
- hardware-specific sparse interaction topology,
- aggressively thermalized nonlinearities.

The proper comparison is:

\[
\text{faithful Adze-T}
\quad\text{vs}\quad
\text{hardware-near variant}.
\]

---

# 36. Architecture invariants

The faithful baseline freezes the following rules:

1. There is one Adze architecture and two computational backends.
2. Adze-D and Adze-T use matching tensor layouts.
3. The original byte-level frontend is preserved.
4. The dual encoder structure is preserved.
5. The persistent \(C\)-site carrier is preserved.
6. \(h,b,\ell\) are preserved.
7. Generated structure is preserved.
8. Hard pack is preserved.
9. The transient \(M\times K\) block stream is preserved.
10. Multi-head attention is preserved.
11. The original DiT FFN is preserved.
12. The looped physical Transformer stack is preserved.
13. \(Q\) means repeated Transformer computation.
14. No \(1/Q\) residual normalization is used in primary recurrence semantics.
15. Pack \(\rightarrow\) DiT \(\rightarrow\) unpool residual flow is preserved.
16. Draft causal masking is preserved.
17. Refine global masking is preserved.
18. Draft and refine share weights.
19. \(S\) remains distinct from \(Q\).
20. \(R\) remains distinct from both.
21. Selection/erase/regenerate is preserved.
22. Mamba/SSM encoder/proposal/decoder topology is preserved.
23. Output remains tokeniser-free and byte-level.
24. Adze-T learned state-changing operations must be explicit Torx factors.
25. Deterministic algebra may remain JAX.
26. A giant hidden deterministic neural model inside one Torx factor is forbidden.
27. Adze-T must approach Adze-D in a near-zero-noise limit.
28. Deterministic checkpoints must transfer into Adze-T mean parameters.
29. Architectural substitutions motivated by Torx convenience are future ablations only.
30. Negative proxy results may not be generalized to the faithful architecture.

---

# 37. Implementation structure

The cleanest implementation is a shared high-level architecture with backend-specific learned operators.

```text
adze/
  architecture/
    carrier.py
    packing.py
    masks.py
    conditioning.py
    dit.py
    encoder.py
    proposal.py
    decoder.py
    denoise.py
    refinement.py

  backends/
    deterministic/
      linear.py
      categorical.py
      ssm.py
      attention_proj.py

    torx/
      linear.py
      categorical.py
      ssm.py
      attention_proj.py

  experiments/
    parity/
    q_scaling/
    denoising/
    structure/
    refinement/
```

A shared operator interface should keep architecture code independent of backend choice.

Conceptually:

```python
class LearnedOps(Protocol):
    def linear(...): ...
    def categorical(...): ...
    def ssm_transition(...): ...
```

Then:

```python
model = AdzeModel(ops=DeterministicOps(...))
```

or:

```python
model = AdzeModel(ops=TorxOps(...))
```

Do not fork the model into two unrelated codebases.

---

# 38. Required parity tests

Before meaningful comparisons, verify:

## Structural parity

- identical carrier shapes,
- identical packing,
- identical block IDs,
- identical masks,
- identical unpool maps,
- identical structure commit behavior.

## Module parity

With matched mean parameters and near-zero Torx variance:

- byte frontend,
- context encoder,
- target encoder,
- proposal,
- Q/K/V projections,
- attention output,
- FFN,
- Mamba transitions,
- clean-state heads,
- decoder.

## End-to-end parity

For fixed inputs, corruption, committed structure, masks, and parameters, require

\[
\|F_T-F_D\|
\]

to lie below a predeclared tolerance as stochastic variance approaches zero.

---

# 39. Scientific questions enabled by the reset

Once the faithful baseline exists, the project can cleanly ask:

### Q1 — stochastic substrate

Can Torx reproduce deterministic Adze behavior?

### Q2 — native trainability

Can Adze-T train directly rather than only inherit deterministic weights?

### Q3 — recurrent depth

Does increasing \(Q_{\mathrm{test}}\) improve task quality for the same trained weights?

### Q4 — parameter efficiency

Can lower \(L\), higher \(Q\) retain or improve quality at lower unique parameter count?

### Q5 — denoising compute

How does quality scale with \(S\)?

### Q6 — refinement

Does global erase-and-regenerate \(R\) improve uncertain regions without unacceptable collateral damage?

### Q7 — generated structure

Do learned \(b,\ell\) and block formation improve quality or efficiency relative to fixed structure?

### Q8 — hardware-near replacements

Can later local/thermal stochastic substitutions match the faithful Adze-T baseline?

---

# 40. Final architecture statement

The canonical Adze model is:

\[
\boxed{
\text{Byte encoder}
\rightarrow
\text{persistent }(h,b,\ell)\text{ carrier}
\rightarrow
\text{generated hard pack}
\rightarrow
(B_L\cdots B_1)^Q
\rightarrow
\text{unpool}
\rightarrow
\text{clean-state prediction}
}
\]

inside an \(S\)-step denoising process, followed by \(R\) rounds of:

\[
\boxed{
\text{select}
\rightarrow
\text{erase/reset}
\rightarrow
\text{global re-denoise}
\rightarrow
\text{commit}
}
\]

and finally a byte-level Mamba/SSM decoder.

Adze-D executes the learned transformations deterministically.

Adze-T executes the same learned transformations as explicit stochastic Torx factors while preserving the original topology.

The intended relationship is:

\[
\boxed{\text{Adze-T}(\sigma\rightarrow0)\approx\text{Adze-D}.}
\]

The primary comparison is therefore not between two different architectures, but between:

\[
\boxed{\text{one Adze architecture under two computational substrates}.}
\]

---

# 41. Project rule going forward

> **When porting an original Adze component to Torx, first ask: “What is the closest semantics-preserving Torx implementation of this exact operation?”**
>
> Do not ask:
>
> “What stochastic architecture could replace this?”

Any replacement is a separate hypothesis and requires an explicit controlled ablation against the faithful Adze-T baseline.


---

# 42. Operational-contract precedence

Sections 42 onward tighten the earlier architecture description into implementation contracts. They are not a redesign; they make the already-selected architecture executable without leaving major choices to the implementation agent.

The precedence rule is:

1. an explicit architectural decision in the original non-Torx Adze endpoint;
2. an explicit contract in this document;
3. a deliberately labelled `TBD / experiment`;
4. otherwise: **stop and raise the ambiguity rather than inventing a replacement**.

No implementation convenience may silently replace attention, packing, Mamba/SSM topology, generated structure, the carrier, or the \(Q/S/R\) loop semantics.

---

# 43. Carrier -> block -> DiT -> carrier operational contract

> **Conformance status:** the pack/block/unpool topology is recovered from original Adze, but several micro-semantics in this section (especially one-carrier/one-packed-slot, exact positional features, and capacity handling) are provisional operational choices until checked against the original non-Torx endpoint source. They must not be described as original-Adze facts.

## 43.1 Boundary coordinates

Committed boundaries live on carrier edges.

For carrier sites

\[
i=0,\ldots,C-1,
\]

define

\[
c_{b,i}\in\{0,1\}
\]

for the edge **after** carrier site \(i\). Thus \(c_{b,i}=1\) means:

\[
i\;|\;i+1.
\]

For implementation, the final edge is a forced sentinel:

\[
c_{b,C-1}=1.
\]

It is not scored as a learned boundary.

Two carrier sites \(i<j\) belong to the same logical block iff there is no committed cut between them:

\[
\operatorname{sameblock}(i,j)
=
\mathbf 1
\left[
\sum_{k=i}^{j-1}c_{b,k}=0
\right].
\]

Boundaries remain attached to carrier coordinates even if one or more sites inside a region have \(\ell=0\).

An inactive hole therefore **does not shift later identities and does not move a boundary**.

## 43.2 Logical block construction

The committed boundary vector partitions the ordered carrier coordinates into \(M\) contiguous logical intervals:

\[
\mathcal B_m=[s_m,e_m],
\qquad
m=0,\ldots,M-1.
\]

Block order is carrier order:

\[
m_1<m_2
\Longrightarrow
e_{m_1}<s_{m_2}.
\]

Each persistent carrier site belongs to exactly one logical block and exactly one packed slot.

There is no duplication of carrier identities in the faithful baseline.

## 43.3 Why inactive sites remain present in the packed representation

The faithful baseline prioritises semantics over an assumed compute saving.

Every carrier site in a logical block is assigned a packed query slot, including a site with

\[
\ell_i=0.
\]

This preserves the invariant:

> query-active, key/value/pool-inactive.

For packed slot \(u\) corresponding to carrier site \(i(u)\), define:

\[
q_u = 1
\]

for every valid packed carrier slot,

\[
kv_u=a_{i(u)},
\]

\[
pool_u=a_{i(u)},
\]

and

\[
emit_u=a_{i(u)}.
\]

Thus an inactive carrier can read active context and reactivate, but its current latent state cannot condition other sites through keys, values, pooling, or output emission.

A future efficiency ablation may physically omit inactive query slots only if it preserves this reactivation semantics by another proven-equivalent path.

That is not the baseline.

## 43.4 Meaning of \(K\)

\(K\) is the compile-time/bucketed maximum number of carrier sites represented inside one logical block.

It is not a semantic subtoken expansion factor.

For a block

\[
\mathcal B_m=[s_m,e_m],
\]

the logical block length is

\[
k_m=e_m-s_m+1.
\]

The baseline requires:

\[
k_m\le K
\]

for the selected execution bucket.

If a generated block is longer than \(K\):

- select a larger precompiled \(K\) bucket, or
- fail the batch with a structured capacity error.

Do **not**:

- truncate the block,
- silently split it,
- move its boundary.

Those would change generated structure.

Likewise if

\[
M>M_{\max},
\]

use a larger \(M_{\max}\) bucket or fail explicitly. Never drop blocks.

## 43.5 Packed tensor and maps

For batch size \(B\),

\[
Z_{\mathrm{pack}}
\in
\mathbb R^{B\times M_{\max}\times K\times d_{\mathrm{model}}}.
\]

The pack operation must also return:

```text
block_valid      [B, M_max]
slot_valid       [B, M_max, K]
query_mask       [B, M_max, K]
kv_mask          [B, M_max, K]
pool_mask        [B, M_max, K]
emit_mask        [B, M_max, K]

carrier_to_m     [B, C]
carrier_to_k     [B, C]

packed_to_carrier[B, M_max, K]

block_id         [B, M_max, K]
carrier_id       [B, M_max, K]
within_block_pos [B, M_max, K]
```

`packed_to_carrier=-1` marks padding.

The forward and inverse maps must be exact inverses on valid carrier sites.

## 43.6 Packed feature construction

For carrier \(i\) mapped to block \(m\), position \(k\), the baseline packed hidden state is:

\[
z_{m,k}
=
P_h h_i
+
e_{\mathrm{carrier}}(i)
+
e_{\mathrm{block}}(m)
+
e_{\mathrm{within}}(k)
+
e_{\ell}(s_{\ell,i})
+
e_{b,L}(s_{b,i-1})
+
e_{b,R}(s_{b,i}),
\]

with learned sentinel embeddings at the left and right sequence edges.

The structural inputs here are the **currently observed/corrupted structural state** \(s_b,s_\ell\), not the current uncommitted predictions \(p_b,p_\ell\).

Committed \(c_b,a\) determine topology/masks.

Predicted \(p_b,p_\ell\) remain outputs/evidence until the next commit boundary.

This prevents a same-step prediction from rewriting its own input topology.

The projection

\[
P_h:\mathbb R^{d_h}\rightarrow\mathbb R^{d_{\mathrm{model}}}
\]

has deterministic and Torx implementations with matched mean weights.

## 43.7 Position information

The baseline uses both:

1. explicit carrier/block/within-block embeddings in the packed input, and
2. deterministic RoPE on attention \(Q/K\) using the persistent carrier coordinate \(i\).

RoPE uses the persistent carrier coordinate, not the packed array index, so repacking does not change a site's absolute identity.

Ablating either positional path is allowed later.

## 43.8 Unpool

Each valid packed slot corresponds to exactly one persistent carrier site, so the faithful baseline uses one-to-one scatter rather than averaging.

Let the final looped-DiT packed state be

\[
z'_{m,k}.
\]

Then

\[
\Delta h_i
=
P_{\mathrm{out}}z'_{m(i),k(i)}.
\]

The carrier residual update is

\[
h_i^{+}
=
h_i+\Delta h_i.
\]

Padding slots contribute nothing.

No mean over block members is used in the faithful baseline.

If the original Adze endpoint specifies an additional block summary channel, it must be added explicitly as a separate path rather than hidden inside unpool.

## 43.9 Worked block example

Suppose

```text
carrier index:  0  1  2  3  4  5
activity a:     1  1  0  1  1  1
cut after i:    0  1  0  0  1  1
```

Then the logical blocks are:

```text
B0 = [0,1]
B1 = [2,3,4]
B2 = [5]
```

Carrier 2 remains physically present in `B1` as a query slot even though it is inactive.

For `B1`:

```text
carrier:    2  3  4
query:      1  1  1
key/value:  0  1  1
pool:       0  1  1
emit:       0  1  1
```

The boundary coordinates do not move because carrier 2 is inactive.

If carrier 2 later commits to positive extent, it becomes active in place; carrier 3 and carrier 4 keep their identities.

---

# 44. DiT block operational specification

> **Conformance status:** full attention, an MLP/SwiGLU path, residual computation, looped physical blocks, and layer/iteration/noise conditioning are recovered Adze commitments. The exact pre-norm/AdaLN modulation recipe and residual-gate initialization below are provisional operational defaults unless the original endpoint source confirms them.

The faithful baseline uses a conventional modern DiT/Transformer block rather than a proxy recurrence.

## 44.1 Block form

Each physical block \(B_\ell\) is pre-norm with adaptive LayerNorm modulation and two residual branches:

\[
x_1
=
x
+
g_{\mathrm{attn}}
\odot
\operatorname{MHA}
\left(
\operatorname{AdaLN}_{\mathrm{attn}}(x;c)
\right),
\]

\[
x_2
=
x_1
+
g_{\mathrm{ffn}}
\odot
\operatorname{SwiGLU}
\left(
\operatorname{AdaLN}_{\mathrm{ffn}}(x_1;c)
\right).
\]

LayerNorm itself is deterministic and parameter-free with respect to learned affine scale/bias; shift/scale come from the conditioning path.

The FFN is SwiGLU.

The attention is standard multi-head scaled dot-product attention.

## 44.2 Adaptive LayerNorm

For normalized state

\[
\bar x=\operatorname{LN}(x),
\]

conditioning produces

\[
(\Delta_{\mathrm{attn}},
\Gamma_{\mathrm{attn}},
g_{\mathrm{attn}},
\Delta_{\mathrm{ffn}},
\Gamma_{\mathrm{ffn}},
g_{\mathrm{ffn}})
=
f_{\ell}(c).
\]

Then:

\[
\operatorname{AdaLN}_{\mathrm{attn}}(x;c)
=
(1+\Gamma_{\mathrm{attn}})\odot\bar x
+
\Delta_{\mathrm{attn}},
\]

and similarly for the FFN branch.

The modulation network is physical-block-specific and tied across recurrence cycles with the rest of \(B_\ell\).

The deterministic and Torx backends share the same modulation topology.

Its learned affine transformations are deterministic in Adze-D and explicit Torx stochastic transforms in Adze-T.

## 44.3 Residual-gate initialization

The faithful baseline uses small non-zero residual gates rather than exactly zero gates:

\[
g_{\mathrm{attn}},g_{\mathrm{ffn}}
\approx g_0,
\qquad
0<g_0\ll1.
\]

This preserves near-identity initialization without intentionally giving the recurrent branch an exactly zero first-step gradient.

The same \(g_0\) is used in Adze-D and Adze-T.

Its numerical value is a configuration hyperparameter and must be fixed before matched runs.

## 44.4 Attention

For \(H\) heads and head width \(d_h=d_{\mathrm{model}}/H\):

\[
Q=XW_Q+b_Q,
\qquad
K=XW_K+b_K,
\qquad
V=XW_V+b_V.
\]

Apply deterministic RoPE using persistent carrier coordinates.

Then:

\[
A=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}+M
\right),
\]

\[
Y=AW_V,
\]

followed by output projection \(W_O\).

All learned projections have backend-paired deterministic/Torx implementations.

## 44.5 SwiGLU FFN

The FFN is:

\[
u=W_u x+b_u,
\]

\[
g=W_g x+b_g,
\]

\[
z=\operatorname{SiLU}(g)\odot u,
\]

\[
y=W_dz+b_d.
\]

The expansion ratio

\[
d_{\mathrm{ff}}/d_{\mathrm{model}}
\]

is a shared configuration value between Adze-D and Adze-T.

## 44.6 No \(1/Q\) scaling

Neither attention nor FFN residual branches are divided by \(Q\).

The block executed at recurrence cycle \(q\) is the same block, with the same learned parameters and the same residual semantics, as at every other cycle.

---

# 45. DiT conditioning operational interface

> **Conformance status:** the existence of noise, iteration, and layer/effective-depth conditioning is part of the recovered Adze design. Concatenation layout, embedding dimensions, and the exact modulation network remain implementation-level defaults until source-conformance is complete.

## 45.1 Conditioning inputs

One core evaluation receives:

- global prompt context \(c_{\mathrm{prompt}}\),
- corruption/noise embedding \(e_\nu\),
- draft/refine mode embedding \(e_m\),
- denoise-step embedding \(e_s\),
- outer-refinement embedding \(e_r\),
- effective-depth embedding \(e_d\).

Define:

\[
d_{\mathrm{eff}}=qL+\ell,
\]

where \(q\in[0,Q-1]\) is recurrence cycle and \(\ell\in[0,L-1]\) is physical block index.

The conditioning input is:

\[
c_{\mathrm{raw}}
=
[
c_{\mathrm{prompt}};
e_\nu;
e_m;
e_s;
e_r;
e_d
].
\]

A shared conditioning trunk maps this to \(d_{\mathrm{cond}}\), then each physical block owns its own modulation head \(f_\ell\).

## 45.2 Prompt context

For the faithful baseline, the context encoder returns a contextual sequence

\[
C_{\mathrm{seq}}
\in
\mathbb R^{B\times P\times d_c}
\]

and a masked mean-pooled global conditioning vector:

\[
c_{\mathrm{prompt}}
=
\frac{\sum_j m_j C_{\mathrm{seq},j}}
{\sum_j m_j}.
\]

This preserves the original global context-conditioning path.

A richer cross-attention path may be tested later, but it is not silently substituted into the faithful baseline.

## 45.3 Effective-depth conditioning is enabled in the baseline

The baseline includes \(e_d\).

A no-effective-depth-conditioning run is a required ablation when studying recurrence.

Physical block identity is already encoded by distinct block parameters; \(e_d\) identifies the effective application depth across repeated cycles.

---

# 46. Exact draft/refine attention masks

Flatten valid packed slots into attention positions \(u,v\).

Let:

- \(g(u)\): logical block ID,
- \(i(u)\): persistent carrier ID,
- \(q_u\): query eligibility,
- \(kv_u\): key/value eligibility.

## 46.1 Draft mask

A draft query \(u\) may attend to key/value \(v\) iff:

\[
q_u=1,
\qquad
kv_v=1,
\]

and either:

\[
g(v)=g(u)
\]

or

\[
g(v)<g(u).
\]

Therefore:

\[
M^{\mathrm{draft}}_{uv}
=
\begin{cases}
0,
&
q_u=1,\ kv_v=1,\
[g(v)=g(u)\ \lor\ g(v)<g(u)],
\\
-\infty,
&
\text{otherwise}.
\end{cases}
\]

Within a block, attention is fully bidirectional.

Across blocks, information flows only from earlier blocks to later blocks.

The causal unit is therefore the generated block, not the individual carrier site.

## 46.2 Refine mask

In refine mode:

\[
M^{\mathrm{refine}}_{uv}
=
\begin{cases}
0,
&
q_u=1,\ kv_v=1,
\\
-\infty,
&
\text{otherwise}.
\end{cases}
\]

Thus every valid query may read every active key/value across the packed carrier.

## 46.3 Inactive carriers

For an inactive carrier slot:

\[
q_u=1,
\qquad
kv_u=0.
\]

It may read active sites but may not contribute its current state as key/value context.

This is true in both draft and refine modes.

## 46.4 Invalid/padded queries

If

\[
q_u=0,
\]

the attention output for \(u\) is forced to zero before residual addition.

Padding must never acquire state through numerical softmax artifacts.

## 46.5 Causal leakage test

For fixed parameters, structure, inputs, and random keys:

- perturb a later block;
- in draft mode, all earlier block outputs must remain invariant within numerical tolerance;
- in refine mode, earlier blocks are allowed to change.

This is a hard architecture test, not merely a metric.

---

# 47. Adze-T stochastic-operator contract

## 47.1 Continuous learned transforms

Every major learned affine state transform in Adze-T uses an explicit reparameterized Torx Gaussian factor.

For input \(x\):

\[
\mu=W x+b.
\]

Each output channel owns a trainable raw scale \(\rho\).

Define:

\[
\sigma_{\mathrm{channel}}
=
\operatorname{clamp}
\left(
\operatorname{softplus}(\rho),
\sigma_{\min},
\sigma_{\max}
\right).
\]

A runtime master scale \(\lambda_{\mathrm{op}}\ge0\) gives:

\[
y
=
\mu
+
\lambda_{\mathrm{op}}
\sigma_{\mathrm{channel}}
\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

The stochastic scale parameters are tied whenever the corresponding mean transformation is tied.

There is no separate variance parameter per recurrence cycle.

## 47.2 Exact parity mode

Parity mode sets:

\[
\lambda_{\mathrm{op}}=0.
\]

Then the Torx factor executes its mean path exactly:

\[
y=\mu=Wx+b.
\]

This is the required deterministic limit.

No fake `log(0)` variance is used.

The factor/backend must expose a true zero-noise mean execution path.

## 47.3 Three independent stochasticity switches

The runtime configuration must expose at least:

```text
operator_stochasticity
diffusion_stochasticity
structure_sampling
```

They control different sources of randomness.

### Operator stochasticity

Noise inside Torx learned transforms:

\[
\epsilon_{\mathrm{op}}.
\]

### Diffusion stochasticity

Noise in content corruption / re-noising and SDE-like sampling:

\[
\epsilon_{\mathrm{diff}}.
\]

### Structural stochasticity

Sampling of discrete model states such as:

\[
b,\ell,
\]

and later selector/commit variables when enabled.

These switches must be independently controllable.

## 47.4 PRNG semantics

Parameter tying does not imply random-noise tying.

By default, a stochastic occurrence gets an independent key derived by folding in:

```text
global_seed
optimizer_step / eval_sample_id
batch-example identity
module identity
outer iteration r
denoise step s
recurrence cycle q
physical block l
stochastic-site identity
```

Thus two applications of the same tied \(W_Q\) at different \(q\) use the same mean parameters but different stochastic draws.

Explicit common-random-number experiments may override this only when declared as controls.

## 47.5 No automatic \(Q\)-noise normalization

The primary semantics do not divide operator variance by \(Q\).

Extra recurrence means extra stochastic computation.

A matched-total-variance experiment may exist as a separate control, but it must be labelled as a different stochastic process.

## 47.6 Required operator diagnostics

For each stochastic module family, record:

- mean activation RMS,
- sampled activation RMS,
- noise RMS,
- noise/activation ratio,
- learned \(\sigma\) distribution,
- fraction of channels at \(\sigma_{\min}\),
- fraction at \(\sigma_{\max}\),
- empirical sample mean error relative to deterministic mean,
- empirical covariance sanity checks on small cases.

Variance collapse or saturation must be reported rather than hidden.

---

# 48. Encoder / proposal / decoder operational interfaces

> **Conformance status:** the component topology is recovered Adze. Exact tensor widths and some slot-level constructions below are reference implementation choices rather than claims about the original endpoint.

All dimensions below are symbolic configuration values shared by both backends.

## 48.1 Shared byte frontend

Input:

```text
byte_ids    [B, N]       uint8 / categorical
byte_mask   [B, N]       bool
```

Output:

```text
byte_hidden [B, N, d_front]
```

The frontend contains the original shared byte lift / Router-1 / local Mamba-style processing.

Adze-D uses deterministic learned transforms.

Adze-T uses matched Torx transforms for learned state-changing operations.

## 48.2 Context encoder

Input:

```text
prompt_hidden [B, P, d_front]
prompt_mask   [B, P]
```

Output:

```text
context_seq   [B, P, d_ctx]
context_global[B, d_ctx]
```

with:

\[
context_{\mathrm{global}}
=
\operatorname{maskedmean}(context_{\mathrm{seq}}).
\]

The contextual sequence is retained for diagnostics/future richer conditioning even though the faithful DiT baseline uses the global pooled vector.

## 48.3 Clean target/carrier encoder

Input:

```text
target_hidden [B, T, d_front]
target_mask   [B, T]
```

Output:

```text
h0            [B, C, d_h]
b0            [B, C]        # final entry is forced end sentinel
l0            [B, C]        # values 0...L_max
```

The target-analysis path owns separate parameters after the shared frontend.

Its job is to produce the clean latent carrier target and generated-structure targets.

## 48.4 Proposal SSM

Generation begins from prompt context plus a carrier prior.

Input:

```text
context_global [B, d_ctx]
carrier_prior  [B, C, d_h]
carrier_pos    [C]
```

Output:

```text
h_prop          [B, C, d_h]
b_prop_logits   [B, C, n_b]
l_prop_logits   [B, C, L_max+1]
```

The final boundary entry is forced to the end sentinel.

A deterministic initial commit produces:

```text
c_b             [B, C]
a               [B, C]
```

for the first heavy-core trajectory.

The proposal must be materially cheaper than a full looped-DiT evaluation.

## 48.5 Heavy-core interface

Input:

```text
h               [B, C, d_h]
s_b             [B, C]
s_l             [B, C]
c_b             [B, C]
a               [B, C]

context_global  [B, d_ctx]
nu              [...]
mode            draft | refine
s_index         scalar / [B]
r_index         scalar / [B]
Q               static/runtime-bucketed integer
```

Output:

```text
h_updated        [B, C, d_h]
b_logits         [B, C, n_b]
l_logits         [B, C, L_max+1]
aux              trajectory diagnostics
```

## 48.6 Decoder

Given committed/final carrier:

```text
h_final          [B, C, d_h]
l_final          [B, C]
```

allocate fixed candidate output slots:

\[
[B,C,L_{\max},d_{\mathrm{dec}}].
\]

For carrier \(i\), slot \(j\) is emitted iff:

\[
j<\ell_i.
\]

Construct each decoder slot from:

\[
P_{\mathrm{dec}}h_i
+
e_{\mathrm{slot}}(j)
+
e_{\mathrm{carrier}}(i).
\]

Flatten in carrier order to:

```text
decoder_hidden   [B, C*L_max, d_dec]
decoder_mask     [B, C*L_max]
```

Run the original Mamba/SSM decoder topology.

Output:

```text
byte_logits / byte_distribution
[B, C, L_max, 256]
```

Emission compacts only valid slots into the final byte string.

Internal carrier deletion does not reindex carrier identities; it only changes which decoder slots are emitted.

---

# 49. Training objectives and gradient contract

## 49.1 Primary prediction target

The faithful diffusion baseline uses clean-state / \(x_0\)-prediction.

At sampled corruption level \(\nu\), the heavy core predicts:

\[
\hat h_0,
\qquad
p_\theta(b_0),
\qquad
p_\theta(\ell_0).
\]

## 49.2 Content loss

Use normalized carrier MSE:

\[
\mathcal L_h
=
\frac{1}{BCd_h}
\sum_{b,i,j}
\left(
\hat h_{0,b i j}
-
h_{0,b i j}
\right)^2.
\]

All persistent carrier sites are included by default, including sites whose clean \(\ell=0\), because inactive sites retain a meaningful latent/re-activation state.

A masking ablation may be run later.

## 49.3 Boundary loss

For non-sentinel boundary positions:

\[
\mathcal L_b
=
-\frac{1}{B(C-1)}
\sum_{b,i=0}^{C-2}
\log p_\theta(b_{0,b i}).
\]

The forced final boundary is excluded.

## 49.4 Extent loss

\[
\mathcal L_\ell
=
-\frac{1}{BC}
\sum_{b,i}
\log p_\theta(\ell_{0,b i}).
\]

## 49.5 Decoder byte loss

For clean emitted slots:

\[
\mathcal L_{\mathrm{byte}}
=
-\frac{1}{N_{\mathrm{byte}}}
\sum_{\mathrm{emitted\ slots}}
\log p_\theta(x_{\mathrm{byte}}).
\]

Padding and non-emitted slots are excluded.

## 49.6 Proposal losses

The proposal receives separate auxiliary losses against clean targets:

\[
\mathcal L_{\mathrm{prop},h},
\qquad
\mathcal L_{\mathrm{prop},b},
\qquad
\mathcal L_{\mathrm{prop},\ell}.
\]

These are reported separately.

They must not be allowed to dominate the heavy-core objective.

## 49.7 Total loss

The baseline training objective is:

\[
\mathcal L
=
\lambda_h\mathcal L_h
+
\lambda_b\mathcal L_b
+
\lambda_\ell\mathcal L_\ell
+
\lambda_{\mathrm{byte}}\mathcal L_{\mathrm{byte}}
+
\lambda_{\mathrm{prop}}\mathcal L_{\mathrm{prop}}.
\]

The \(\lambda\) values are fixed in configuration and shared between matched Adze-D and Adze-T runs.

No backend-specific rescue reweighting is allowed inside a declared matched comparison.

## 49.8 No intermediate \(Q\) supervision

There is no ground-truth supervision after each recurrent cycle.

Training loss is attached to the output of the full selected \(Q\)-cycle core evaluation.

Intermediate recurrent states may be:

- logged,
- probed,
- decoded by frozen/post-hoc probes,

but those probes do not train the core.

This prevents the experiment from teaching the model an artificial per-cycle algorithm.

## 49.9 Initial denoising curriculum

Before differentiating through long self-generated stochastic trajectories:

1. sample a clean target carrier;
2. sample one corruption level;
3. corrupt it with the known kernel;
4. run one full \(Q\)-cycle clean-state prediction;
5. apply analytic MSE/CE losses.

Only after this is trainable should training expand to:

- short self-generated \(S\) rollouts,
- full \(S\),
- sampled mutable structure,
- outer \(R\).

## 49.10 Continuous stochastic gradients

Reparameterized Gaussian Torx factors use ordinary pathwise JAX gradients.

No score bridge is added to purely reparameterized continuous paths.

## 49.11 Discrete structural gradients

There are two regimes.

### Prediction-only regime

If \(b,\ell\) are predicted and CE-supervised but their sampled values do not control any downstream computation in the same differentiable trajectory:

- use ordinary categorical NLL;
- do not use the score bridge.

### Sampled-downstream regime

Once sampled discrete \(b,\ell\) values causally affect later:

- packing,
- attention masks,
- activity,
- decoder structure,
- downstream losses,

the relevant sampled occurrences require the validated local score estimator / score bridge or another independently oracle-validated estimator.

The bridge is applied only to the descendant loss of the sampled discrete occurrence.

Do not double-count native Torx gradient routes.

The existing score-bridge implementation remains frozen unless an oracle-backed bug is found.

## 49.12 Hard commit boundaries

If structure is committed through deterministic argmax/hysteresis between trajectories, gradients stop through the commit decision unless a separately validated stochastic estimator is deliberately enabled.

The model still trains the pre-commit structural distribution with \(\mathcal L_b,\mathcal L_\ell\).

---

# 50. Deterministic-vs-Torx parity and fairness protocol

## 50.1 Mean-parameter equality

Every matched pair must have identical mean-model topology and matching initial mean parameters.

The canonical pairing is:

\[
\theta_T^{\mathrm{mean}}
\leftarrow
\theta_D.
\]

## 50.2 Matched quantities

A declared Adze-D vs Adze-T comparison holds fixed:

- data,
- train/validation splits,
- batch size,
- number of examples/tokens,
- \(C\),
- \(L_{\max}\),
- \(M_{\max}\) bucket,
- \(K\) bucket,
- \(d_h\),
- \(d_{\mathrm{model}}\),
- number of attention heads,
- FFN expansion,
- number of physical blocks \(L\),
- \(Q\),
- \(S\),
- \(R\),
- optimizer,
- learning-rate schedule,
- gradient clipping,
- loss weights,
- corruption schedule,
- draft/refine policy,
- selector policy,
- random-seed set where semantically possible.

Adze-T may contain extra variance parameters. Report their count separately.

Do not pretend total parameter count is exactly equal when those parameters are learned.

Mean-transformation parameter count must be equal.

## 50.3 Structural parity gate

Before model-quality comparisons:

- pack maps identical,
- block IDs identical,
- masks identical,
- unpool maps identical,
- commit semantics identical.

## 50.4 Near-zero operator-noise parity gate

With:

```text
operator_stochasticity = 0
structure_sampling     = 0
```

and identical external corruption:

\[
F_T(X)\approx F_D(X).
\]

Require module-level checks for:

- byte frontend,
- context encoder,
- target encoder,
- proposal,
- every Q/K/V/O projection,
- attention output,
- SwiGLU FFN,
- unpool,
- clean-state heads,
- decoder.

## 50.5 Finite-noise moment gate

For small fixed inputs and many Torx samples:

\[
\mathbb E[F_T(X)]
\]

must agree with the expected mean behavior implied by the stochastic operator composition to the tolerance appropriate for the nonlinear graph.

At individual affine factors, empirical mean and diagonal variance must match their analytic values.

## 50.6 Trainability gate

The full deterministic architecture must first solve the selected sanity task.

Then near-zero-noise Adze-T must match its optimization behavior closely enough to rule out a backend implementation defect.

Only then is finite stochasticity introduced.

## 50.7 Port gate

A trained Adze-D checkpoint is loaded into Adze-T.

At \(\lambda_{\mathrm{op}}=0\), task metrics must match.

Increase \(\lambda_{\mathrm{op}}\) on a predeclared schedule and measure:

- immediate degradation,
- fine-tuning recovery,
- learned variance behavior,
- stability.

## 50.8 Scratch gate

Adze-T(scratch) is evaluated separately from Adze-T(port).

Failure from scratch does not erase evidence that the stochastic substrate can express a ported working solution; likewise successful porting does not establish native trainability.

## 50.9 Recurrence evidence

Only after deterministic trainability, Torx parity, and finite-noise stability pass may \(Q\) be interpreted scientifically.

The strongest compute evidence is:

> the same trained weights, same input information, same noise policy, and larger \(Q_{\mathrm{test}}\) improve performance on harder examples.

Separate-trained-\(Q\) runs remain useful but have optimization confounds.

---

# 51. Required experiment switches

Every experiment configuration must explicitly record:

```text
backend:
    deterministic | torx

operator_stochasticity:
    off | on

diffusion_stochasticity:
    off | on

structure_sampling:
    off | on

mode:
    draft | refine

Q:
S:
R:

effective_depth_conditioning:
    off | on
```

No experiment should infer these from hidden defaults in a runner.

The serialized run record must contain them.

---

# 52. Deliberately unresolved / experimental choices

The following are **not** delegated to the implementation agent as “sensible defaults”. They remain explicit experimental/configuration choices.

## 52.1 Numeric model scale

Still configuration-level:

- \(C\),
- \(d_h\),
- \(d_{\mathrm{model}}\),
- \(d_{\mathrm{ctx}}\),
- number of heads,
- FFN ratio,
- \(K\) buckets,
- \(M_{\max}\) buckets,
- \(L_{\max}\),
- number of Mamba layers/state width.

The architecture contract does not depend on one particular scale.

## 52.2 Torx variance hyperparameters

The functional parameterization is fixed in section 47.

Still to choose per experiment:

- \(\sigma_{\min}\),
- \(\sigma_{\max}\),
- initialization of \(\rho\),
- master \(\lambda_{\mathrm{op}}\) schedule.

These values must be predeclared and common across matched \(Q\) runs.

## 52.3 Structure commit rule

Initial baseline may use deterministic hysteresis/argmax after a full inner trajectory.

The exact thresholds/inertia are configuration choices.

A stochastic commit kernel is a later measured variant.

## 52.4 Selector

The faithful architecture requires select/erase/refine semantics, but the exact selector feature set and calibrator are not frozen here.

The first implementation should reuse the simplest previously validated selector/control rather than inventing a large learned policy.

## 52.5 Adaptive compute

Adaptive:

- \(Q\),
- \(S\),
- \(\rho\),
- \(R\),

remain later milestones.

Fixed-compute correctness comes first.

## 52.6 Hardware-near substitutions

Not part of the baseline:

- local couplers replacing attention,
- eliminating hard pack,
- pbit-only bytes,
- replacing SwiGLU with local mixtures,
- alternative thermalized context networks.

Each requires a direct ablation against faithful Adze-T.

---

# 53. Additional hard implementation tests

Before any large run, add tests for the contracts above.

## 53.1 Boundary-to-block examples

Test:

- no internal cuts,
- cut after every site,
- inactive hole inside a block,
- consecutive inactive holes,
- cut adjacent to an inactive site,
- first/last active site inactive,
- forced terminal boundary.

## 53.2 Pack/unpool bijection

For random valid structure:

\[
\operatorname{unpack\_ids}(\operatorname{pack\_ids}(0,\ldots,C-1))
=
(0,\ldots,C-1).
\]

No carrier identity may be duplicated or lost.

## 53.3 Query/KV isolation

For inactive carrier \(i\):

- perturb \(h_i\);
- with fixed keys/randomness, other sites' outputs must not change through K/V use;
- carrier \(i\)'s own output may change because it remains a query;
- perturb active context and verify carrier \(i\) can respond.

## 53.4 Draft mask exactness

Construct a three-block synthetic pack and assert the allowed attention matrix exactly.

## 53.5 Refine mask exactness

Every valid query can read every active key/value; no invalid/padded key contributes.

## 53.6 Recurrence parameter tying

For each physical block \(\ell\):

\[
\theta_{\ell,q_1}
=
\theta_{\ell,q_2}
\]

by identity, not merely equal initialization.

Different physical blocks remain distinct.

## 53.7 Recurrence random-key independence

With Torx stochasticity enabled, repeated applications of a tied block must receive different default random keys.

## 53.8 Near-zero parity

With operator stochasticity disabled, deterministic and Torx block outputs must match to tolerance for the same weights and inputs.

## 53.9 First-step gradient gate

At initialization, a nontrivial supervised loss must yield finite, non-zero gradients in:

- output head,
- DiT output projection,
- DiT FFN,
- at least one Q/K/V projection,
- conditioning path,
- encoder/proposal path where connected.

This explicitly prevents recurrence of the M4.5 zero-core-gradient-at-step-one handicap.

---

# 54. Updated definition of the baseline

The faithful baseline is now operationally:

\[
\boxed{
\begin{aligned}
&\text{bytes}
\rightarrow
\text{shared Mamba/Router frontend}
\rightarrow
\text{context + clean-target encoders}
\\
&\rightarrow
\text{persistent }(h,b,\ell)\text{ carrier}
\rightarrow
\text{proposal / committed block structure}
\\
&\rightarrow
\text{hard }C\rightarrow M\times K\text{ pack}
\rightarrow
\left(B_L\cdots B_1\right)^Q
\\
&\rightarrow
\text{one-to-one unpool + carrier residual}
\rightarrow
x_0\text{-prediction of }(h,b,\ell)
\\
&\rightarrow
S\text{-step denoising}
\rightarrow
R\text{-step select/erase/global-refine}
\\
&\rightarrow
\text{Mamba byte decoder}.
\end{aligned}
}
\]

Adze-D and Adze-T execute this same graph.

Adze-T changes the implementation of learned state transformations to explicit stochastic Torx factors; it does not change what the model is.


---

# 55. Pass 1 — original-Adze conformance audit

## 55.1 Audit status and source limitation

This pass distinguishes the architectural facts we can recover from the preserved Adze project record from choices introduced while making the Torx port executable.

The exact original non-Torx **Adze Endpoint (v2)** markdown is not present in the currently accessible conversation/library file set. The available historical Adze-T design does, however, preserve several explicit sections labelled as the **original top-level diagram**, **original carrier/block-stream diagram**, and **original nested-loop diagram**, plus several statements about the original frontend and conditioning.

Therefore this pass is a **source-backed recovered conformance audit**, but not yet a literal line-by-line audit of the missing endpoint file.

This distinction is mandatory:

> **No `PROVISIONAL` choice below may be promoted to “original Adze” merely because it appears elsewhere in this document.**

When the original endpoint source is available, rerun this table against it and resolve every `PROVISIONAL` row before claiming exact architectural identity.

## 55.2 Conformance classes

Use exactly these labels in implementation notes and future design reviews:

- `ADZE_RECOVERED` — explicitly preserved as part of the original Adze architecture in the available project record.
- `TORX_TRANSLATION` — changes implementation substrate while preserving the recovered Adze computation/topology.
- `POST_ADZE_CORRECTION` — a later semantic correction from Adze experiments that should be carried forward, but is not claimed to have appeared in the earliest endpoint text.
- `PROVISIONAL` — an operational choice added to make the design concrete; it must be checked against the original endpoint or retained only as an explicit reference implementation choice.
- `FUTURE_ABLATION` — intentionally outside the faithful baseline.

## 55.3 Conformance matrix

| Component / decision | Class | Baseline treatment |
|---|---|---|
| Tokeniser-free byte interface | `ADZE_RECOVERED` | Keep |
| Mamba byte frontend | `ADZE_RECOVERED` | Keep |
| Shared low-level byte / Router-1 frontend between context and target-analysis paths | `ADZE_RECOVERED` | Keep |
| Separate inference-visible context encoder and training-only target/carrier encoder | `ADZE_RECOVERED` | Keep |
| Global prompt conditioning via original `mean-pool(c)` path | `ADZE_RECOVERED` | Keep for faithful baseline |
| SSM/Mamba-like bootstrap proposal | `ADZE_RECOVERED` | Keep |
| Persistent fixed-capacity carrier \(C\) | `ADZE_RECOVERED` | Keep |
| Carrier content \(h\), boundary \(b\), extent/length \(\ell\) | `ADZE_RECOVERED` | Keep |
| Generated structure rather than externally fixed token blocks | `ADZE_RECOVERED` | Keep |
| Hard pack from carrier to transient block stream | `ADZE_RECOVERED` | Keep |
| Transient \(M\times K\) block stream | `ADZE_RECOVERED` | Keep |
| Full attention as heavy-core communication | `ADZE_RECOVERED` | Keep |
| Transformer MLP / SwiGLU path | `ADZE_RECOVERED` | Keep |
| Residual pack -> DiT -> unpool -> carrier update | `ADZE_RECOVERED` | Keep |
| \(L\) distinct physical DiT blocks | `ADZE_RECOVERED` | Keep |
| Reuse physical stack for recurrent depth \(Q=12/L\) in original comparison | `ADZE_RECOVERED` | Keep |
| Separate denoising/resampling depth \(S\) | `ADZE_RECOVERED` | Keep |
| Separate outer refinement depth \(R\) | `ADZE_RECOVERED` | Keep |
| Draft computation before global refinement | `ADZE_RECOVERED` | Keep |
| Draft: same block bidirectional, cross-block causal | `ADZE_RECOVERED` | Keep |
| Refine: global/bidirectional cross-block communication | `ADZE_RECOVERED` | Keep |
| Same heavy-core weights in draft and refine modes | `ADZE_RECOVERED` | Keep |
| Layer / effective-depth conditioning in looped core | `ADZE_RECOVERED` | Keep |
| Noise/denoising conditioning | `ADZE_RECOVERED` | Keep |
| Outer-iteration conditioning | `ADZE_RECOVERED` | Keep |
| Mamba decoder | `ADZE_RECOVERED` | Keep |
| Zero extent means non-emitting | `ADZE_RECOVERED` | Keep |
| Observed \(s_b,s_\ell\) vs predicted \(p_b,p_\ell\) vs committed \(c_b,a\) structure | `POST_ADZE_CORRECTION` | Keep; fixes routing circularity |
| Query-active but key/value/pool-inactive inactive sites | `POST_ADZE_CORRECTION` | Keep unless original endpoint provides a stronger exact rule |
| Structure commits at controlled boundaries rather than every inner prediction | `POST_ADZE_CORRECTION` | Keep |
| Local score bridge only for causally downstream sampled discrete states lacking a native gradient route | `POST_ADZE_CORRECTION` | Keep |
| Deterministic learned affine -> Torx stochastic conditional with same mean map | `TORX_TRANSLATION` | Keep |
| Q/K/V/O learned projections become explicit Torx stochastic factors | `TORX_TRANSLATION` | Keep |
| FFN learned projections become explicit Torx stochastic factors | `TORX_TRANSLATION` | Keep |
| Mamba learned projections / stochastic transition kernels represented through public Torx factors | `TORX_TRANSLATION` | Keep |
| Deterministic softmax, masks, normalization, reshape, gather/scatter | `TORX_TRANSLATION` | Keep as exact algebra/glue |
| 256-state pdit as Torx byte representation | `TORX_TRANSLATION` | Reference default; not an Adze topology change |
| Exact boundary coordinate convention `cut after i` | `PROVISIONAL` | Keep as implementation convention unless source contradicts |
| Exactly one packed query slot per carrier site | `PROVISIONAL` | Do not call original-Adze fact |
| Exact meaning of \(K\) as max carrier sites per logical block | `PROVISIONAL` | Must be checked against original endpoint |
| Exact block overflow/bucketing policy | `PROVISIONAL` | Implementation policy |
| Exact packed feature sum \(P_hh + e_{\rm carrier}+e_{\rm block}+...\) | `PROVISIONAL` | Reference implementation only |
| RoPE on persistent carrier coordinate | `PROVISIONAL` | Reference implementation only |
| Exact pre-norm AdaLN modulation recipe | `PROVISIONAL` | Reference implementation only |
| Small non-zero residual-gate initialization | `PROVISIONAL` | Optimization default, not architecture claim |
| Per-output-channel learned Torx variance | `PROVISIONAL` / `TORX_TRANSLATION` | Reference stochastic parameterization |
| Exact decoder slot embedding formula | `PROVISIONAL` | Reference implementation only |
| Local/multiscale stochastic couplers replacing attention | `FUTURE_ABLATION` | Excluded from faithful baseline |
| Removing hard pack | `FUTURE_ABLATION` | Excluded |
| 8 pbits/byte in place of matched byte categorical representation | `FUTURE_ABLATION` | Later hardware-near comparison |
| Replacing SwiGLU/MLP with local mixtures | `FUTURE_ABLATION` | Excluded |
| Replacing Mamba encoder/decoder with a new stochastic lattice | `FUTURE_ABLATION` | Excluded |

## 55.4 Result of pass 1

The **macro-architecture is sufficiently recovered to freeze**:

```text
bytes
 -> shared Mamba/Router frontend
 -> context + target-analysis paths
 -> SSM proposal
 -> persistent h/b/l carrier
 -> generated hard pack
 -> M x K block stream
 -> real attention + MLP DiT stack
 -> repeated Q
 -> unpool + carrier residual
 -> x0 / structure heads
 -> S denoise
 -> draft/select/erase/global-refine x R
 -> Mamba byte decoder
```

The unresolved risk is now concentrated in **micro-semantics**, chiefly:

1. exact original \(M\times K\) packing meaning;
2. exact original DiT conditioning/modulation recipe;
3. exact original positional encoding;
4. exact original decoder slot construction.

Those four items are explicitly quarantined as `PROVISIONAL`. They are not allowed to trigger another architecture redesign.

---

# 56. Pass 2 — Torx public-API feasibility audit

## 56.1 Audit target

This audit is against the **project-pinned Torx revision**:

```text
f1fc858ed950ecd41935d15c06d0ec7c5e0674ae
```

as declared by the Adze-T repository dependency.

Do not silently update Torx merely because upstream `main` changes.

Any pin update is a separate dependency decision and must rerun the public-boundary tests.

## 56.2 Public API confirmed at the pin

The pinned top-level `torx` package publicly exports:

```text
AbstractFactor
AbstractReferenceFactor
DFG
DFGInfo
DFGParams
Site
ChainFactor
TiledFactor
DeterministicFactor
AbstractHasLogProbability
AbstractHasExplicitOutputDistribution
AbstractFiniteStateSpaceFactor
...
```

Therefore the faithful port does **not** require private `torx._...` access.

## 56.3 Factor contract

`AbstractFactor` exposes the required public semantic boundary:

```python
sample(key, inputs, params, info=None, site_info=None, return_aux=False)
init_params(key)
sample_with_references(...)
```

A directed factor represents:

\[
P(\mathrm{output}\mid\mathrm{inputs}).
\]

This is sufficient to implement explicit stochastic learned operators without wrapping the entire model in one opaque factor.

## 56.4 Q recurrence

`ChainFactor` directly supports:

- `n_steps`;
- feedback from one step to the next;
- `weight_tied=True`;
- `slice_info=True`;
- `jax.lax.scan` execution.

It also splits the supplied PRNG key across chain steps.

Therefore the intended:

\[
(B_L\cdots B_1)^Q
\]

weight-tied recurrent computation is directly expressible through public Torx composition.

**Verdict: `FEASIBLE / DIRECT`.**

## 56.5 Physical DiT stack

A `DFG` is itself a factor and may contain a DAG of explicit factor sites.

Site parameter addresses allow multiple sites to share a parameter entry when required.

A physical block can therefore be represented as a factor graph containing explicit stochastic learned transformations plus deterministic algebra nodes.

A stack of \(L\) distinct physical blocks can be one composite factor and then reused by a `ChainFactor` over \(Q\).

**Verdict: `FEASIBLE / DIRECT`.**

## 56.6 Randomness semantics

Pinned `ChainFactor` splits the supplied key across recurrence steps.

Pinned `TiledFactor` splits the supplied key across tiles.

Therefore:

- weight tying does not imply noise tying;
- independent recurrence/attention-site draws are natural;
- reproducible common-random-number controls remain possible by deliberately supplying controlled keys.

**Verdict: `FEASIBLE / DIRECT`.**

## 56.7 Stochastic affine / Gaussian operators

The pinned public package does not need to provide a special built-in `Linear` or `Normal` layer for this design.

A project-defined factor can subclass public `AbstractFactor` / `AbstractReferenceFactor` and implement:

\[
y = Wx+b+\sigma\epsilon.
\]

The Adze-T project already established this pattern with its public `AffineGaussianGate` experiments.

The important rule is that this factor remains **small and semantically local**: it implements one stochastic learned transformation, not a hidden Transformer.

**Verdict: `FEASIBLE / CUSTOM PUBLIC FACTOR`.**

## 56.8 Attention

Faithful stochastic attention should be decomposed explicitly:

```text
Torx Q projection
Torx K projection
Torx V projection
    |
deterministic RoPE/mask/QK^T/softmax/AV
    |
Torx O projection
```

The deterministic middle is exact algebra and may be:

- ordinary JAX plumbing in the architecture code, or
- explicit `DeterministicFactor` nodes if graph-level visibility is useful.

No opaque Transformer factor is required.

**Verdict: `FEASIBLE`.**

## 56.9 SwiGLU / MLP

Likewise:

```text
Torx up projection
Torx gate projection
deterministic SiLU and product
Torx down projection
```

is directly compatible with the factor contract.

**Verdict: `FEASIBLE`.**

## 56.10 Mamba / selective SSM

Mamba is the highest-risk porting component but there is no architectural blocker.

The faithful rule is:

- keep the selective state-space recurrence;
- keep input-dependent transition/gating semantics;
- express the learned maps that generate those quantities as explicit Torx factors;
- where the state transition itself is stochastic, make it an explicit small transition factor;
- use scan/chain composition for recurrence rather than hiding a complete Mamba network in `Factor.sample()`.

A sequence scan can use explicit per-step runtime information / sliced data while recurrent state is fed back.

Prototype this component independently before wiring the full encoder/decoder.

**Verdict: `FEASIBLE, IMPLEMENTATION-RISK HIGH`.**

## 56.11 Discrete \(b,\ell\) factors

The public API exposes finite/enumerable/log-probability factor abstractions.

Therefore explicit categorical boundary/extent factors can be implemented using public Torx interfaces even if the exact desired generic categorical convenience factor is not supplied at the pinned revision.

For sampled discrete states that affect downstream losses, use only the already oracle-validated project gradient route.

**Verdict: `FEASIBLE / CUSTOM PUBLIC FACTOR`.**

## 56.12 Hard pack / unpool

Torx does not need to own deterministic indexing.

Hard pack and unpool may remain JAX functions around the stochastic core.

For JIT/static-shape execution:

- bucket \(M\) and \(K\),
- pad,
- mask unused positions.

This preserves semantic hard packing without requiring dynamically shaped JAX arrays inside a compiled stochastic factor.

**Verdict: `FEASIBLE`.**

## 56.13 Nested \(Q/S/R\)

A DFG is itself a factor and `ChainFactor` can chain factors with feedback.

However, the clean implementation boundary should remain:

- \(Q\): Torx `ChainFactor` over the physical DiT stack;
- \(S\): model-level scan/chain over clean-predict -> known re-corruption;
- \(R\): model-level scan/loop over select -> reset -> global denoise -> commit.

Do not force every outer control decision into one giant DFG merely for aesthetic purity.

The **learned stochastic computation** must be Torx; deterministic orchestration may remain explicit JAX/model logic.

**Verdict: `FEASIBLE`.**

## 56.14 Zero-noise deterministic parity

A custom stochastic factor can expose a true mean-path execution when operator stochasticity is disabled.

This makes:

\[
\mathrm{Adze\mbox{-}T}(\lambda_{\rm op}=0)
=
\mathrm{Adze\mbox{-}D}
\]

an implementation invariant for matched learned operators.

No Torx API change is required.

**Verdict: `FEASIBLE`.**

## 56.15 Feasibility conclusion

There is no currently identified Torx API limitation that justifies changing the recovered Adze topology.

The expected implementation work is:

1. explicit small stochastic operator factors;
2. composition through public DFG/Chain/Tiled abstractions;
3. deterministic JAX algebra for exact glue;
4. project-local training estimators where already validated.

The main engineering risk is **stochastic selective-SSM/Mamba implementation**, not attention, recurrent DiT, packing, or nested loops.

Therefore:

\[
\boxed{
\text{TORX\_PORT\_FEASIBLE}
}
\]

with the condition that the Mamba port passes its own deterministic-limit and trainability gates before full integration.

---

# 57. Pass 3 — frozen first reference configuration

## 57.1 Purpose

This configuration is **not claimed to be the original Adze scale**.

It is the first implementation/reference configuration used to establish:

- deterministic architectural correctness;
- deterministic/Torx parity;
- stochastic trainability;
- real looped-Transformer \(Q\) behavior.

It should be small enough for fast iteration and large enough that the heavy core is a genuine structured Transformer rather than another toy affine proxy.

Name:

```text
adze_reference_small_v0
```

## 57.2 Reference dimensions

```yaml
io:
  byte_vocab: 256
  prompt_max_bytes: 128
  target_max_bytes: 128

carrier:
  C: 32
  h_dim: 64
  L_max: 4

frontend:
  d_front: 64
  layers: 2

context_encoder:
  d_ctx: 128
  layers: 2

target_encoder:
  layers: 2

proposal:
  layers: 2
  hidden_dim: 64

packing:
  M_max: 32
  K: 8
  overflow_policy: explicit_error_or_larger_bucket

dit:
  d_model: 128
  heads: 4
  head_dim: 32
  ffn_hidden: 256
  physical_blocks_L: 4
  cycles_Q: 3
  effective_applications: 12
  draft_mask: block_causal
  refine_mask: global
  effective_depth_conditioning: true

decoder:
  d_dec: 128
  layers: 2

denoise:
  S_initial: 1

refinement:
  R_initial: 0
```

### Important packing qualification

`K=8` is an execution bucket for this first reference configuration.

Until the original endpoint's exact \(K\) semantics are recovered, do not treat `K=8` or the section-43 one-slot mapping as architectural truth.

For initial tests, the synthetic/reference structure generator must produce blocks that fit the chosen bucket. Overflow is an explicit error, never truncation.

## 57.3 Initial stochastic configuration

```yaml
stochasticity:
  operator_stochasticity: false
  diffusion_stochasticity: controllable
  structure_sampling: false

torx_operator:
  variance_parameterization: per_output_channel_diagonal
  sigma_min: 1.0e-6
  sigma_max: 0.25
  initial_sigma: 1.0e-3
  master_scale_lambda_op: 0.0
```

These noise values are **reference defaults**, not architectural commitments.

The first finite-noise sweep should use predeclared:

\[
\lambda_{\rm op}
\in
\{0,\ 0.1,\ 0.25,\ 0.5,\ 1.0\}.
\]

If this parameterization proves numerically poor, change it as a Torx backend experiment without changing the architecture.

## 57.4 Training defaults

```yaml
training:
  optimizer: adamw
  learning_rate: 3.0e-4
  weight_decay: 0.01
  grad_clip_norm: 1.0
  batch_size: 32

  objective:
    h_weight: 1.0
    boundary_weight: 1.0
    extent_weight: 1.0
    byte_weight: 1.0
    proposal_weight: 0.25

  validation_every_steps: 100
  seeds: [0, 1, 2]
```

These are common defaults for the first matched D/T runs.

Do not tune Adze-D and Adze-T separately before the first comparison.

If a common setting is changed, rerun both arms.

---

# 58. Frozen build order and gates

## 58.0 Record the methodological reset

Before new architecture experiments, add a repository decision record stating:

- M1-M4.5 remain valid as Torx substrate/proxy experiments;
- they are not evidence against the original Adze looped-DiT recurrence;
- the new baseline is one architecture with deterministic and Torx backends;
- hardware-near replacements are future ablations.

No historical result files are rewritten.

## 58.1 Phase A — shared architecture types and deterministic indexing

Implement only:

- carrier datatypes;
- observed/predicted/committed structural state;
- corruption datatypes;
- boundary -> logical block mapping;
- hard pack metadata;
- exact masks;
- unpool;
- reference configuration.

No trainable neural model required yet.

### Gate A

All section-53 structural, pack/unpool, inactive-site, and mask tests pass.

## 58.2 Phase B — Adze-D deterministic reference

Implement deterministic:

- shared byte frontend;
- context encoder;
- target encoder;
- proposal;
- genuine DiT block;
- \(L=4,Q=3\) loop;
- clean-state heads;
- Mamba decoder.

Start with:

\[
S=1,\qquad R=0,
\]

and fixed/teacher committed structure where useful for isolation.

### Gate B1 — first-step gradient

Every major connected learned family has finite, non-zero gradient at optimizer step 1.

### Gate B2 — overfit

The model can:

1. memorize one example;
2. strongly overfit a fixed mini-dataset;
3. fit a small generated training set.

### Gate B3 — held-out sanity

It learns at least copy/reverse and one simple algorithmic task above strong baselines.

Failure here is:

```text
ADZE_D_CORE_TRAINABILITY_FAILURE
```

not evidence about Torx.

## 58.3 Phase C — Torx operator library and zero-noise parity

Implement public-API factors for:

- stochastic affine Gaussian transform;
- categorical structure;
- stochastic SSM transition as required.

Wire the exact same architecture through the Torx backend.

Set:

\[
\lambda_{\rm op}=0.
\]

### Gate C

For copied mean parameters:

- per-module parity passes;
- full-core parity passes;
- encoder parity passes;
- decoder parity passes;
- task metrics match deterministic Adze within numerical tolerance.

Failure:

```text
TORX_PARITY_FAILURE
```

No finite stochastic experiment may proceed.

## 58.4 Phase D — finite stochasticity

Use the same trained/copied mean model and increase \(\lambda_{\rm op}\).

Measure:

- immediate metric delta;
- activation/noise ratio;
- trajectory variance;
- learned \(\sigma\);
- fine-tuning recovery.

Then train Adze-T from scratch using the same reference configuration.

### Gate D

At least one non-zero stochastic setting remains trainable and materially above task baselines without variance collapse or numerical instability.

Failure:

```text
TORX_STOCHASTIC_TRAINABILITY_UNRESOLVED
```

This is a backend/training result, not a recurrence result.

## 58.5 Phase E — real recurrent \(Q\)

Only now study recurrence in the actual DiT.

### E1 — effective-depth factorization

Compare matched effective depth:

\[
(12,1),\ (6,2),\ (4,3),\ (3,4),\ (2,6),\ (1,12)
\]

where computationally practical.

No \(1/Q\) residual scaling.

### E2 — same-model test-time recurrence

For a trained recurrent model, evaluate identical examples under:

\[
Q_{\rm test}=1,2,3,4,6,8,12
\]

where shape/control semantics permit.

### E3 — difficulty x Q

Use tasks with controlled causal/reasoning depth.

The key evidence is whether harder examples benefit from larger \(Q_{\rm test}\) using the same weights and same input information.

### E4 — overthinking

Evaluate beyond the trained/useful depth.

Report plateau and degradation rather than selecting only the best \(Q\).

Only Phase E may issue a scientific recurrence decision.

## 58.6 Phase F — denoising depth \(S\)

Restore clean-predict -> re-corrupt trajectories.

Keep \(Q\) fixed while varying \(S\).

Compare deterministic diffusion path and stochastic re-noising separately from operator stochasticity.

## 58.7 Phase G — generated \(b,\ell\) structure

Start with prediction-only structure.

Then enable controlled commit between trajectories.

Only after that enable sampled discrete structure with downstream causal effects and the validated score-gradient route.

Required gates:

- boundary recovery;
- extent recovery;
- internal deletion;
- reactivation;
- no illegal routing from `UNKNOWN`;
- block churn measurement.

## 58.8 Phase H — draft and global refinement \(R\)

Run:

```text
draft causal denoise x S
 -> select
 -> erase/reset
 -> global denoise x S
 -> commit
 -> repeat x R
```

Use the same heavy-core weights.

Re-run the controlled downstream-context intervention on this faithful architecture.

Required metrics:

- selected-region baseline error;
- gain after refinement;
- collateral change outside target region;
- preserved non-target state;
- selector calibration;
- gain by refinement iteration.

## 58.9 Phase I — natural byte generation

Only after A-H are stable:

- longer byte sequences;
- natural-text training;
- quality vs \(Q,S,R\);
- deterministic vs Torx quality/compute;
- eventual hardware-near ablations.

---

# 59. Decision vocabulary after the reset

Use decision labels that separate architecture, backend, and recurrence questions.

```text
ADZE_D_CORE_PASS
ADZE_D_CORE_TRAINABILITY_FAILURE

TORX_PARITY_PASS
TORX_PARITY_FAILURE

TORX_STOCHASTIC_TRAINABILITY_PASS
TORX_STOCHASTIC_TRAINABILITY_UNRESOLVED

ADZE_Q_BENEFIT
ADZE_Q_NEUTRAL
ADZE_Q_NEGATIVE
ADZE_Q_OPTIMIZATION_UNRESOLVED

ADZE_S_BENEFIT
ADZE_S_NEUTRAL
ADZE_S_NEGATIVE

ADZE_REFINEMENT_BENEFIT
ADZE_REFINEMENT_NEUTRAL
ADZE_REFINEMENT_NEGATIVE

BLOCKED
```

A \(Q\) verdict is forbidden unless:

1. Adze-D trainability passed;
2. Torx zero-noise parity passed for a Torx \(Q\) claim;
3. the actual Transformer/DiT core is used;
4. the task is sufficiently depth-sensitive;
5. the comparison does not change input information;
6. common optimization budgets are reasonably converged.

---

# 60. Final status after the three passes

The project is now frozen around:

\[
\boxed{\text{one Adze architecture, two computational backends}}
\]

and the current public Torx API at the project pin presents no known architectural blocker to a faithful port.

The remaining source-level caveat is narrow and explicit:

> The exact original non-Torx Endpoint markdown is still required to convert the four `PROVISIONAL` micro-semantics in section 55.4 into either confirmed Adze behavior or explicit reference defaults.

That caveat does **not** reopen the macro-architecture.

In particular, it does not justify removing or replacing:

- Mamba,
- hard pack,
- the \(M\times K\) block stream,
- attention,
- SwiGLU/MLP,
- the looped DiT,
- \(Q\),
- \(S\),
- generated \(b,\ell\),
- draft/global-refine,
- \(R\),
- the byte decoder.

Those are frozen.
