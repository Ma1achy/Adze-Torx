# Agent contract

## Authority

`docs/architecture/adze-architecture-v3.md` is the architecture contract.

Do not redesign the architecture.

## Non-negotiable architecture

Preserve:

- byte-level/tokeniser-free interface;
- shared low-level byte frontend and split context/target-analysis paths;
- persistent `C`-site carrier;
- `h`, `b`, `l`;
- generated block structure;
- hard pack;
- transient `M x K` block stream;
- full Transformer/DiT attention;
- Transformer MLP/SwiGLU path;
- `L` distinct physical blocks tied across recurrent cycles;
- `Q` recurrent compute;
- pack -> DiT -> unpool -> carrier residual;
- draft causal / refine global communication;
- separate `S` denoising loop;
- separate `R` outer refinement loop;
- Mamba/SSM frontend/proposal/decoder topology;
- one architecture with deterministic and Torx backends.

Do not replace any of the above because another design seems more "Torx-native".

## Torx rules

- pinned revision: `f1fc858ed950ecd41935d15c06d0ec7c5e0674ae`;
- public API only;
- no `torx._...`;
- explicit JAX PRNG;
- no Python/NumPy RNG in stochastic execution;
- learned heavy state transforms must be explicit Torx factors in Adze-T;
- deterministic algebra/glue may remain JAX;
- never hide a deterministic Transformer/Mamba model inside one custom factor;
- do not modify the validated score bridge absent an oracle-backed bug.

## Milestone boundary

Follow the latest explicit user-approved milestone instruction.  Treat the
architecture contract and committed milestone result records as frozen unless
that instruction explicitly authorizes a change.  Do not infer permission to
advance experimental science or redesign the architecture from the repository
state alone.

For Phase-D work, preserve the accepted baseline and consult
`results/phase_d/` and any subsequent versioned evidence directory before
changing evaluation, training, or decision semantics.  Record corrections as
new, versioned evidence; never silently rewrite historical scientific records.
