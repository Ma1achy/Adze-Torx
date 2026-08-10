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

## Initial task boundary

Implement **Phase A only**.

Allowed files for substantive implementation:

```text
src/adze_t/state.py
src/adze_t/packing.py
src/adze_t/masking.py
src/adze_t/unpool.py
src/adze_t/config.py
tests/test_state.py
tests/test_blocks.py
tests/test_pack_unpool.py
tests/test_masks.py
tests/test_inactive_sites.py
tests/test_reference_config.py
```

Do not implement:

- encoder;
- proposal network;
- DiT;
- Mamba;
- decoder;
- training loop;
- Torx backend;
- `S`;
- `R`.

Those modules are placeholders only.

## Phase A gate

Before stopping:

1. run the complete test suite;
2. all Phase A tests must pass;
3. pack/unpool carrier identities must be bijective on valid sites;
4. inactive sites must remain query-active but key/value/pool-inactive;
5. draft masks must prevent later-block -> earlier-block influence;
6. refine masks must permit global active K/V access;
7. overflow must be explicit, never truncation;
8. report exactly what changed;
9. STOP.

Do not continue to Phase B unless explicitly instructed.
