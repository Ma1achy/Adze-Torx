# C3 — full-model zero-noise parity

Configuration: `adze_reference_small_v0`, `L=4`, `Q=3`, `S=1`, `R=0`,
operator stochasticity off, structure sampling off, `lambda_op=0`.

Initialized-weight Adze-D and Adze-T use the same architecture graph and copied
mean parameters. Ordered comparison runs from prompt/target frontend through
context, target analysis, proposal, packed input, `q0/b0 ... q2/b3`, unpool,
clean heads, and decoder. The automatic first-divergence result is `None`.

Results:

- eager full forward: exact;
- JIT full forward: exact against JIT Adze-D;
- draft and refine recurrent traces: exact;
- integer/bool structure and pack metadata: exact;
- Phase-B total/component losses: exact;
- all 237 mapped raw mean-gradient leaves: exact;
- maximum rho-gradient magnitude: `0`;
- different root keys at zero noise: identical;
- extreme finite rho changes at zero noise: identical;
- valid byte `0x00`, reverse-style input, and explicit padding: passed.

The committed `target_codec_b1.pkl`, `copy.pkl`, and `reverse.pkl` artifacts
were present and reproducibly loadable. Their full forward traces also matched
exactly. These checkpoint checks are additional evidence, not prerequisites.

Machine evidence: `parity/full_model.json` and `parity/gradients.json`.
Worst forward, module, and mapped-gradient absolute errors are all `0`.

The optional optimizer-step comparison was not added: raw mathematical
gradient parity is exact, and introducing optimizer-container conversion would
not strengthen the required backend computation result.
