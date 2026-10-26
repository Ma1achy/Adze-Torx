# G.0 validation record

## Focused validation

At source `94a2416`:

```sh
PYTHONPATH=src pytest tests/test_phase_g_structure.py tests/test_phase_g_oracle.py tests/test_phase_g0.py -m 'not slow' -ra
```

59 passed, 21 deselected, 26.23 seconds. The deselected legacy MC tests are
separate from the executed new production-policy 21-case MC run.

Formatting: 138 files already formatted. Ruff: passed. Pyright: zero errors,
warnings or information messages. Public dependency boundaries: passed.
Private `torx._...` imports: none introduced; explicit scan clean.

The new compiled full-model test initially captured the batch as a JAX
constant, exposing an eager validation/tracing mismatch in the historical
corruption wrapper. Passing the batch as the intended dynamic JIT argument
fixes the test without modifying the frozen corruption protocol.
One new identity test initially omitted the required `C` argument to unpool;
corrected and rerun. Type-only optional/shape guards were fixed before the
final focused run. No numerical training failure occurred.

## Checkpoint audit

Pure NumPy/hash inspection (no optimizer execution) found:

| Step | Parameters finite | Adam moments finite | Nonzero moment elements |
| --- | --- | --- | ---: |
| 0 | yes | yes | 0 |
| 100 | yes | yes | 1240846 |
| 250 | yes | yes | 1240846 |

All checkpoint protocols agree. Frozen parameters equal selected checkpoint
parameters bitwise. Parameter count: 636807. Joint target codec, not
decoder-only. Checkpoints and original accepted artifacts remain preserved.

## Full suites

On immutable source `94a2416e4e6056d4a6421f178ca7e0f738bcb486`:

```sh
PYTHONPATH=src make test
```

245 passed, 1 skipped, 30 deselected in 1478.93 seconds (24:38).
The skip is the existing `tests/test_phase_d_training.py` fixture guard:
accepted Phase-B COPY/REVERSE checkpoint fixtures are absent in this isolated
checkout. They were not copied or substituted to force that test to run.

```sh
PYTHONPATH=src make test-slow
```

30 passed, 246 deselected in 60.21 seconds (1:00). This ran sequentially only
after regular succeeded. Combined full-suite pytest time: 1539.14 seconds.

Final tracked source still equals the tested source; subsequent edits contain
only G.0 reports and evidence indexing. No full suite was skipped. The one
fixture skip above is disclosed, not counted as a pass.

No G policy arm, G campaign, H experiment, full-byte or LM training is part of
this validation. Historical F science and the original G failed gate are not
rewritten.
