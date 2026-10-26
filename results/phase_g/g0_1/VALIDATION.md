# G.0.1 validation and scope

Source for diagnosis/training: `c80920e`; candidate selection and protected test
predate the audit/report commits. `PROTOCOL.md` has not changed since that source.

```sh
PYTHONPATH=src pytest tests/test_phase_g01.py -ra
```

4 passed in15.49s. Tests cover equal present-class weighting despite duplicated
inactive sites, rejection of aggregate-only partial recovery, probe input/shape
contracts, finite compiled codec update, bitwise non-readout preservation in
head-only mode, and decoder inference without teacher-length arguments.

```sh
PYTHONPATH=src pytest tests/test_phase_g01.py tests/test_phase_g0.py tests/test_phase_g_structure.py tests/test_phase_g_oracle.py -ra
```

At source `c59581cdb749e5d4c1a90af152fe07286cdce220`, 84 passed in 63.86s.
No skips/deselection in this run. It includes the preserved
exact score-gradient and21 Monte Carlo tests, dynamic capacity eager/JIT/scan/
vmap/identity/inactive cases, and the real-model dynamic policy regression.

```sh
make format-check
make lint
make typecheck
make boundaries
rg -n 'torx\._|from torx\._' src scripts tests
git diff --check
```

Format:143 files already formatted. Ruff pass. Pyright zero errors/warnings.
Boundary scan pass. Explicit private import scan no matches. Diff-check pass.
A type annotation in the supplemental audit was corrected before its execution;
no candidate weights, criteria, action probabilities or scientific data changed.

The whole-project regular/slow suites were NOT repeated in this increment:
only new modules, scripts, tests and versioned evidence were added; no existing
engine or sampling source changed. The base's previous full validation remains
245 regular passed/1 disclosed fixture skip and30 slow passed. This is not
represented as a fresh whole-project pass. Full integration validation remains
required before any future G merge/scientific execution under the original gates.

Scientific execution: diagnostic probes (fixed800 updates each), then100 joint
target-codec updates with fresh moments. Every optimizer occurrence checked
finite loss/norm/parameters/moments before persistence. No numerical failure or
hyperparameter adjustment. No G policy arm/model training, H/I/LM training or S4.

The host denied the requested nice adjustment; execution continued at its
ordinary priority. No duplicate ML runner was active; workloads ran sequentially.
No unrelated process was terminated or modified.

Archived all13 binary artifacts with independent destination hash checks;
frozen candidate is read-only. G.0/F1 hashes rechecked and unchanged, F3 files
untouched, tracked main/H/I remain unchanged. No merge/rebase/push was performed.
