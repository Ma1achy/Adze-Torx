# G23 V0 — provenance and blocking-result reproduction

No scientific training is needed or justified to reproduce the saved blocker.
Do not reroll or extend the failed finite-MC gate to obtain a pass. Preserve the
saved roots and every component, including the nonpass.

## Immutable execution sequence

1. `6e68e0f688261ec954ba718f9c7e8da0fa59ef47`: implementation, G2/G3 protocols and
   regression tests committed prospectively.
2. `10aec599e4ce08244e297d752c19ee5a770179fd`: full execution identity and historical
   preservation baseline committed, then the only preflight launched:

   ```sh
   nice -n 10 env PYTHONPATH=src:scripts python scripts/run_phase_g23.py --stage preflight
   ```

   `nice` could not change priority in this sandbox; Python nevertheless ran at
   normal priority. This did not alter the scientific configuration. The launch
   record is `../g2/PREFLIGHT_LAUNCH.json`. Preprocessing froze/checked train and
   validation observations; the new oracle failed before parent-model gradient
   or trial-update gates. Exit status was nonzero; traceback is preserved in
   `../g2/PREFLIGHT_FAILURE.json`.
3. `08b1299d5e75ebcdc788e420ba93b348f70e087c`: raw oracle evidence plus diagnostic
   source committed before same-draw diagnosis:

   ```sh
   PYTHONPATH=src:scripts python scripts/diagnose_phase_g23_oracle.py
   ```

   This enumerated actions and reconstructed the original draws; it did not
   generate new root identities or repeat confirmation until a pass. The
   current script has a later static-type-only cast; the above commit is the
   exact source of the saved diagnosis. Evidence records that distinction.

The frozen identity includes Python/JAX/JAXlib/NumPy/Torx/backend/PRNG settings,
configuration, loss/update-mask identity, datasets and all then-existing
`src/adze_t` and `scripts` Python files. Its binding is in
`../g2/G2_G3_PROTOCOL_IDENTITY.json`; do not silently refreeze it. The independent
diagnosis was added later and does not replace that source identity.

## Engineering validation (not scientific training)

```sh
PYTHONPATH=src:scripts make format-check lint typecheck boundaries
PYTHONPATH=src:scripts make test
PYTHONPATH=src:scripts make test-slow
PYTHONPATH=src:scripts pytest -m 'not slow' tests/test_phase_g2.py tests/test_phase_g23_statistics.py tests/test_phase_g0.py
```

Commands were sequential for JAX workloads. See `VALIDATION.json` and retained
logs for actual outcomes and skips; a prior static invocation without the
explicit import path is retained rather than hidden. A separate private-Torx
scan includes scripts as well as repository boundary-check roots.

## Recovery and next work

There is no new official optimizer state to resume: only a launch record, empty
lock file and failure record exist in `results/runs/phase_g/g23/v0/`. The immutable
warm-start remains accepted G11 teacher100 (path/hash in `CLOSURE.json`), not a
selected G2 policy. No protected-test construction, selection or test-opening
record exists. Do not treat the accepted teacher checkpoint as generated-policy
evidence.

The failed V0 statistic is not repaired by its plausible rare-event explanation.
A future prospectively versioned uncertainty/variance-control validation design
must preserve the failure and calibrate its rule independently. It is not part
of this completed blocking-outcome record. Main/H/I/Thermo are unchanged; no
dependent training, merge or push was performed.
