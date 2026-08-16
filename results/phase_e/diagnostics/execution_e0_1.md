# Phase E.0.1 — execution diagnosis

## Root cause

The Phase-E runner did not crash. The command tool yielded before JAX work
completed, which had been misread as process termination. Three duplicate
jobs were subsequently found active:

| PID | Command | Elapsed at audit | RSS at audit |
|---:|---|---:|---:|
| 4583 | `run_phase_e.py --stage e1 --max-steps 500` | 01:32:34 | 1.24 GiB |
| 16335 | `run_phase_e.py --stage e1 --max-steps 500` | 01:30:31 | 1.75 GiB |
| 19818 | `run_phase_e.py --stage e1 --max-steps 1` | 01:29:54 | 1.55 GiB |

They were confirmed duplicate Phase-E jobs and terminated with `SIGTERM`.
No unrelated process was killed. Before termination, macOS reported 109 MiB
free physical memory and 9.98 GiB used swap; after it, free pages rose to
about 7.6 GiB. The host backend is `CpuDevice(id=0)`.

## Minimal reproducer

`scripts/diagnose_phase_e_training.py` ran E_Q1/COPY/lambda=1 with the clean
teacher and fixed structure. Batch-1 eager forward/backward, JIT forward,
JIT loss, JIT value-and-grad, and JIT optimizer step all exited zero. The
batch-32 JIT optimizer step also exited zero with loss `9.082194`.

A Q=3 Phase-D-compatible optimizer-step control likewise exited zero with
loss `10.497722`. Thus the problem was not a Phase-E model path regression.

## Fix

The ordinary training path now defaults to no diagnostic capture: it neither
materializes attention weights or diagnostic summaries nor selects
intervention/depth-override paths. Those features remain available only for
diagnostics. Primary E2 is run serially, with live process checks, to prevent
concurrent JAX compilations from exhausting host memory.
