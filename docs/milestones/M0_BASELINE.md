# M0 — Repository and public-API baseline

## Goal

Make the repo reproducible and enforce the public dependency boundary.

## Deliverables

- dependency pin recorded;
- public Torx import contract test;
- CI;
- Ruff/Pyright/Pytest;
- boundary scanner;
- M1 experiment directory;
- no model implementation.

## Gate

`PASS` when a fresh environment can run the scaffold checks and the pinned public Torx surface is inspectable without private imports.
