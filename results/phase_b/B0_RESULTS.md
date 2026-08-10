# B0 — shared backend/interface skeleton

Status: **B0_INTERFACE_PASS**.

- Added typed reference/model/training configuration loading.
- Added explicit committed carrier extent with activity consistency checks.
- Implemented the deterministic learned-operator interface and left `TorxOps`
  as a Phase C placeholder.
- The high-level model is a single topology with deterministic parameters
  arranged for later mean-parameter transfer.

Tests: deterministic operator construction and YAML/reference configuration
loading pass. No Torx code or private Torx import was added.
