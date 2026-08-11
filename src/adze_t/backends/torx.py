"""Torx stochastic learned-operator backend.

Phase C placeholder.

Rules:
- public Torx API only;
- no torx._ imports;
- do not hide a deterministic Transformer/Mamba model inside one Factor.
"""


class TorxOps:
    def __getattr__(self, name):
        raise NotImplementedError(f"TorxOps.{name} is deferred to Phase C")
