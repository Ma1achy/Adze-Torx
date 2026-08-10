"""Context and training-only carrier encoders."""


def build_context_encoder(*args, **kwargs):
    """M2+: construct prompt conditioning. Exact architecture is milestone-gated."""
    raise NotImplementedError("M2: context encoder not implemented")


def build_carrier_encoder(*args, **kwargs):
    """M2+: training-only target-to-carrier encoder."""
    raise NotImplementedError("M2: carrier encoder not implemented")
