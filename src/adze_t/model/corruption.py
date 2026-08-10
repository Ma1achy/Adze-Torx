"""Carrier corruption processes."""


def corrupt_content(*args, **kwargs):
    raise NotImplementedError("M3: continuous carrier corruption not implemented")


def corrupt_structure(*args, **kwargs):
    """Absorbing corruption for boundary/length state, including UNKNOWN."""
    raise NotImplementedError("M3: structural corruption not implemented")
