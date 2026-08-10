"""Explicit Q/S/R loop composition.

Keep core recurrence (Q), denoising time (S), and outer refinement (R) separate
in code and metrics.
"""


def apply_core_cycles(*args, **kwargs):
    raise NotImplementedError("M4: Q loop not implemented")


def apply_denoise_steps(*args, **kwargs):
    raise NotImplementedError("M5: S loop not implemented")


def apply_refinement_steps(*args, **kwargs):
    raise NotImplementedError("M7: R loop not implemented")
