"""Adaptive compute policies for S, selection, and R."""


def convergence_signal(*args, **kwargs):
    raise NotImplementedError("M8: convergence signal not implemented")


def should_refine(*args, **kwargs):
    raise NotImplementedError("M8: refinement stopping policy not implemented")
