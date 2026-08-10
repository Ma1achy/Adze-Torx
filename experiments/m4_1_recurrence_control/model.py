"""M4.1 reuses the frozen M3 heads and only changes the core family."""

from experiments.m4_recurrent_core.model import (
    M4Config,
    batch_loss,
    cross_entropy,
    initialise_params,
    local_features,
    predict_batch,
    predict_one,
)

__all__ = [
    "M4Config",
    "batch_loss",
    "cross_entropy",
    "initialise_params",
    "local_features",
    "predict_batch",
    "predict_one",
]
