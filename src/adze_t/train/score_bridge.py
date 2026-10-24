"""Small local score-function correction for mixed stochastic losses.

This is the oracle-validated M1.5 bridge, restored without semantic changes.
The caller must provide log probabilities only for stochastic operations whose
gradient is not already supplied by a native Torx/JAX route.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def score_corrected_loss(loss: jax.Array, log_prob_sum: jax.Array) -> jax.Array:
    """Add a zero-primal-value score correction without broadcasting."""
    loss = jnp.asarray(loss)
    log_prob_sum = jnp.asarray(log_prob_sum)
    if loss.shape != log_prob_sum.shape:
        raise ValueError(
            "loss and log_prob_sum must both be scalar or have the same "
            f"per-trajectory shape; got {loss.shape} and {log_prob_sum.shape}"
        )
    score_zero = log_prob_sum - jax.lax.stop_gradient(log_prob_sum)
    return loss + jax.lax.stop_gradient(loss) * score_zero
