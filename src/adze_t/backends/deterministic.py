"""Deterministic learned operators shared by the Phase B architecture."""

from __future__ import annotations

import jax
from jax import Array
import jax.numpy as jnp


class DeterministicOps:
    """Pure-JAX learned-operator implementation.

    Parameters are ordinary named pytrees so their leaves can later become
    matched Torx factor means without changing the surrounding model graph.
    """

    def linear(self, x: Array, params: dict[str, Array], *, name: str = "linear") -> Array:
        del name
        return x @ params["weight"] + params["bias"]

    def categorical_logits(
        self, x: Array, params: dict[str, Array], *, name: str = "categorical_logits"
    ) -> Array:
        return self.linear(x, params, name=name)

    def embedding(self, indices: Array, params: Array, *, name: str) -> Array:
        del name
        return params[indices.astype(jnp.int32)]

    def depthwise_conv1d(self, x: Array, params: dict[str, Array], *, name: str) -> Array:
        """Apply learned depthwise kernels; causal padding is explicit here."""
        del name
        kernel = params["kernel"][:, None, :]
        padded = jnp.pad(x, ((0, 0), (kernel.shape[0] - 1, 0), (0, 0)))
        out = jax.lax.conv_general_dilated(
            padded,
            kernel,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=x.shape[-1],
        )
        return out + params["bias"]

    def parameter(self, value: Array, *, name: str) -> Array:
        del name
        return value

    def init_linear(
        self, key: Array, in_dim: int, out_dim: int, *, scale: float = 1.0
    ) -> dict[str, Array]:
        weight_key, _ = jax.random.split(key)
        weight = jax.random.normal(weight_key, (in_dim, out_dim)) * (scale / jnp.sqrt(in_dim))
        return {"weight": weight, "bias": jnp.zeros((out_dim,))}

    def init_embedding(self, key: Array, size: int, width: int) -> Array:
        return jax.random.normal(key, (size, width)) / jnp.sqrt(width)

    def init_depthwise_conv(self, key: Array, kernel_size: int, channels: int) -> dict[str, Array]:
        return {
            "kernel": jax.random.normal(key, (kernel_size, channels)) / jnp.sqrt(kernel_size),
            "bias": jnp.zeros((channels,)),
        }
