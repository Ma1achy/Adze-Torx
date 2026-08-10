"""Deterministic learned operators shared by the Phase B architecture."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp


class DeterministicOps:
    """Pure-JAX learned-operator implementation.

    Parameters are ordinary named pytrees so their leaves can later become
    matched Torx factor means without changing the surrounding model graph.
    """

    def linear(self, x: Array, params: dict[str, Array]) -> Array:
        return x @ params["weight"] + params["bias"]

    def categorical_logits(self, x: Array, params: dict[str, Array]) -> Array:
        return self.linear(x, params)

    def ssm_transition(self, state: Array, x: Array, params: dict[str, Array]) -> Array:
        gate = jax_sigmoid(x @ params["gate_weight"] + params["gate_bias"])
        proposal = state @ params["state_weight"] + x @ params["input_weight"]
        return gate * jnp.tanh(proposal) + (1.0 - gate) * state

    @staticmethod
    def init_linear(key: Array, in_dim: int, out_dim: int, scale: float = 1.0) -> dict[str, Array]:
        import jax

        weight_key, _ = jax.random.split(key)
        weight = jax.random.normal(weight_key, (in_dim, out_dim)) * (scale / jnp.sqrt(in_dim))
        return {"weight": weight, "bias": jnp.zeros((out_dim,))}

    @staticmethod
    def init_ssm(key: Array, state_dim: int, input_dim: int) -> dict[str, Array]:
        import jax

        keys = jax.random.split(key, 3)
        return {
            "state_weight": jax.random.normal(keys[0], (state_dim, state_dim))
            / jnp.sqrt(state_dim),
            "input_weight": jax.random.normal(keys[1], (input_dim, state_dim))
            / jnp.sqrt(input_dim),
            "gate_weight": jax.random.normal(keys[2], (input_dim, state_dim)) / jnp.sqrt(input_dim),
            "gate_bias": jnp.zeros((state_dim,)),
        }


def jax_sigmoid(x: Array) -> Array:
    return jax_clip_sigmoid(x)


def jax_clip_sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -30.0, 30.0)))
