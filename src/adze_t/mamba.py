"""Small explicit deterministic Mamba-1 reference block.

The micro-choices here are PROVISIONAL reference defaults, not claims about
the original Adze endpoint. Learned transforms route through ``LearnedOps``;
selective-scan and discretisation algebra remain explicit JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .backends.protocol import LearnedOps


@dataclass(frozen=True)
class MambaConfig:
    width: int
    layers: int
    expand: int = 2
    state_dim: int = 16
    conv_kernel: int = 3

    @property
    def inner_width(self) -> int:
        return self.width * self.expand


def init_mamba_stack(
    key: jax.Array, config: MambaConfig, ops: LearnedOps, *, name: str
) -> list[dict[str, Any]]:
    keys = iter(jax.random.split(key, config.layers * 4))
    stack = []
    for layer in range(config.layers):
        prefix = f"{name}.layer_{layer}"
        del prefix
        inner = config.inner_width
        stack.append(
            {
                "in_proj": ops.init_linear(next(keys), config.width, 2 * inner),
                "conv": ops.init_depthwise_conv(next(keys), config.conv_kernel, inner),
                "dbc_proj": ops.init_linear(next(keys), inner, inner + 2 * config.state_dim),
                "out_proj": ops.init_linear(next(keys), inner, config.width),
                "a_log": jnp.log(
                    jnp.broadcast_to(
                        jnp.arange(1, config.state_dim + 1, dtype=jnp.float32),
                        (inner, config.state_dim),
                    )
                ),
                "d_skip": jnp.ones((inner,), dtype=jnp.float32),
                "delta_bias": jnp.full((inner,), -2.0, dtype=jnp.float32),
                "layer_scale": jnp.ones((config.width,), dtype=jnp.float32),
            }
        )
    return stack


def _selective_scan(
    u: jax.Array,
    delta_raw: jax.Array,
    b_t: jax.Array,
    c_t: jax.Array,
    a_log: jax.Array,
    d_skip: jax.Array,
    delta_bias: jax.Array,
) -> jax.Array:
    """Stable diagonal selective scan with explicit deterministic algebra."""
    a = -jnp.exp(a_log)
    delta = jax.nn.softplus(delta_raw + delta_bias)
    initial = jnp.zeros((u.shape[0], u.shape[2], a.shape[1]), dtype=u.dtype)

    def step(state, inputs):
        u_step, dt_step, b_step, c_step = inputs
        transition = jnp.exp(dt_step[..., None] * a[None, :, :])
        drive = dt_step[..., None] * u_step[..., None] * b_step[:, None, :]
        state = transition * state + drive
        output = jnp.sum(state * c_step[:, None, :], axis=-1) + d_skip * u_step
        return state, output

    inputs = tuple(jnp.swapaxes(x, 0, 1) for x in (u, delta, b_t, c_t))
    _, output = jax.lax.scan(step, initial, inputs)
    return jnp.swapaxes(output, 0, 1)


def apply_mamba_block(
    x: jax.Array,
    params: dict[str, Any],
    config: MambaConfig,
    ops: LearnedOps,
    *,
    name: str,
    mask: jax.Array | None = None,
) -> jax.Array:
    residual = x
    projected = ops.linear(x, params["in_proj"], name=f"{name}.in_proj")
    u, gate = jnp.split(projected, 2, axis=-1)
    u = jax.nn.silu(ops.depthwise_conv1d(u, params["conv"], name=f"{name}.conv"))
    dbc = ops.linear(u, params["dbc_proj"], name=f"{name}.dbc_proj")
    delta_raw, b_t, c_t = jnp.split(
        dbc, [config.inner_width, config.inner_width + config.state_dim], axis=-1
    )
    scanned = _selective_scan(
        u,
        delta_raw,
        b_t,
        c_t,
        ops.parameter(params["a_log"], name=f"{name}.a_log"),
        ops.parameter(params["d_skip"], name=f"{name}.d_skip"),
        ops.parameter(params["delta_bias"], name=f"{name}.delta_bias"),
    )
    output = scanned * jax.nn.silu(gate)
    output = ops.linear(output, params["out_proj"], name=f"{name}.out_proj")
    scale = ops.parameter(params["layer_scale"], name=f"{name}.layer_scale")
    output = residual + scale * output
    if mask is not None:
        output = jnp.where(mask[..., None], output, residual)
    return output


def apply_mamba_stack(
    x: jax.Array,
    params: list[dict[str, Any]],
    config: MambaConfig,
    ops: LearnedOps,
    *,
    name: str,
    mask: jax.Array | None = None,
) -> jax.Array:
    for layer, layer_params in enumerate(params):
        x = apply_mamba_block(x, layer_params, config, ops, name=f"{name}.layer_{layer}", mask=mask)
    return x
