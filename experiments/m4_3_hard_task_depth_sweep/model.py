"""M4.2 public-Torx core with per-physical-block trace diagnostics."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from torx.psc import AffineGaussianGate

from experiments.m4_2_nonlinear_looped_core.model import NonlinearCoreConfig


def _gate(config: NonlinearCoreConfig) -> AffineGaussianGate:
    return AffineGaussianGate(sites=[0], dims=(config.width,))


def _sample(config, params, state, key):
    gate = _gate(config)
    inputs = {
        "continuous": state,
        "discrete": jnp.empty((0,), dtype=gate.input_ports["discrete"].dtype),
    }
    return cast(jax.Array, gate.sample(key, inputs, params))


def apply_core_trace(config, params, state, key):
    """Apply the reused M4.2 stack and retain every physical-block state."""
    keys = jax.random.split(key, config.blocks * config.q)
    current = state
    block_states = [state]
    cycle_states = [state]
    index = 0
    for _cycle in range(config.q):
        for block in range(config.blocks):
            proposal = _sample(config, params[block], current, keys[index])
            if config.nonlinear:
                proposal = jnp.tanh(proposal)
            current = current + config.per_block_step * config.mask * proposal
            block_states.append(current)
            index += 1
        cycle_states.append(current)
    return jnp.stack(current), jnp.stack(block_states), jnp.stack(cycle_states)


def deterministic_trace(config, params, state):
    current = state
    block_states = [state]
    cycle_states = [state]
    for _cycle in range(config.q):
        for block in range(config.blocks):
            proposal = params[block]["A"] @ current + params[block]["b"]
            if config.nonlinear:
                proposal = jnp.tanh(proposal)
            current = current + config.per_block_step * config.mask * proposal
            block_states.append(current)
        cycle_states.append(current)
    return current, jnp.stack(block_states), jnp.stack(cycle_states)
