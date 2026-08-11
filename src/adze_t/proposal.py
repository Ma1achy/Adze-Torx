"""Backend-driven selective-SSM carrier proposal."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .backends.deterministic import DeterministicOps
from .backends.protocol import LearnedOps
from .config import ReferenceConfig
from .mamba import MambaConfig, apply_mamba_stack, init_mamba_stack


def _config(config: ReferenceConfig) -> MambaConfig:
    return MambaConfig(
        width=config.model.proposal_hidden_dim,
        layers=config.model.proposal_layers,
        expand=config.model.mamba_expand,
        state_dim=config.model.mamba_state_dim,
        conv_kernel=config.model.mamba_conv_kernel,
    )


def init_proposal_params(
    key: jax.Array, config: ReferenceConfig, ops: LearnedOps | None = None
) -> dict[str, Any]:
    ops = ops or DeterministicOps()
    m = config.model
    keys = iter(jax.random.split(key, 7))
    return {
        "context": ops.init_linear(next(keys), m.d_ctx, m.proposal_hidden_dim),
        "prior": ops.init_linear(next(keys), config.carrier.h_dim, m.proposal_hidden_dim),
        "position": ops.init_embedding(next(keys), config.carrier.C, m.proposal_hidden_dim),
        "stack": init_mamba_stack(next(keys), _config(config), ops, name="proposal"),
        "h": ops.init_linear(next(keys), m.proposal_hidden_dim, config.carrier.h_dim),
        "b": ops.init_linear(next(keys), m.proposal_hidden_dim, 2),
        "l": ops.init_linear(next(keys), m.proposal_hidden_dim, config.carrier.L_max + 1),
    }


def apply_proposal(
    context_global: jax.Array,
    carrier_prior: jax.Array,
    params: dict[str, Any],
    config: ReferenceConfig,
    ops: LearnedOps | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    ops = ops or DeterministicOps()
    batch = context_global.shape[0]
    context = ops.linear(context_global, params["context"], name="proposal.context")
    context = jnp.broadcast_to(context[:, None, :], (batch, config.carrier.C, context.shape[-1]))
    prior = ops.linear(carrier_prior, params["prior"], name="proposal.prior")
    positions = ops.embedding(
        jnp.arange(config.carrier.C)[None, :], params["position"], name="proposal.position"
    )
    hidden = context + prior + positions
    hidden = apply_mamba_stack(hidden, params["stack"], _config(config), ops, name="proposal")
    h = ops.linear(hidden, params["h"], name="proposal.h")
    b = ops.categorical_logits(hidden, params["b"], name="proposal.b")
    length = ops.categorical_logits(hidden, params["l"], name="proposal.l")
    b = b.at[:, -1, 1].set(10.0).at[:, -1, 0].set(-10.0)
    return h, b, length
