"""M4.4 initialization and recurrent trajectory diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from experiments.m4_4_faithful_loop_core.model import (
    apply_trace,
    deterministic_trace,
)


def initial_report(config, params, state):
    _, blocks, cycles = deterministic_trace(config, params, state)

    def objective(candidate):
        return jnp.mean(deterministic_trace(config, candidate, state)[0][:8] ** 2)

    grads = jax.grad(objective)(params)
    updates, _ = optax.adam(0.03).update(grads, optax.adam(0.03).init(params), params)
    gate_norms = {}
    for block, name in enumerate(("block1", "block2")):
        for gate in ("gate1", "gate2"):
            gate_norms[f"{name}_{gate}_grad"] = float(jnp.linalg.norm(grads[block][gate]["A"]))
    return {
        "displacement": float(jnp.linalg.norm(cycles[-1] - state)),
        "state_norm": float(jnp.linalg.norm(cycles[-1])),
        "update_norm": float(jnp.linalg.norm(blocks[-1] - blocks[-2])),
        "gradient_norm": float(jnp.linalg.norm(jax.tree_util.tree_leaves(grads)[0])),
        "optimizer_update_norm": float(
            jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(updates)))
        ),
        "conditioning_error": float(jnp.max(jnp.abs(blocks[:, 32:] - state[32:]))),
        **gate_norms,
    }


def trajectory(result, keys):
    config = result["config"]
    batch = result["validation_batch"]
    states = jax.vmap(lambda x, k: apply_trace(config, result["params"], x, k)[2])(
        batch.initial, keys
    )
    errors = []
    rows = []
    for cycle in range(config.q + 1):
        error = jnp.mean((states[:, cycle, :8] - batch.target[:, :8]) ** 2, axis=-1)
        errors.append(error)
        rows.append(
            {
                "cycle": cycle,
                "mse": float(jnp.mean(error)),
                "best_so_far": float(jnp.mean(jnp.min(jnp.stack(errors), axis=0))),
                "update_norm": 0.0
                if cycle == 0
                else float(
                    jnp.mean(jnp.linalg.norm(states[:, cycle] - states[:, cycle - 1], axis=-1))
                ),
                "state_norm": float(jnp.mean(jnp.linalg.norm(states[:, cycle, :32], axis=-1))),
                "scratch_norm": float(jnp.mean(jnp.linalg.norm(states[:, cycle, 8:32], axis=-1))),
                "helped_fraction": 0.0
                if cycle == 0
                else float(jnp.mean(error < errors[cycle - 1])),
            }
        )
    return rows
