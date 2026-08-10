"""M4.1 training with optimizer-scale and wall-clock diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax

from experiments.m3_carrier_structure.data import StructuredCarrier
from experiments.m4_1_recurrence_control.model import M4Config, batch_loss
from experiments.m4_recurrent_core.train import TrainConfig as M4TrainConfig
from experiments.m4_recurrent_core.train import evaluate


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 60
    batch_size: int = 64
    learning_rate: float = 0.03
    alpha: float = 0.6
    sigma: float = 0.5
    rho_b: float = 0.5
    rho_length: float = 0.5
    seed: int = 0
    eval_interval: int = 10


def keys_for(seed: int, step: int, n: int) -> jax.Array:
    return jax.random.split(jax.random.key(seed + 1009 * step), n)


def _tree_norm(tree: Any) -> jax.Array:
    return optax.global_norm(tree)


def _finite(tree: Any) -> bool:
    return bool(
        jnp.all(
            jnp.asarray(
                [
                    jnp.all(jnp.isfinite(cast(jax.Array, leaf)))
                    for leaf in jax.tree_util.tree_leaves(tree)
                ]
            )
        )
    )


def _group_norm(tree: Any, group: str) -> jax.Array:
    return _tree_norm(tree[group])


def _delta_norm(tree: Any) -> jax.Array:
    return _tree_norm({"A": tree["A"], "b": tree["b"]})


def _batch(data: StructuredCarrier, indices: jax.Array) -> StructuredCarrier:
    return jax.tree_util.tree_map(lambda value, indices=indices: value[indices], data)


def run(
    config: M4Config,
    train_config: TrainConfig,
    train_data: StructuredCarrier,
    val_data: StructuredCarrier,
    initialise_params,
) -> tuple[Any, dict[str, Any]]:
    params: Any = initialise_params(config, jax.random.key(train_config.seed))
    optimizer = optax.adam(train_config.learning_rate)
    opt_state = optimizer.init(params)
    validation_keys = keys_for(train_config.seed + 50000, 0, val_data.h.shape[0])
    eval_config = M4TrainConfig(
        steps=train_config.steps,
        batch_size=train_config.batch_size,
        learning_rate=train_config.learning_rate,
        alpha=train_config.alpha,
        sigma=train_config.sigma,
        rho_b=train_config.rho_b,
        rho_length=train_config.rho_length,
        seed=train_config.seed,
    )

    def loss_fn(p, batch, keys):
        return batch_loss(
            p,
            batch,
            keys,
            config,
            train_config.alpha,
            train_config.sigma,
            train_config.rho_b,
            train_config.rho_length,
        )

    step_fn = jax.jit(jax.value_and_grad(loss_fn))
    records: list[dict[str, float | int | None]] = []
    validation_curve: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for step in range(train_config.steps):
        indices = jax.random.randint(
            jax.random.key(train_config.seed + step),
            (train_config.batch_size,),
            0,
            train_data.h.shape[0],
        )
        batch = _batch(train_data, indices)
        step_start = time.perf_counter()
        value, grads = step_fn(
            params, batch, keys_for(train_config.seed, step, train_config.batch_size)
        )
        value.block_until_ready()
        step_seconds = time.perf_counter() - step_start
        if not bool(jnp.isfinite(value)) or not _finite(grads):
            raise FloatingPointError("non-finite M4.1 loss or gradient")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        update_norm = _tree_norm(updates)
        parameter_norm = _tree_norm(params)
        core_parameter_norm = _group_norm(params, "core")
        core_update_norm = _group_norm(updates, "core")
        delta_parameter_norm = _delta_norm(params["core"])
        delta_update_norm = _delta_norm(cast(Any, updates)["core"])
        core_ratio = (
            None
            if float(core_parameter_norm) == 0.0
            else float(core_update_norm / core_parameter_norm)
        )
        delta_ratio = (
            None
            if float(delta_parameter_norm) == 0.0
            else float(delta_update_norm / delta_parameter_norm)
        )
        record = {
            "step": step,
            "loss": float(value),
            "grad_norm": float(_tree_norm(grads)),
            "core_grad_norm": float(_group_norm(grads, "core")),
            "head_grad_norm": float(_tree_norm({k: v for k, v in grads.items() if k != "core"})),
            "update_norm": float(update_norm),
            "core_update_norm": float(core_update_norm),
            "parameter_norm": float(parameter_norm),
            "core_parameter_norm": float(core_parameter_norm),
            "core_update_parameter_ratio": core_ratio,
            "delta_parameter_norm": float(delta_parameter_norm),
            "delta_update_norm": float(delta_update_norm),
            "delta_update_parameter_ratio": delta_ratio,
            "update_over_state_scale": float(update_norm / jnp.maximum(parameter_norm, 1e-12)),
            "step_seconds": step_seconds,
        }
        records.append(record)
        params = optax.apply_updates(params, updates)
        completed_step = step + 1
        if completed_step % train_config.eval_interval == 0 or completed_step == train_config.steps:
            metrics = evaluate(params, val_data, validation_keys, config, eval_config)
            validation_curve.append(
                {"step": completed_step, "h_mse": float(metrics["h_mse"]), "loss": float(value)}
            )
    total_seconds = time.perf_counter() - start
    final_metrics = evaluate(params, val_data, validation_keys, config, eval_config)
    h_values = [float(row["h_mse"]) for row in validation_curve]
    best_index = min(range(len(h_values)), key=lambda index: h_values[index])
    steady = [float(cast(float, row["step_seconds"])) for row in records[1:]]
    steady_step = (
        float(jnp.median(jnp.asarray(steady)))
        if steady
        else float(cast(float, records[0]["step_seconds"]))
    )
    output: dict[str, Any] = dict(final_metrics)
    output.update(
        {
            "train_curve": records,
            "validation_curve": validation_curve,
            "best_val_h_mse": h_values[best_index],
            "best_val_step": validation_curve[best_index]["step"],
            "final_val_h_mse": float(final_metrics["h_mse"]),
            "total_train_seconds": total_seconds,
            "first_step_seconds": cast(float, records[0]["step_seconds"]),
            "steady_step_seconds": steady_step,
            "compile_seconds_estimate": max(
                0.0, float(cast(float, records[0]["step_seconds"])) - steady_step
            ),
            "parameter_count": sum(
                int(cast(jax.Array, leaf).size) for leaf in jax.tree_util.tree_leaves(params)
            ),
        }
    )
    return params, output
