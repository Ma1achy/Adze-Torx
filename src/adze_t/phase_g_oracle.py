"""Independent finite-state and Monte Carlo oracle for Phase-G score gradients."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .phase_g import generated_byte_metrics
from .train.score_bridge import score_corrected_loss


ORACLE_BATCH = 1
ORACLE_CAPACITY = 3
ORACLE_ACTION_COUNT = 108
ORACLE_ROOTS = 16
ORACLE_SAMPLES_PER_ROOT = 4096
ORACLE_Z_LIMIT = 4.0
ORACLE_ZERO_SE_ABS_TOL = 1.0e-6


class OracleParameters(NamedTuple):
    boundary_logits: jax.Array  # [2, 2], terminal is excluded
    extent_logits: jax.Array  # [3, 3]
    direct_theta: jax.Array


class OracleActions(NamedTuple):
    boundary: jax.Array  # [N, 3], terminal is one
    extent: jax.Array  # [N, 3]


def enumerate_actions() -> OracleActions:
    """Enumerate exactly 2^2 * 3^3 = 108 structural actions."""
    values = list(product(range(2), range(2), range(3), range(3), range(3)))
    raw = jnp.asarray(values, dtype=jnp.int32)
    boundary = jnp.concatenate((raw[:, :2], jnp.ones((len(values), 1), jnp.int32)), axis=1)
    return OracleActions(boundary, raw[:, 2:])


def fixture_parameters(name: str) -> OracleParameters:
    if name == "uniform":
        return OracleParameters(jnp.zeros((2, 2)), jnp.zeros((3, 3)), jnp.asarray(0.4))
    if name == "asymmetric":
        return OracleParameters(
            jnp.asarray([[0.7, -0.2], [-0.4, 0.9]]),
            jnp.asarray([[0.2, -0.7, 0.8], [1.1, -0.2, -0.4], [-0.3, 0.4, 0.9]]),
            jnp.asarray(-0.25),
        )
    if name == "peaked":
        return OracleParameters(
            jnp.asarray([[5.0, -4.0], [-3.0, 4.5]]),
            jnp.asarray([[4.0, -3.0, -5.0], [-4.0, 4.0, -3.0], [-2.0, -4.0, 5.0]]),
            jnp.asarray(0.75),
        )
    raise ValueError(f"unknown oracle parameter fixture: {name}")


def descendant_fixture(
    name: str,
) -> Callable[[OracleParameters, OracleActions], jax.Array]:
    """Return the production generated-length loss with a direct parameter path."""

    target_lengths = {
        "inactive": (1, 0, 1),
        "reactivation": (1, 1, 1),
        "zero": (0, 0, 0),
        "missing": (2, 2, 2),
        "extra": (1, 0, 0),
        "direct_dependencies": (1, 0, 2),
        "baseline": (1, 0, 2),
    }
    if name not in target_lengths:
        raise ValueError(f"unknown oracle descendant fixture: {name}")

    def loss(parameters: OracleParameters, actions: OracleActions) -> jax.Array:
        b = actions.boundary.astype(jnp.float32)
        count = actions.extent.shape[0]
        class_ids = jnp.asarray([[3, 4], [5, 6], [7, 8]], dtype=jnp.int32)
        site_weight = jnp.asarray([[0.4, -0.2], [0.7, 0.3], [-0.5, 0.9]])
        edge_effect = jnp.stack((b[:, 0], b[:, 0] - b[:, 1], b[:, 1]), axis=1)
        correct_logit = (
            parameters.direct_theta * site_weight[None, ...] + 0.6 * edge_effect[..., None]
        )
        logits = jnp.zeros((count, 3, 2, 256), dtype=jnp.float32)
        batch = jnp.arange(count)[:, None, None]
        carrier = jnp.arange(3)[None, :, None]
        slot = jnp.arange(2)[None, None, :]
        logits = logits.at[batch, carrier, slot, class_ids[None, ...]].set(correct_logit)
        target_length = jnp.asarray(target_lengths[name], dtype=jnp.int32)
        target_mask = jnp.arange(2)[None, :] < target_length[:, None]
        target_bytes = jnp.where(target_mask, class_ids, 0)
        return generated_byte_metrics(
            logits,
            actions.extent,
            jnp.broadcast_to(target_bytes, (count, 3, 2)),
            jnp.broadcast_to(target_mask, (count, 3, 2)),
        ).loss_per_example

    return loss


def _action_log_prob(parameters: OracleParameters, actions: OracleActions) -> jax.Array:
    b_lp = jax.nn.log_softmax(parameters.boundary_logits, axis=-1)
    l_lp = jax.nn.log_softmax(parameters.extent_logits, axis=-1)
    chosen_b = b_lp[jnp.arange(2)[None, :], actions.boundary[:, :2]]
    chosen_l = l_lp[jnp.arange(3)[None, :], actions.extent]
    return jnp.sum(chosen_b, axis=1) + jnp.sum(chosen_l, axis=1)


def exact_value_and_gradient(
    parameters: OracleParameters,
    descendant: Callable[[OracleParameters, OracleActions], jax.Array],
) -> tuple[jax.Array, OracleParameters]:
    actions = enumerate_actions()

    def expected(value):
        log_prob = _action_log_prob(value, actions)
        return jnp.sum(jnp.exp(log_prob) * descendant(value, actions))

    return jax.value_and_grad(expected)(parameters)


def enumerated_bridge_gradient(
    parameters: OracleParameters,
    descendant: Callable[[OracleParameters, OracleActions], jax.Array],
    *,
    mutation: str = "none",
) -> OracleParameters:
    actions = enumerate_actions()
    probabilities = jax.lax.stop_gradient(jnp.exp(_action_log_prob(parameters, actions)))

    def estimator(value):
        log_prob = _action_log_prob(value, actions)
        if mutation == "omit_first_boundary":
            first = jax.nn.log_softmax(value.boundary_logits[0])[actions.boundary[:, 0]]
            log_prob = log_prob - first
        elif mutation == "duplicate_all":
            log_prob = 2.0 * log_prob
        elif mutation != "none":
            raise ValueError(f"unknown oracle mutation: {mutation}")
        corrected = score_corrected_loss(descendant(value, actions), log_prob)
        return jnp.sum(probabilities * corrected)

    return jax.grad(estimator)(parameters)


def monte_carlo_root_gradient(
    parameters: OracleParameters,
    key: jax.Array,
    descendant: Callable[[OracleParameters, OracleActions], jax.Array],
    *,
    samples: int = ORACLE_SAMPLES_PER_ROOT,
) -> OracleParameters:
    boundary_key, extent_key = jax.random.split(key)

    def estimate(value):
        boundary = jax.random.categorical(
            boundary_key, value.boundary_logits, shape=(samples, 2)
        ).astype(jnp.int32)
        extent = jax.random.categorical(extent_key, value.extent_logits, shape=(samples, 3)).astype(
            jnp.int32
        )
        actions = OracleActions(
            jnp.concatenate((boundary, jnp.ones((samples, 1), jnp.int32)), axis=1), extent
        )
        log_prob = _action_log_prob(value, actions)
        return jnp.mean(score_corrected_loss(descendant(value, actions), log_prob))

    return jax.grad(estimate)(parameters)


def monte_carlo_conformance(
    parameters: OracleParameters,
    descendant: Callable[[OracleParameters, OracleActions], jax.Array],
    *,
    root_seed: int,
    roots: int = ORACLE_ROOTS,
    samples_per_root: int = ORACLE_SAMPLES_PER_ROOT,
) -> dict[str, jax.Array]:
    """Apply the frozen componentwise four-root-SE acceptance rule."""
    _, exact = exact_value_and_gradient(parameters, descendant)
    keys = jax.random.split(jax.random.key(root_seed), roots)
    estimates = jax.vmap(
        lambda key: monte_carlo_root_gradient(parameters, key, descendant, samples=samples_per_root)
    )(keys)

    def summarize(values, expected):
        mean = jnp.mean(values, axis=0)
        se = jnp.std(values, axis=0, ddof=1) / jnp.sqrt(roots)
        error = jnp.abs(mean - expected)
        accepted = jnp.where(
            se == 0,
            error <= ORACLE_ZERO_SE_ABS_TOL,
            error <= ORACLE_Z_LIMIT * se,
        )
        return mean, se, error, accepted

    b_mean, b_se, b_error, b_ok = summarize(estimates.boundary_logits, exact.boundary_logits)
    l_mean, l_se, l_error, l_ok = summarize(estimates.extent_logits, exact.extent_logits)
    t_mean, t_se, t_error, t_ok = summarize(estimates.direct_theta, exact.direct_theta)
    return {
        "boundary_exact": exact.boundary_logits,
        "boundary_mean": b_mean,
        "boundary_root_se": b_se,
        "boundary_abs_error": b_error,
        "boundary_accepted": b_ok,
        "extent_exact": exact.extent_logits,
        "extent_mean": l_mean,
        "extent_root_se": l_se,
        "extent_abs_error": l_error,
        "extent_accepted": l_ok,
        "direct_exact": exact.direct_theta,
        "direct_mean": t_mean,
        "direct_root_se": t_se,
        "direct_abs_error": t_error,
        "direct_accepted": t_ok,
        "boundary_root_estimates": estimates.boundary_logits,
        "extent_root_estimates": estimates.extent_logits,
        "direct_root_estimates": estimates.direct_theta,
        "accepted": jnp.all(b_ok) & jnp.all(l_ok) & jnp.all(t_ok),
    }
