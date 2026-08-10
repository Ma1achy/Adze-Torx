import jax
import jax.numpy as jnp

from experiments.m4_2_nonlinear_looped_core.model import (
    NonlinearCoreConfig,
    deterministic_core,
    initialise_params,
)
from experiments.m4_2_nonlinear_looped_core.tasks import make_data


def test_zero_initialization_is_identity_for_all_q():
    state = jnp.linspace(-0.5, 0.5, 18)
    for q in (1, 2, 4):
        config = NonlinearCoreConfig(q=q)
        params = initialise_params(config, jax.random.PRNGKey(q))
        assert jnp.allclose(deterministic_core(config, params, state), state)
        assert config.per_block_variance * config.blocks * config.q == config.total_variance


def test_physical_blocks_are_distinct_and_q_does_not_clone_parameters():
    config = NonlinearCoreConfig(blocks=2, q=4)
    params = initialise_params(config, jax.random.PRNGKey(0))
    assert len(params) == 2
    assert config.blocks * (config.width**2 + 2 * config.width) == sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(params)
    )


def test_nonlinear_map_has_state_dependent_jacobian_after_nonzero_parameters():
    config = NonlinearCoreConfig(q=1)
    params = initialise_params(config, jax.random.PRNGKey(0))
    params[0]["A"] = 0.7 * jnp.eye(config.width)
    x = jnp.linspace(-0.7, 0.7, config.width)
    y = jnp.full_like(x, 0.1)
    jac_x = jax.jacobian(lambda value: deterministic_core(config, params, value))(x)
    jac_y = jax.jacobian(lambda value: deterministic_core(config, params, value))(y)
    assert jnp.linalg.norm(jac_x - jac_y) > 1e-4


def test_operator_ids_are_input_features_and_validation_sequences_are_held_out():
    train = make_data(jax.random.PRNGKey(1), 64, 4, True)
    validation = make_data(jax.random.PRNGKey(2), 64, 4, False)
    assert jnp.any(train.initial[:, 3:15] != 0)
    assert not bool(
        jnp.any(jnp.all(train.operators[:, None, :] == validation.operators[None, :, :], axis=-1))
    )
