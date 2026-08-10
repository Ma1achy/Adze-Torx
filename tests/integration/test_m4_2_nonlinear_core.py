import jax
import jax.numpy as jnp

from experiments.m4_2_nonlinear_looped_core.model import (
    NonlinearCoreConfig,
    apply_core,
    initialise_params,
)
from experiments.m4_2_nonlinear_looped_core.tasks import make_data, task_loss


def test_public_torx_nonlinear_core_forward_and_gradient_are_finite():
    config = NonlinearCoreConfig(blocks=2, q=2)
    params = initialise_params(config, jax.random.PRNGKey(3))
    batch = make_data(jax.random.PRNGKey(4), 8, 2, True, config.width)
    keys = jax.random.split(jax.random.PRNGKey(5), 8)
    loss = task_loss(params, batch, keys, lambda p, x, k: apply_core(config, p, x, k))
    gradients = jax.grad(
        lambda p: task_loss(p, batch, keys, lambda pp, x, k: apply_core(config, pp, x, k))
    )(params)
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(gradients))


def test_q1_is_one_physical_stack_and_shapes_are_fixed():
    config = NonlinearCoreConfig(blocks=2, q=1)
    params = initialise_params(config, jax.random.PRNGKey(6))
    state = jnp.zeros((config.width,))
    output = apply_core(config, params, state, jax.random.PRNGKey(7), return_trajectory=True)
    assert output.shape == (2, config.width)
