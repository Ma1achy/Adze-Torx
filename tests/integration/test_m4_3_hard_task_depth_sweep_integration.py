import jax
import jax.numpy as jnp

from experiments.m4_2_nonlinear_looped_core.model import NonlinearCoreConfig, initialise_params
from experiments.m4_3_hard_task_depth_sweep.model import apply_core_trace
from experiments.m4_3_hard_task_depth_sweep.tasks import WIDTH, make_data


def test_q12_forward_and_gradient_are_finite():
    config = NonlinearCoreConfig(
        width=WIDTH,
        blocks=2,
        q=12,
        eta=0.75,
        total_variance=0.04,
        active_mask=(1.0,) * 8 + (0.0,) * 56,
    )
    params = initialise_params(config, jax.random.PRNGKey(5))
    batch = make_data(jax.random.PRNGKey(6), 4, 4, True)
    keys = jax.random.split(jax.random.PRNGKey(7), 4)

    def loss(current):
        output = jax.vmap(lambda state, key: apply_core_trace(config, current, state, key)[0])(
            batch.initial, keys
        )
        return jnp.mean((output[:, :8] - batch.target[:, :8]) ** 2)

    value, gradients = jax.value_and_grad(loss)(params)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(gradients))
