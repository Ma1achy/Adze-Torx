import jax
import jax.numpy as jnp

from experiments.m4_4_faithful_loop_core.model import FaithfulConfig, apply_trace, initialise_params
from experiments.m4_4_faithful_loop_core.tasks import make_data


def test_q12_two_gate_path_forward_gradient_and_scratch():
    config = FaithfulConfig(q=12)
    params = initialise_params(config, jax.random.PRNGKey(5))
    batch = make_data(jax.random.PRNGKey(6), 2, 8, True)
    keys = jax.random.split(jax.random.PRNGKey(7), 2)

    def loss(candidate):
        output = jax.vmap(lambda x, k: apply_trace(config, candidate, x, k)[0])(batch.initial, keys)
        return jnp.mean((output[:, :8] - batch.target[:, :8]) ** 2)

    value, gradients = jax.value_and_grad(loss)(params)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(gradients))
    output, blocks, cycles = apply_trace(config, params, batch.initial[0], keys[0])
    assert output.shape == (92,)
    assert blocks.shape == (1 + 2 * 12, 92)
    assert cycles.shape == (13, 92)
    assert bool(jnp.all(blocks[:, 80:] == batch.initial[0, 80:]))
