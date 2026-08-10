import jax
import jax.numpy as jnp

from experiments.m4_5_algorithmic_reasoning.arithmetic import make_data
from experiments.m4_5_algorithmic_reasoning.model import (
    AlgorithmicConfig,
    apply_state,
    initialise_params,
)
from experiments.m4_5_algorithmic_reasoning.programs import make_data as make_program


def test_all_at_once_forward_and_cursor_forward_are_finite():
    arithmetic = make_data(jax.random.PRNGKey(21), 2, 8)
    config = AlgorithmicConfig(q=4, conditioning_width=48, output_shape=(13, 10))
    params = initialise_params(config, jax.random.PRNGKey(22))
    schedule = jnp.broadcast_to(arithmetic.all_conditioning[:, None, :], (2, 4, 48))
    output = jax.vmap(lambda d, c: apply_state(config, params, d, c, jax.random.PRNGKey(23)))(
        arithmetic.initial_dynamic, schedule
    )
    assert output.shape == (2, 32)
    assert bool(jnp.all(jnp.isfinite(output)))

    program = make_program(jax.random.PRNGKey(24), 2, 4)
    cursor_config = AlgorithmicConfig(q=4, conditioning_width=12, output_shape=(2, 16))
    cursor_params = initialise_params(cursor_config, jax.random.PRNGKey(25))
    cursor = jax.vmap(
        lambda d, c: apply_state(cursor_config, cursor_params, d, c[:4], jax.random.PRNGKey(26))
    )(program.initial_dynamic, program.cursor_conditioning)
    assert cursor.shape == (2, 32)
    assert bool(jnp.all(jnp.isfinite(cursor)))
