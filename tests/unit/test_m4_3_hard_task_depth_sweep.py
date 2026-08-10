import jax
import jax.numpy as jnp

from experiments.m4_2_nonlinear_looped_core.model import NonlinearCoreConfig, initialise_params
from experiments.m4_3_hard_task_depth_sweep.model import apply_core_trace, deterministic_trace
from experiments.m4_3_hard_task_depth_sweep.tasks import WIDTH, make_data


def _config(q=1):
    return NonlinearCoreConfig(
        width=WIDTH,
        blocks=2,
        q=q,
        eta=0.75,
        total_variance=0.04,
        active_mask=(1.0,) * 8 + (0.0,) * 56,
    )


def test_hard_task_is_deterministic_and_holds_out_sequence_residues():
    first = make_data(jax.random.PRNGKey(1), 32, 8, True)
    second = make_data(jax.random.PRNGKey(1), 32, 8, True)
    validation = make_data(jax.random.PRNGKey(2), 32, 8, False)
    assert jnp.array_equal(first.initial, second.initial)
    assert not bool(
        jnp.any(jnp.all(first.operators[:, None] == validation.operators[None], axis=(-1, -2)))
    )
    assert first.initial.shape == (32, WIDTH)
    assert first.intermediates.shape == (32, 9, 8)


def test_conditioning_slots_are_read_only_but_affect_dynamic_state():
    config = _config()
    params = initialise_params(config, jax.random.PRNGKey(3))
    params[0]["A"] = params[0]["A"].at[:8, 8].set(0.5)
    state = jnp.zeros((WIDTH,)).at[8].set(1.0)
    _, blocks, cycles = apply_core_trace(config, params, state, jax.random.PRNGKey(4))
    assert bool(jnp.all(blocks[:, 8:] == state[8:]))
    assert bool(jnp.any(blocks[-1, :8] != 0.0))
    assert bool(jnp.all(cycles[:, 8:] == state[8:]))


def test_q12_identity_initialization_parameter_count_and_variance():
    counts = []
    for q in (1, 2, 4, 8, 12):
        config = _config(q)
        params = initialise_params(config, jax.random.PRNGKey(q))
        state = jnp.linspace(-0.5, 0.5, WIDTH)
        _, _, cycles = deterministic_trace(config, params, state)
        counts.append(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))
        assert jnp.allclose(cycles[-1], state)
        assert config.per_block_variance * config.blocks * config.q == config.total_variance
    assert len(set(counts)) == 1
