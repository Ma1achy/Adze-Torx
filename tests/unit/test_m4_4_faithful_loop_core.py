import jax
import jax.numpy as jnp

from experiments.m4_4_faithful_loop_core.model import (
    WIDTH,
    FaithfulConfig,
    apply_trace,
    deterministic_trace,
    initialise_params,
)
from experiments.m4_4_faithful_loop_core.tasks import make_data


def test_layout_mask_and_read_only_conditioning():
    batch = make_data(jax.random.PRNGKey(1), 4, 8, True)
    assert batch.initial.shape == (4, WIDTH)
    assert jnp.all(batch.initial[:, 80:88] == 1)
    assert jnp.all(batch.initial[:, 88:92] == 0)


def test_q_semantics_and_gate_count():
    for fixed in (True, False):
        for q in (1, 2, 4, 8, 12):
            config = FaithfulConfig(q=q, fixed_horizon=fixed)
            assert config.step_scale == (config.eta / q if fixed else config.eta)
            assert config.variance_per_gate == config.total_variance / (2 * config.blocks * q)


def test_identity_init_has_finite_gate_gradients_and_stable_map():
    config = FaithfulConfig(q=12)
    params = initialise_params(config, jax.random.PRNGKey(2))
    state = jnp.linspace(-0.3, 0.3, WIDTH)
    output, blocks, _ = deterministic_trace(config, params, state)
    assert bool(jnp.all(jnp.isfinite(output)))
    assert bool(jnp.linalg.norm(output - state) < 0.1)
    assert bool(jnp.allclose(blocks[:, 80:], state[80:]))

    def loss(candidate):
        return jnp.mean(deterministic_trace(config, candidate, state)[0][:8] ** 2)

    gradients = jax.grad(loss)(params)
    for block in gradients:
        for gate in ("gate1", "gate2"):
            assert bool(jnp.isfinite(jnp.linalg.norm(block[gate]["A"])))
            assert bool(jnp.linalg.norm(block[gate]["A"]) > 0)


def test_valid_conditioning_changes_state_but_padding_is_masked():
    config = FaithfulConfig(q=1)
    params = initialise_params(config, jax.random.PRNGKey(3))
    params[0]["gate1"]["A"] = params[0]["gate1"]["A"].at[:8, 32].set(0.5)
    state = jnp.zeros((WIDTH,)).at[80].set(1.0).at[32].set(1.0)
    padded = state.at[48].set(1.0)
    valid_changed = state.at[32].set(0.0).at[36].set(1.0).at[81].set(1.0)
    first = apply_trace(config, params, state, jax.random.PRNGKey(4))[0]
    assert jnp.allclose(
        first[:32], apply_trace(config, params, padded, jax.random.PRNGKey(4))[0][:32]
    )
    assert not bool(
        jnp.allclose(
            first[:8], apply_trace(config, params, valid_changed, jax.random.PRNGKey(4))[0][:8]
        )
    )
