import jax
import jax.numpy as jnp

from adze_t.model.core import RecurrentCoreConfig, apply_core, deterministic_core, initialise_params


def test_current_and_residual_public_torx_paths_have_fixed_shapes_and_finite_gradients():
    for family in ("current", "residual"):
        config = RecurrentCoreConfig(width=6, q=2, family=family, eta=0.25)
        params = initialise_params(config, jax.random.key(10))
        state = jnp.arange(6.0)
        output = apply_core(config, params, state, jax.random.key(11))
        assert output.shape == state.shape
        assert bool(jnp.all(jnp.isfinite(output)))

        def objective(p, config=config, state=state):
            return jnp.mean(deterministic_core(config, p, state) ** 2)

        grads = jax.grad(objective)(params)
        assert bool(
            jnp.all(
                jnp.asarray(
                    [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads)]
                )
            )
        )


def test_cycle_conditioning_adds_shared_parameters_without_untying():
    plain = RecurrentCoreConfig(width=4, q=4, family="residual", eta=0.25)
    conditioned = RecurrentCoreConfig(
        width=4, q=4, family="residual", eta=0.25, cycle_conditioning=True
    )
    plain_params = initialise_params(plain, jax.random.key(12))
    conditioned_params = initialise_params(conditioned, jax.random.key(12))
    assert not isinstance(plain_params, list)
    assert not isinstance(conditioned_params, list)
    assert "cycle_bias" in conditioned_params
    assert sum(x.size for x in jax.tree_util.tree_leaves(conditioned_params)) > sum(
        x.size for x in jax.tree_util.tree_leaves(plain_params)
    )


def test_zero_noise_diagnostic_is_analytical_mean_not_invalid_log_variance():
    config = RecurrentCoreConfig(width=4, q=2, family="residual", eta=0.25)
    params = initialise_params(config, jax.random.key(13))
    state = jnp.ones(4)
    mean = deterministic_core(config, params, state)
    assert bool(jnp.all(jnp.isfinite(mean)))
    assert bool(jnp.all(jnp.isfinite(params["log_var"])))
