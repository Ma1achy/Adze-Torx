import jax
import jax.numpy as jnp

from adze_t.config import REFERENCE_SMALL_V0
from adze_t.training import initialise_training, make_fixed_structure_batch, train_step


def test_fixed_structure_training_step_and_gradients():
    cfg = REFERENCE_SMALL_V0
    params, moments = initialise_training(jax.random.PRNGKey(0), cfg)
    prompt = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    batch = make_fixed_structure_batch(prompt, prompt, config=cfg)
    params, moments, metrics = train_step(params, moments, 1, batch, config=cfg)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in metrics.values())
    assert float(metrics["loss"]) > 0.0
    _, _, metrics2 = train_step(params, moments, 2, batch, config=cfg)
    assert bool(jnp.isfinite(metrics2["loss"]))
