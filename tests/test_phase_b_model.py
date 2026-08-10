import jax
import jax.numpy as jnp

from adze_t.backends.deterministic import DeterministicOps
from adze_t.config import REFERENCE_SMALL_V0, load_reference_config
from adze_t.model import apply_model, init_model_params


def test_deterministic_backend_operator_interface():
    ops = DeterministicOps()
    params = ops.init_linear(jax.random.PRNGKey(0), 3, 4)
    x = jnp.ones((2, 3))
    assert ops.linear(x, params).shape == (2, 4)
    assert ops.categorical_logits(x, params).shape == (2, 4)


def test_reference_config_yaml_matches_typed_defaults():
    cfg = load_reference_config("configs/adze_reference_small_v0.yaml")
    assert cfg == REFERENCE_SMALL_V0


def test_complete_deterministic_forward_and_jit():
    cfg = REFERENCE_SMALL_V0
    params = init_model_params(jax.random.PRNGKey(0), cfg)
    prompt = jnp.arange(8, dtype=jnp.int32)[None, :]
    target = jnp.arange(8, dtype=jnp.int32)[None, :]
    mask = jnp.ones_like(prompt, dtype=bool)
    out = apply_model(params, prompt, mask, target, mask, config=cfg)
    assert out["context_seq"].shape == (1, 8, cfg.model.d_ctx)
    assert out["carrier"].shape == (1, cfg.carrier.C, cfg.carrier.h_dim)
    assert out["byte_logits"].shape == (1, cfg.carrier.C, cfg.carrier.L_max, cfg.model.byte_vocab)
    jitted = jax.jit(apply_model, static_argnames=("config", "mode"))
    out_jit = jitted(params, prompt, mask, target, mask, config=cfg)
    assert out_jit["byte_logits"].shape == out["byte_logits"].shape
