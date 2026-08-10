import jax
import jax.numpy as jnp

from adze_t.dit import DiTConfig, apply_dit, init_dit_params
from adze_t.packing import build_pack_metadata_core, pack_values


def _core_inputs():
    c_b = jnp.array([[0, 1, 0, 0, 1, 1]], dtype=jnp.int32)
    activity = jnp.ones_like(c_b)
    metadata = build_pack_metadata_core(c_b, activity, M_max=4, K=4)
    values = jnp.arange(6 * 128, dtype=jnp.float32).reshape(1, 6, 128) / 100.0
    packed = pack_values(values, metadata)
    cfg = DiTConfig(carrier_capacity=6, d_context=128)
    params = init_dit_params(jax.random.PRNGKey(0), cfg)
    context = jnp.zeros((1, 128))
    return packed, metadata, cfg, params, context


def test_looped_dit_shapes_and_parameter_tying():
    packed, metadata, cfg, params, context = _core_inputs()
    out, aux = apply_dit(packed, metadata, context, params, cfg)
    assert out.shape == packed.shape
    assert aux["trajectory"].shape[0] == cfg.cycles
    assert len(params["blocks"]) == cfg.physical_blocks


def test_draft_attention_has_no_later_block_leakage():
    packed, metadata, cfg, params, context = _core_inputs()
    out_a, _ = apply_dit(packed, metadata, context, params, cfg, mode="draft")
    later = packed.at[:, 1:, :, :].add(10.0)
    out_b, _ = apply_dit(later, metadata, context, params, cfg, mode="draft")
    assert jnp.allclose(out_a[:, :1], out_b[:, :1], atol=1.0e-5)


def test_refine_allows_later_block_influence():
    packed, metadata, cfg, params, context = _core_inputs()
    out_a, _ = apply_dit(packed, metadata, context, params, cfg, mode="refine")
    later = packed.at[:, 1:, :, :].add(10.0)
    out_b, _ = apply_dit(later, metadata, context, params, cfg, mode="refine")
    assert not jnp.allclose(out_a[:, :1], out_b[:, :1], atol=1.0e-5)


def test_dit_jit_and_gradient_coverage():
    packed, metadata, cfg, params, context = _core_inputs()

    def loss(p):
        output, _ = apply_dit(packed, metadata, context, p, cfg)
        return jnp.mean(output**2)

    grads = jax.grad(loss)(params)
    for name in ("q", "k", "v", "o", "up", "gate", "down"):
        norm = jnp.linalg.norm(grads["blocks"][0][name]["weight"])
        assert bool(jnp.isfinite(norm)) and float(norm) > 0.0
    assert jax.jit(loss)(params).shape == ()
