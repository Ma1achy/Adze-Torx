from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from adze_t.phase_g01 import (
    balanced_extent_ce,
    candidate_step,
    decode_inferred_extents,
    extent_report,
    initialize_probe,
    length_labels,
    probe_apply,
    structurally_acceptable,
)


def truth_fixture():
    lengths = np.repeat(np.arange(8, 65), 2)
    return np.clip(lengths[:, None] - np.arange(32)[None, :] * 4, 0, 4).astype(np.int32)


def test_balanced_loss_is_not_diluted_by_duplicate_inactive_sites():
    logits = jnp.asarray([[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]])
    labels = jnp.asarray([[0, 2]])
    expanded = jnp.concatenate((jnp.repeat(logits[:, :1], 100, axis=1), logits[:, 1:]), axis=1)
    many = jnp.concatenate((jnp.zeros((1, 100), jnp.int32), labels[:, 1:]), axis=1)
    assert jnp.allclose(balanced_extent_ce(logits, labels), balanced_extent_ce(expanded, many))
    assert jnp.isfinite(jax.grad(lambda x: balanced_extent_ce(x, labels))(logits)).all()


def test_global_labels_and_exact_structural_gates_detect_partial_failures():
    truth = truth_fixture()
    labels = np.asarray(length_labels(jnp.asarray(truth)))
    assert np.array_equal(labels[:, 0] + 8, truth.sum(1))
    assert np.array_equal(labels[:, 1], (truth > 0).sum(1))
    assert structurally_acceptable(extent_report(truth, truth))
    bad = truth.copy()
    bad[(truth > 0) & (truth < 4)] = 4
    report = extent_report(bad, truth)
    assert report["partial_extent_accuracy"] == 0
    assert not structurally_acceptable(report)
    assert report["zero_detection_recall"] == 1
    assert report["exact_total_length_accuracy"] < 0.3


def test_probe_architectures_do_not_take_targets_or_masks_as_inputs():
    local = jnp.ones((2, 32, 64))
    global_x = local.reshape(2, -1)
    for mlp in (False, True):
        params = initialize_probe(jax.random.key(1020), 64, 2048, mlp)
        ell, glob = jax.jit(probe_apply, static_argnames=("mlp",))(params, local, global_x, mlp)
        assert ell.shape == (2, 32, 5)
        assert glob.shape == (2, 78)
        assert jnp.isfinite(ell).all() and jnp.isfinite(glob).all()


def test_readout_only_update_keeps_representation_decoder_and_other_heads_bitwise():
    from adze_t.config import REFERENCE_SMALL_V0
    from adze_t.model import init_model_params
    from adze_t.objectives import adamw_init
    from adze_t.phase_g0 import codec_subtree

    params = codec_subtree(init_model_params(jax.random.key(1021), REFERENCE_SMALL_V0))
    moments = (adamw_init(params), adamw_init(params))
    x = jnp.arange(1, 65, dtype=jnp.int32)[None, :] % 32 + 1
    mask = jnp.arange(64)[None, :] < 9
    updated, new_moments, result = jax.jit(candidate_step, static_argnames=("readout_only",))(
        params, moments, 1, x, mask, jax.random.key(1022), True
    )
    assert result["finite"]
    changed = False
    for (path, before), (_, after) in zip(
        jax.tree_util.tree_leaves_with_path(params),
        jax.tree_util.tree_leaves_with_path(updated),
        strict=True,
    ):
        is_head = any(key in jax.tree_util.keystr(path) for key in ("target_l", "target_b"))
        if is_head:
            changed |= not bool(jnp.array_equal(before, after))
        else:
            assert jnp.array_equal(before, after)
    assert changed
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(new_moments))
    h = jnp.ones((1, 32, 64))
    first = decode_inferred_extents(h, updated["encoder"]["target_l"], updated["decoder"])
    second = decode_inferred_extents(h, updated["encoder"]["target_l"], updated["decoder"])
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, first, second))
