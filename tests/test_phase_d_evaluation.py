"""Phase-D Monte Carlo root and interval helpers."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.evaluation import (
    aggregate_root_chunks,
    all_d3_runs_pass,
    paired_chunk_statistics,
    phase_d_root,
    phase_d_stage_names,
    student_t_summary,
)
from adze_t.model import apply_model, init_model_params
from adze_t.objectives import emitted_metrics, loss_components, total_loss
from adze_t.parity import all_records_pass


def _tiny_config():
    cfg = REFERENCE_SMALL_V0
    return replace(
        cfg,
        carrier=replace(cfg.carrier, C=4, h_dim=8, L_max=2),
        packing=replace(cfg.packing, M_max=2, K=2),
        model=replace(
            cfg.model,
            d_front=4,
            d_ctx=8,
            frontend_layers=1,
            context_layers=1,
            target_layers=1,
            proposal_layers=1,
            proposal_hidden_dim=4,
            d_model=8,
            heads=1,
            head_dim=8,
            ffn_hidden=16,
            physical_blocks_L=1,
            cycles_Q=2,
            d_dec=8,
            decoder_layers=1,
            mamba_expand=1,
            mamba_state_dim=2,
        ),
    )


def _evaluate_chunks(params, prompt, target, root, chunk_size, lambda_op, config):
    chunks = []
    for start in range(0, prompt.shape[0], chunk_size):
        end = min(start + chunk_size, prompt.shape[0])
        chunks.append(
            paired_chunk_statistics(
                params,
                prompt[start:end],
                target[start:end],
                root,
                jnp.arange(start, end, dtype=jnp.uint32),
                jnp.asarray(lambda_op, jnp.float32),
                config=config,
            )
        )
    return chunks, aggregate_root_chunks(chunks, [chunk_size] * len(chunks), config=config)


def test_mc_roots_are_nested_reproducible_and_distinct():
    first_16 = [phase_d_root(4100, index) for index in range(16)]
    first_32 = [phase_d_root(4100, index) for index in range(32)]
    assert all(
        jnp.array_equal(left, right) for left, right in zip(first_16, first_32[:16], strict=True)
    )
    assert len({tuple(map(int, jax.random.key_data(key))) for key in first_32}) == 32


def test_student_t_summary_and_stage_manifest_are_frozen():
    singleton = student_t_summary([0.75])
    assert singleton == {"count": 1, "mean": 0.75, "sample_sd": 0.0, "ci95": [0.75, 0.75]}
    summary = student_t_summary([float(value) for value in range(16)])
    assert summary["count"] == 16
    assert summary["sample_sd"] > 0
    assert summary["ci95"][0] < summary["mean"] < summary["ci95"][1]
    names = phase_d_stage_names()
    assert names[:3] == ("frontend", "proposal", "pack")
    assert names[-4:] == ("unpool", "h_hat", "carrier", "decoder_logits")
    assert len([name for name in names if name.startswith("dit.q") and ".block" in name]) == 12


def test_mc_evaluation_is_chunking_invariant_and_lambda_zero_matches_dataset_metrics():
    config = _tiny_config()
    deterministic = init_model_params(jax.random.key(901), config)
    params, _ = deterministic_to_torx(deterministic)
    prompt = jnp.arange(8, dtype=jnp.int32).reshape(2, 4) + 1
    target = prompt[:, ::-1]
    root = jax.random.key(902)
    chunks16, result16 = _evaluate_chunks(params, prompt, target, root, 1, 1.0, config)
    chunks32, result32 = _evaluate_chunks(params, prompt, target, root, 2, 1.0, config)
    chunks64, result64 = _evaluate_chunks(params, prompt, target, root, 2, 1.0, config)
    logits16 = jnp.concatenate([chunk["byte_logits"] for chunk in chunks16], axis=0)
    logits32 = jnp.concatenate([chunk["byte_logits"] for chunk in chunks32], axis=0)
    logits64 = jnp.concatenate([chunk["byte_logits"] for chunk in chunks64], axis=0)
    assert jnp.allclose(logits16, logits32, atol=1.0e-6)
    assert jnp.allclose(logits16, logits64, atol=1.0e-6)
    for name in ("byte_accuracy", "exact_sequence_accuracy", "loss"):
        assert jnp.allclose(result16[name], result32[name], atol=1.0e-6)
        assert jnp.allclose(result16[name], result64[name], atol=1.0e-6)

    zero_chunks, zero_result = _evaluate_chunks(params, prompt, target, root, 2, 0.0, config)
    zero_logits = jnp.concatenate([chunk["byte_logits"] for chunk in zero_chunks], axis=0)
    mask = jnp.ones_like(prompt, dtype=bool)
    direct = apply_model(
        params,
        prompt,
        mask,
        target,
        mask,
        config=config,
        ops=TorxOps.create(
            root, config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0)
        ),
    )
    accuracy, exact = emitted_metrics(
        direct["byte_logits"],
        direct["target"]["teacher"].slot_bytes,
        direct["target"]["teacher"].slot_mask,
    )
    assert jnp.array_equal(zero_logits, direct["byte_logits"])
    assert jnp.array_equal(zero_result["byte_accuracy"], accuracy)
    assert jnp.array_equal(zero_result["exact_sequence_accuracy"], exact)
    # B=1 execution aggregates the exact objective numerators; float32 reduction
    # order can differ by one ULP from the original B=2 mean reduction.
    assert jnp.allclose(
        zero_result["loss"], total_loss(loss_components(direct), config), atol=1.0e-6, rtol=0.0
    )


def test_global_example_identity_is_stable_and_distinct():
    root = jax.random.key(903)
    config = TorxOperatorConfig(operator_stochasticity=True, lambda_op=1.0)
    first = TorxOps.create(root, config=config, global_example_id=7).context.key_for("oracle")
    repeated = TorxOps.create(root, config=config, global_example_id=7).context.key_for("oracle")
    other = TorxOps.create(root, config=config, global_example_id=8).context.key_for("oracle")
    changed_root = TorxOps.create(
        jax.random.key(904), config=config, global_example_id=7
    ).context.key_for("oracle")
    assert jnp.array_equal(first, repeated)
    assert not jnp.array_equal(first, other)
    assert not jnp.array_equal(first, changed_root)


def test_d3_aggregate_requires_every_present_stochastic_training_seed():
    def run(passed=True):
        return {
            "passed": passed,
            "lambda_zero": {"byte_accuracy": {"mean": 0.91}},
            "lambda_one": {"byte_accuracy": {"mean": 0.91}},
        }

    assert all_d3_runs_pass([run(), run(), run()])
    assert not all_d3_runs_pass([run(), run(False)])


def test_gradient_evidence_aggregation_rejects_a_non_worst_failure():
    records = [
        {"max_absolute_difference": 1.0, "passed": True},
        {"max_absolute_difference": 0.1, "passed": False},
    ]
    assert not all_records_pass(records)
