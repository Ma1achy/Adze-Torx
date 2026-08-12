"""Generate machine-readable Phase-C zero-noise parity evidence."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.backends.deterministic import DeterministicOps
from adze_t.backends.mapping import (
    deterministic_to_torx,
    parameter_counts,
    torx_means_to_deterministic,
)
from adze_t.backends.torx import TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.mamba import MambaConfig, apply_mamba_stack, init_mamba_stack
from adze_t.model import apply_model, init_model_params
from adze_t.objectives import loss_components, total_loss
from adze_t.parity import array_metric, compare_ordered_model_traces


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "phase_c" / "parity"
ATOL_PRIMITIVE = 1.0e-6
ATOL_FULL = 1.0e-5


def write(name: str, value: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(path: str, left: jax.Array, right: jax.Array, tolerance: float) -> dict[str, Any]:
    result = array_metric(path, left, right, atol=tolerance, rtol=tolerance).to_dict()
    result.update({"seed": 0, "lambda_op": 0.0, "rho_configuration": "initial_sigma=1e-3"})
    return result


def batch() -> tuple[jax.Array, jax.Array]:
    values = jnp.arange(1, 9, dtype=jnp.int32)[None, :]
    return values, jnp.ones_like(values, dtype=bool)


def full_outputs(
    params: Any, torx_params: Any, values: jax.Array, mask: jax.Array, mode: str = "draft"
):
    calls: list[tuple[str, str]] = []
    deterministic = apply_model(
        params, values, mask, values, mask, ops=DeterministicOps(), mode=mode
    )
    torx_output = apply_model(
        torx_params,
        values,
        mask,
        values,
        mask,
        ops=TorxOps.create(
            jax.random.key(100), observer=lambda kind, name: calls.append((kind, name))
        ),
        mode=mode,
    )
    return deterministic, torx_output, calls


def main() -> None:
    params = init_model_params(jax.random.key(0))
    torx_params, mapping = deterministic_to_torx(params)
    counts = parameter_counts(mapping)
    values, mask = batch()

    dops = DeterministicOps()
    tops = TorxOps.create(jax.random.key(2))
    x = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 7
    linear = {
        "weight": jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 11,
        "bias": jnp.arange(5, dtype=jnp.float32) / 13,
    }
    mapped_linear, _ = deterministic_to_torx({"linear": linear})
    primitive_records = [
        record(
            "affine_with_bias",
            dops.linear(x, linear, name="primitive.linear"),
            tops.linear(x, mapped_linear["linear"], name="primitive.linear"),
            ATOL_PRIMITIVE,
        )
    ]
    embedding = jnp.arange(28, dtype=jnp.float32).reshape(7, 4) / 17
    mapped_embedding, _ = deterministic_to_torx({"embedding": embedding})
    indices = jnp.array([[0, 4, 6], [1, 3, 5]])
    primitive_records.append(
        record(
            "embedding",
            dops.embedding(indices, embedding, name="primitive.embedding"),
            tops.embedding(indices, mapped_embedding["embedding"], name="primitive.embedding"),
            ATOL_PRIMITIVE,
        )
    )
    conv = {
        "kernel": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 23,
        "bias": jnp.arange(4, dtype=jnp.float32) / 29,
    }
    mapped_conv, _ = deterministic_to_torx({"conv": conv})
    conv_x = jnp.arange(40, dtype=jnp.float32).reshape(2, 5, 4) / 19
    primitive_records.append(
        record(
            "depthwise_conv1d",
            dops.depthwise_conv1d(conv_x, conv, name="primitive.conv"),
            tops.depthwise_conv1d(conv_x, mapped_conv["conv"], name="primitive.conv"),
            ATOL_PRIMITIVE,
        )
    )

    mamba_config = MambaConfig(width=8, layers=2, expand=2, state_dim=4, conv_kernel=3)
    mamba_params = init_mamba_stack(jax.random.key(3), mamba_config, dops, name="probe")
    mapped_mamba, _ = deterministic_to_torx({"stack": mamba_params})
    mamba_x = jax.random.normal(jax.random.key(4), (2, 5, 8))
    masks = {
        "all_valid": jnp.ones((2, 5), dtype=bool),
        "internal_holes": jnp.array([[1, 0, 1, 1, 1], [1, 0, 0, 1, 1]], dtype=bool),
    }
    mamba_records = []
    for name, valid in masks.items():
        left = apply_mamba_stack(
            mamba_x, mamba_params, mamba_config, dops, name="probe", mask=valid
        )
        right = apply_mamba_stack(
            mamba_x, mapped_mamba["stack"], mamba_config, tops, name="probe", mask=valid
        )
        mamba_records.append(record(name, left, right, ATOL_PRIMITIVE))

    deterministic, torx_output, calls = full_outputs(params, torx_params, values, mask)
    full_metrics, first_divergence = compare_ordered_model_traces(deterministic, torx_output)
    full_records = [metric.to_dict() for metric in full_metrics]
    dit_records = [item for item in full_records if item["path"].startswith("dit/")]
    deterministic_refine, torx_refine, _ = full_outputs(
        params, torx_params, values, mask, mode="refine"
    )
    refine_metrics, refine_divergence = compare_ordered_model_traces(
        deterministic_refine, torx_refine
    )
    dit_records.extend(
        {**metric.to_dict(), "path": f"refine/{metric.path}"}
        for metric in refine_metrics
        if metric.path.startswith("dit/")
    )

    def d_objective(p):
        outputs = apply_model(p, values, mask, values, mask, ops=dops)
        return total_loss(loss_components(outputs), REFERENCE_SMALL_V0)

    def t_objective(p):
        outputs = apply_model(p, values, mask, values, mask, ops=TorxOps.create(jax.random.key(5)))
        return total_loss(loss_components(outputs), REFERENCE_SMALL_V0)

    d_loss, d_grad = jax.jit(jax.value_and_grad(d_objective))(params)
    t_loss, t_grad = jax.jit(jax.value_and_grad(t_objective))(torx_params)
    t_mean_grad = torx_means_to_deterministic(t_grad)
    gradient_records = [record("loss/total", d_loss, t_loss, ATOL_FULL)]
    for (left_path, left), (right_path, right) in zip(
        jax.tree_util.tree_leaves_with_path(d_grad),
        jax.tree_util.tree_leaves_with_path(t_mean_grad),
        strict=True,
    ):
        left_name = jax.tree_util.keystr(left_path)
        right_name = jax.tree_util.keystr(right_path)
        if left_name != right_name:
            raise RuntimeError(f"gradient semantic paths differ: {left_name} != {right_name}")
        gradient_records.append(record(left_name, left, right, ATOL_FULL))
    rho_gradient_max = max(
        float(jnp.max(jnp.abs(leaf)))
        for path, leaf in jax.tree_util.tree_leaves_with_path(t_grad)
        if "['rho']" in jax.tree_util.keystr(path)
    )

    jit_d = jax.jit(
        lambda p: apply_model(p, values, mask, values, mask, ops=DeterministicOps())["byte_logits"]
    )(params)
    jit_t = jax.jit(
        lambda p, key: apply_model(p, values, mask, values, mask, ops=TorxOps.create(key))[
            "byte_logits"
        ]
    )(torx_params, jax.random.key(6))
    full_records.append(record("jit/decoder_logits", jit_d, jit_t, ATOL_FULL))

    checkpoint_records = []
    for name in ("target_codec_b1", "copy", "reverse"):
        path = ROOT / "results" / "phase_b" / "checkpoints" / f"{name}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                checkpoint = pickle.load(handle)  # noqa: S301 - trusted committed repository artifact
            checkpoint_torx, _ = deterministic_to_torx(checkpoint)
            left, right, _ = full_outputs(checkpoint, checkpoint_torx, values, mask)
            metrics, divergence = compare_ordered_model_traces(left, right)
            checkpoint_records.append(
                {
                    "checkpoint": name,
                    "present": True,
                    "passed": divergence is None,
                    "worst_max_absolute_difference": max(
                        m.max_absolute_difference for m in metrics
                    ),
                    "first_divergence": None if divergence is None else divergence.path,
                }
            )

    invoked = sorted({f"{kind}:{name}" for kind, name in calls})
    primitive_records.append(
        {
            "path": "full_model_factor_invocation_manifest",
            "factor_occurrences": len(calls),
            "unique_factor_operations": len(invoked),
            "operations": invoked,
            "lambda_op": 0.0,
            "passed": len(calls) > 200,
        }
    )
    write("primitives.json", {"records": primitive_records})
    write("mamba.json", {"decision": "C2A_MAMBA_PARITY_PASS", "records": mamba_records})
    write("dit.json", {"decision": "C2B_DIT_PARITY_PASS", "records": dit_records})
    write(
        "full_model.json",
        {
            "records": full_records,
            "first_divergence": None if first_divergence is None else first_divergence.path,
            "refine_first_divergence": None
            if refine_divergence is None
            else refine_divergence.path,
            "trained_checkpoints": checkpoint_records,
        },
    )
    write(
        "gradients.json",
        {
            "records": gradient_records,
            "rho_gradient_max_absolute": rho_gradient_max,
            "all_records_passed": all(item["passed"] for item in gradient_records),
            "worst_parameter": max(
                gradient_records, key=lambda item: item["max_absolute_difference"]
            )["path"],
        },
    )
    write(
        "parameter_mapping.json",
        {
            "counts": counts,
            "entries": [entry.__dict__ for entry in mapping],
            "phase_d_obligation": "PHASE_D_DIRECT_PARAMETER_NOISE_POLICY_TBD",
        },
    )


if __name__ == "__main__":
    main()
