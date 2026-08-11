"""Generate Phase C.1 occurrence-key and stochastic-capable mean-path evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.backends.deterministic import DeterministicOps
from adze_t.backends.mapping import (
    deterministic_to_torx,
    parameter_counts,
    torx_means_to_deterministic,
)
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.model import apply_model, init_model_params
from adze_t.objectives import loss_components, total_loss
from adze_t.parity import array_metric, compare_ordered_model_traces


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "phase_c_1"


def _write(name: str, value: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _key(key: jax.Array) -> list[int]:
    return [int(value) for value in jax.random.key_data(key)]


def _batch() -> tuple[jax.Array, jax.Array]:
    values = jnp.arange(1, 9, dtype=jnp.int32)[None, :]
    return values, jnp.ones_like(values, dtype=bool)


def _ops(key: jax.Array, **kwargs) -> TorxOps:
    return TorxOps.create(
        key,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        **kwargs,
    )


def _occurrences(params: Any, root_key: jax.Array):
    records = []

    def observe(kind, name, occurrence_path, occurrence_key):
        records.append(
            {
                "factor_kind": kind,
                "parameter_path": name,
                "occurrence_path": occurrence_path,
                "derived_key": _key(occurrence_key),
            }
        )

    values, mask = _batch()
    outputs = apply_model(
        params,
        values,
        mask,
        values,
        mask,
        ops=_ops(root_key, occurrence_observer=observe),
    )
    return outputs, records


def main() -> None:
    deterministic = init_model_params(jax.random.key(200))
    torx_params, mapping = deterministic_to_torx(deterministic)
    counts = parameter_counts(mapping)
    values, mask = _batch()

    torx_output, records = _occurrences(torx_params, jax.random.key(201))
    _, repeated = _occurrences(torx_params, jax.random.key(201))
    _, changed_root = _occurrences(torx_params, jax.random.key(202))

    def selected(items, name):
        return [item for item in items if item["parameter_path"] == name]

    prompt_target = selected(records, "frontend.byte_embed")
    q_projection = selected(records, "dit.block_0.q")
    modulation = selected(records, "dit.block_0.modulation")
    occurrence_evidence = {
        "base_commit": "ae19b50dce9bef04cf483314169533c0b7ef5961",
        "root_key": _key(jax.random.key(201)),
        "changed_root_key": _key(jax.random.key(202)),
        "derivation_order": [
            "root_key",
            "evaluation_id",
            "optimizer_step",
            "ordered_static_scopes",
            "module_path",
            "denoise_step_s",
            "refinement_iteration_r",
            "recurrence_cycle_q",
            "physical_layer_ell",
            "site_coordinate",
        ],
        "scope_and_module_hash": "BLAKE2s-32 little-endian uint32 over UTF-8",
        "prompt_target_frontend": prompt_target,
        "q3_block_0_q_projection": q_projection,
        "q3_block_0_modulation": modulation,
        "repeated_run_identical": records == repeated,
        "changed_root_changes_frontend_keys": selected(changed_root, "frontend.byte_embed")
        != prompt_target,
        "shared_frontend_mean_parameter_paths": [
            entry.torx_path
            for entry in mapping
            if entry.deterministic_path == "encoder/byte_embed" and entry.role == "mean"
        ],
        "parameter_counts": counts,
        "factor_occurrence_count": len(records),
        "passed": (
            len(prompt_target) == 2
            and prompt_target[0]["derived_key"] != prompt_target[1]["derived_key"]
            and len({tuple(item["derived_key"]) for item in q_projection}) == 3
            and len({tuple(item["derived_key"]) for item in modulation}) == 3
            and records == repeated
        ),
    }
    _write("occurrence_keys.json", occurrence_evidence)

    expected = apply_model(deterministic, values, mask, values, mask, ops=DeterministicOps())
    trace_metrics, first_divergence = compare_ordered_model_traces(expected, torx_output)
    changed_rho = jax.tree_util.tree_map_with_path(
        lambda path, value: (
            jnp.full_like(value, 75.0) if "['rho']" in jax.tree_util.keystr(path) else value
        ),
        torx_params,
    )
    changed_output = apply_model(
        changed_rho,
        values,
        mask,
        values,
        mask,
        ops=_ops(jax.random.key(999)),
    )
    invariance_metrics, invariance_divergence = compare_ordered_model_traces(
        torx_output, changed_output
    )

    def d_objective(params):
        components = loss_components(
            apply_model(params, values, mask, values, mask, ops=DeterministicOps())
        )
        return total_loss(components, REFERENCE_SMALL_V0), components

    def t_objective(params):
        components = loss_components(
            apply_model(
                params,
                values,
                mask,
                values,
                mask,
                ops=_ops(jax.random.key(203)),
            )
        )
        return total_loss(components, REFERENCE_SMALL_V0), components

    (d_loss, d_components), d_grad = jax.jit(jax.value_and_grad(d_objective, has_aux=True))(
        deterministic
    )
    (t_loss, t_components), t_grad = jax.jit(jax.value_and_grad(t_objective, has_aux=True))(
        torx_params
    )
    t_mean_grad = torx_means_to_deterministic(t_grad)
    gradient_records = []
    for (d_path, d_value), (t_path, t_value) in zip(
        jax.tree_util.tree_leaves_with_path(d_grad),
        jax.tree_util.tree_leaves_with_path(t_mean_grad),
        strict=True,
    ):
        d_name = jax.tree_util.keystr(d_path)
        t_name = jax.tree_util.keystr(t_path)
        if d_name != t_name:
            raise RuntimeError(f"gradient paths differ: {d_name} != {t_name}")
        gradient_records.append(
            array_metric(d_name, d_value, t_value, atol=1.0e-5, rtol=1.0e-5).to_dict()
        )
    rho_gradient_max = max(
        float(jnp.max(jnp.abs(leaf)))
        for path, leaf in jax.tree_util.tree_leaves_with_path(t_grad)
        if "['rho']" in jax.tree_util.keystr(path)
    )
    component_records = {
        name: array_metric(
            name, d_components[name], t_components[name], atol=1.0e-5, rtol=1.0e-5
        ).to_dict()
        for name in d_components
    }
    worst_gradient = max(gradient_records, key=lambda item: item["max_absolute_difference"])
    _write(
        "full_gradient_parity.json",
        {
            "operator_stochasticity": True,
            "lambda_op": 0.0,
            "total_loss": array_metric(
                "total_loss", d_loss, t_loss, atol=1.0e-5, rtol=1.0e-5
            ).to_dict(),
            "components": component_records,
            "forward_worst_max_absolute_difference": max(
                metric.max_absolute_difference for metric in trace_metrics
            ),
            "forward_first_divergence": None if first_divergence is None else first_divergence.path,
            "key_rho_invariance_worst_max_absolute_difference": max(
                metric.max_absolute_difference for metric in invariance_metrics
            ),
            "key_rho_invariance_first_divergence": None
            if invariance_divergence is None
            else invariance_divergence.path,
            "gradient_records": gradient_records,
            "worst_mapped_gradient": worst_gradient,
            "max_abs_grad_rho": rho_gradient_max,
            "factor_occurrence_count": len(records),
            "passed": (
                first_divergence is None
                and invariance_divergence is None
                and worst_gradient["passed"]
                and rho_gradient_max == 0.0
            ),
        },
    )


if __name__ == "__main__":
    main()
