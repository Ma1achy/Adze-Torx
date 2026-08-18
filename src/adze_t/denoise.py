"""Leakage-safe same-model denoising trajectories for Phase F.2."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
from jax import Array
import jax.numpy as jnp

from .backends.deterministic import DeterministicOps
from .backends.torx import TorxOperatorConfig, TorxOps
from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .corruption import (
    DiffusionEtaMode,
    DiffusionStage,
    PHASE_F_MAX_DENOISE_STEPS,
    diffusion_key,
    phase_f_schedule,
    recorrupt_h,
)
from .model import apply_model
from .teacher import TeacherStructure


F2_STEP0_CONDITIONING = "F2_STEP0_CONDITIONING"
F2_NATIVE_S_CONDITIONING_UNTRAINED = "F2_NATIVE_S_CONDITIONING_UNTRAINED"
F2_CONDITIONING_MODES = (
    F2_STEP0_CONDITIONING,
    F2_NATIVE_S_CONDITIONING_UNTRAINED,
)


class SanitizedTargetAnalysis(NamedTuple):
    """Forward-only fixed structure plus content placeholders that the graph ignores."""

    target_frontend: Array
    h0_placeholder: Array
    b_logits_placeholder: Array
    l_logits_placeholder: Array
    teacher: TeacherStructure
    target_bytes_placeholder: Array
    target_mask_placeholder: Array


class DenoisingTrajectory(NamedTuple):
    """Step-major outputs from repeated use of one shared parameter tree."""

    h_hat: Array
    byte_logits: Array
    b_logits: Array
    l_logits: Array
    input_states: Array
    schedule: Array
    metadata: Any
    diagnostics: dict[str, Array]
    actual_s_indices: Array
    denoise_condition_indices: Array
    recorruption_epsilon: Array


def make_sanitized_target_analysis(
    boundaries: Array,
    length: Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    target_width: int | None = None,
    placeholder_value: float | int = 0,
) -> SanitizedTargetAnalysis:
    """Build the only target-shaped object allowed inside an F2 forward trajectory."""
    boundaries = jnp.asarray(boundaries)
    length = jnp.asarray(length)
    expected = (boundaries.shape[0], config.carrier.C)
    if boundaries.shape != expected or length.shape != expected:
        raise ValueError(f"boundaries and length must have shape {expected}")
    width = config.carrier.C * config.carrier.L_max if target_width is None else target_width
    if not 1 <= width <= config.carrier.C * config.carrier.L_max:
        raise ValueError("target_width must fit the carrier emission capacity")
    batch = boundaries.shape[0]
    activity = length > 0
    slot_mask = jnp.arange(config.carrier.L_max)[None, None, :] < length[..., None]
    content_int = jnp.asarray(placeholder_value, dtype=jnp.int32)
    content_float = jnp.asarray(placeholder_value, dtype=jnp.float32)
    teacher = TeacherStructure(
        boundaries=boundaries,
        length=length,
        activity=activity,
        slot_mask=slot_mask,
        slot_bytes=jnp.full(
            (batch, config.carrier.C, config.carrier.L_max), content_int, dtype=jnp.int32
        ),
        capacity_overflow=jnp.zeros((batch,), dtype=bool),
        prefix_mask_valid=jnp.ones((batch,), dtype=bool),
    )
    return SanitizedTargetAnalysis(
        target_frontend=jnp.full(
            (batch, width, config.model.d_front), content_float, dtype=jnp.float32
        ),
        h0_placeholder=jnp.full(
            (batch, config.carrier.C, config.carrier.h_dim), content_float, dtype=jnp.float32
        ),
        b_logits_placeholder=jnp.full(
            (batch, config.carrier.C, 2), content_float, dtype=jnp.float32
        ),
        l_logits_placeholder=jnp.full(
            (batch, config.carrier.C, config.carrier.L_max + 1),
            content_float,
            dtype=jnp.float32,
        ),
        teacher=teacher,
        target_bytes_placeholder=jnp.full((batch, width), content_int, dtype=jnp.int32),
        target_mask_placeholder=jnp.zeros((batch, width), dtype=bool),
    )


def _as_model_target_analysis(value: SanitizedTargetAnalysis) -> dict[str, Any]:
    return {
        "target_frontend": value.target_frontend,
        "target": {
            "h0": value.h0_placeholder,
            "b_logits": value.b_logits_placeholder,
            "l_logits": value.l_logits_placeholder,
            "teacher": value.teacher,
        },
    }


def _conditioning_index(mode: str, actual_s_index: int) -> int:
    if mode == F2_STEP0_CONDITIONING:
        return 0
    if mode == F2_NATIVE_S_CONDITIONING_UNTRAINED:
        return actual_s_index
    raise ValueError(f"unknown F2 conditioning mode: {mode}")


def apply_denoising_trajectory(
    params: Any,
    prompt: Array,
    prompt_mask: Array,
    initial_corrupted_carrier: Array,
    sanitized_target: SanitizedTargetAnalysis,
    nu0: Array,
    global_example_ids: Array,
    *,
    s_exec: int,
    eta_diff: int | DiffusionEtaMode,
    diffusion_root: Array,
    operator_backend: str = "deterministic",
    operator_root: Array | None = None,
    operator_config: TorxOperatorConfig | None = None,
    conditioning_mode: str = F2_STEP0_CONDITIONING,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    dit_cycles: int | None = None,
) -> DenoisingTrajectory:
    """Apply one checkpoint up to four times without accepting clean carrier content.

    ``actual_s_index`` controls stochastic occurrence identity.  The independent
    learned conditioning coordinate is frozen to zero in the primary F2 mode.
    """
    if not 1 <= s_exec <= PHASE_F_MAX_DENOISE_STEPS:
        raise ValueError(f"s_exec must lie in [1, {PHASE_F_MAX_DENOISE_STEPS}]")
    if conditioning_mode not in F2_CONDITIONING_MODES:
        raise ValueError(f"unknown F2 conditioning mode: {conditioning_mode}")
    if operator_backend not in {"deterministic", "torx"}:
        raise ValueError("operator_backend must be 'deterministic' or 'torx'")
    if operator_backend == "torx" and operator_root is None:
        raise ValueError("operator_root is required for the Torx backend")
    batch = prompt.shape[0]
    if prompt_mask.shape != prompt.shape:
        raise ValueError("prompt and prompt_mask must have identical shapes")
    if initial_corrupted_carrier.shape[0] != batch:
        raise ValueError("initial carrier batch must match prompt batch")
    if global_example_ids.shape != (batch,) or nu0.shape != (batch,):
        raise ValueError("nu0 and global_example_ids must have shape [batch]")
    if sanitized_target.teacher.boundaries.shape[0] != batch:
        raise ValueError("sanitized target batch must match prompt batch")

    resolved_operator_config = operator_config or TorxOperatorConfig()
    resolved_operator_root = jax.random.PRNGKey(0) if operator_root is None else operator_root

    def run_one(
        sample_prompt: Array,
        sample_prompt_mask: Array,
        initial_state: Array,
        sample_target: SanitizedTargetAnalysis,
        sample_nu0: Array,
        example_id: Array,
    ) -> DenoisingTrajectory:
        target = jax.tree.map(lambda value: value[None, ...], sample_target)
        state = initial_state[None, ...]
        schedule = phase_f_schedule(sample_nu0, s_exec)
        states = []
        h_hats = []
        byte_logits = []
        b_logits = []
        l_logits = []
        metadata = []
        packed_input_rms = []
        unpooled_rms = []
        epsilons = []
        actual_indices = []
        condition_indices = []
        for actual_s_index in range(s_exec):
            denoise_condition_index = _conditioning_index(conditioning_mode, actual_s_index)
            if operator_backend == "torx":
                ops = TorxOps.create(
                    resolved_operator_root,
                    config=resolved_operator_config,
                    global_example_id=example_id,
                ).with_occurrence(denoise_step=actual_s_index)
            else:
                ops = DeterministicOps()
            states.append(state[0])
            output = apply_model(
                params,
                sample_prompt[None, ...],
                sample_prompt_mask[None, ...],
                target.target_bytes_placeholder,
                target.target_mask_placeholder,
                config=config,
                ops=ops,
                target_analysis=_as_model_target_analysis(target),
                carrier_h_input=state,
                noise_level=schedule[actual_s_index][None],
                actual_s_index=actual_s_index,
                denoise_condition_index=denoise_condition_index,
                dit_cycles=dit_cycles,
            )
            h_hat = output["prediction"][0]
            h_hats.append(h_hat[0])
            b_logits.append(output["prediction"][1][0])
            l_logits.append(output["prediction"][2][0])
            byte_logits.append(output["byte_logits"][0])
            metadata.append(jax.tree.map(lambda value: value[0], output["metadata"]))
            packed_input_rms.append(output["activation_rms"]["packed_input"])
            unpooled_rms.append(output["activation_rms"]["unpooled_carrier"])
            actual_indices.append(actual_s_index)
            condition_indices.append(denoise_condition_index)
            if actual_s_index + 1 < s_exec:
                next_s_index = actual_s_index + 1
                key = diffusion_key(
                    diffusion_root,
                    global_example_id=example_id,
                    stage=DiffusionStage.RECORRUPTION,
                    denoise_step=next_s_index,
                )
                epsilon = jax.random.normal(key, h_hat.shape[1:], dtype=h_hat.dtype)
                epsilons.append(epsilon)
                state = recorrupt_h(
                    h_hat,
                    schedule[next_s_index],
                    epsilon[None, ...],
                    eta_diff,
                )

        stacked_h = jnp.stack(h_hats)
        changes = stacked_h[1:] - stacked_h[:-1]
        previous_rms = jnp.sqrt(jnp.mean(stacked_h[:-1] ** 2, axis=(1, 2)))
        change_rms = jnp.sqrt(jnp.mean(changes**2, axis=(1, 2)))
        dot = jnp.sum(stacked_h[1:] * stacked_h[:-1], axis=(1, 2))
        norm_product = jnp.sqrt(
            jnp.sum(stacked_h[1:] ** 2, axis=(1, 2)) * jnp.sum(stacked_h[:-1] ** 2, axis=(1, 2))
        )
        zeros = jnp.zeros((1,), dtype=stacked_h.dtype)
        diagnostics = {
            "h_hat_rms": jnp.sqrt(jnp.mean(stacked_h**2, axis=(1, 2))),
            "inter_step_rms_change": jnp.concatenate((zeros, change_rms)),
            "relative_update": jnp.concatenate(
                (zeros, change_rms / jnp.maximum(previous_rms, 1.0e-12))
            ),
            "cosine_similarity": jnp.concatenate(
                (jnp.ones((1,), dtype=stacked_h.dtype), dot / jnp.maximum(norm_product, 1.0e-12))
            ),
            "packed_input_rms": jnp.stack(packed_input_rms),
            "unpooled_carrier_rms": jnp.stack(unpooled_rms),
            "nonfinite": jnp.asarray(
                [
                    jnp.any(~jnp.isfinite(states[index]))
                    | jnp.any(~jnp.isfinite(h_hats[index]))
                    | jnp.any(~jnp.isfinite(byte_logits[index]))
                    for index in range(s_exec)
                ]
            ),
        }
        epsilon_shape = (0, *initial_state.shape) if not epsilons else None
        stacked_epsilon = (
            jnp.zeros(epsilon_shape, dtype=initial_state.dtype)
            if epsilon_shape is not None
            else jnp.stack(epsilons)
        )
        return DenoisingTrajectory(
            h_hat=stacked_h,
            byte_logits=jnp.stack(byte_logits),
            b_logits=jnp.stack(b_logits),
            l_logits=jnp.stack(l_logits),
            input_states=jnp.stack(states),
            schedule=schedule,
            metadata=jax.tree.map(lambda *values: jnp.stack(values), *metadata),
            diagnostics=diagnostics,
            actual_s_indices=jnp.asarray(actual_indices, dtype=jnp.int32),
            denoise_condition_indices=jnp.asarray(condition_indices, dtype=jnp.int32),
            recorruption_epsilon=stacked_epsilon,
        )

    batch_major = jax.vmap(run_one)(
        prompt,
        prompt_mask,
        initial_corrupted_carrier,
        sanitized_target,
        nu0,
        global_example_ids,
    )
    return jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), batch_major)
