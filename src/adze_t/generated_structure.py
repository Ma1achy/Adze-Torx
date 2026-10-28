"""Validated generated-structure transitions for Phase G.

Observed, predicted, and committed structure are deliberately separate. This
module owns only controlled transitions to committed routing; a prediction
never rewrites the topology of the application that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp
import torx

from .packing import PackMetadata, boundaries_to_blocks, build_pack_metadata_core
from .state import CommittedStructure, PredictedStructure
from .train.score_bridge import score_corrected_loss


PHASE_G_GENERATED_STRUCTURE_INTERFACE_V1 = "PHASE_G_GENERATED_STRUCTURE_INTERFACE_V1"
CommitPolicy = Literal["prediction_only", "argmax", "sample", "hysteresis"]


class StructureEstimatorOwner(str, Enum):
    NONE = "none"
    LOCAL_SCORE_BRIDGE = "local_score_bridge"
    NATIVE_TORX = "native_torx"


ROUTING_OK = 0
ROUTING_NONBINARY = 1
ROUTING_TERMINAL = 2
ROUTING_EXTENT_RANGE = 3
ROUTING_ACTIVITY_MISMATCH = 4
ROUTING_BLOCK_COUNT_OVERFLOW = 5
ROUTING_BLOCK_LENGTH_OVERFLOW = 6
COMMIT_REACTIVATION_FORBIDDEN = 7
ROUTING_DTYPE = 8
COMMIT_TEMPERATURE_INVALID = 10
COMMIT_LOGITS_NONFINITE = 11


def _stable_id(value: str) -> int:
    return int.from_bytes(hashlib.blake2s(value.encode(), digest_size=4).digest(), "little")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CommitMasks:
    """Separate ownership masks for outgoing boundary edges and extent sites."""

    boundary_edge: jax.Array
    extent_site: jax.Array

    def tree_flatten(self):
        return (self.boundary_edge, self.extent_site), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StructureKeyContext:
    """Explicit identity for a discrete structure-commit occurrence."""

    root_key: jax.Array
    occurrence: jax.Array | int
    global_example_id: jax.Array
    actual_s_index: jax.Array | int
    refinement_index: jax.Array | int
    commit_occurrence: jax.Array | int

    def tree_flatten(self):
        return (
            self.root_key,
            jnp.asarray(self.occurrence, dtype=jnp.uint32),
            self.global_example_id,
            jnp.asarray(self.actual_s_index, dtype=jnp.uint32),
            jnp.asarray(self.refinement_index, dtype=jnp.uint32),
            jnp.asarray(self.commit_occurrence, dtype=jnp.uint32),
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    def keys(self, action_kind: Literal["boundary", "extent"], capacity: int) -> jax.Array:
        """Return independently named per-example/per-site keys."""
        action_id = _stable_id(f"phase_g.structure:{action_kind}")

        def one(example_id):
            key = self.root_key
            for coordinate in (
                self.occurrence,
                example_id,
                self.actual_s_index,
                self.refinement_index,
                self.commit_occurrence,
                action_id,
            ):
                key = jax.random.fold_in(key, jnp.asarray(coordinate, dtype=jnp.uint32))
            return jax.vmap(lambda site: jax.random.fold_in(key, site))(
                jnp.arange(capacity, dtype=jnp.uint32)
            )

        return jax.vmap(one)(self.global_example_id)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StructureHysteresisState:
    boundary_last_switch: jax.Array
    boundary_last_confidence: jax.Array
    extent_last_switch: jax.Array
    extent_last_confidence: jax.Array

    def tree_flatten(self):
        return (
            self.boundary_last_switch,
            self.boundary_last_confidence,
            self.extent_last_switch,
            self.extent_last_confidence,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@dataclass(frozen=True)
class HysteresisConfig:
    """Explicit switch policy. There are intentionally no numeric defaults."""

    switch_threshold: float
    switch_margin: float
    minimum_dwell: int


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SampledStructure:
    boundary: jax.Array
    length: jax.Array
    boundary_log_prob: jax.Array
    length_log_prob: jax.Array
    descendant_log_prob: jax.Array

    def tree_flatten(self):
        return (
            self.boundary,
            self.length,
            self.boundary_log_prob,
            self.length_log_prob,
            self.descendant_log_prob,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StructureCommitResult:
    previous: CommittedStructure
    committed: CommittedStructure
    sampled: SampledStructure
    changed: jax.Array
    reactivated: jax.Array
    deactivated: jax.Array
    valid: jax.Array
    error_code: jax.Array
    hysteresis_state: StructureHysteresisState | None
    policy: str
    estimator_owner: StructureEstimatorOwner

    def tree_flatten(self):
        children = (
            self.previous,
            self.committed,
            self.sampled,
            self.changed,
            self.reactivated,
            self.deactivated,
            self.valid,
            self.error_code,
            self.hysteresis_state,
        )
        return children, (self.policy, self.estimator_owner)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GeneratedRoutingResult:
    """Validated fixed-shape routing plus a whole-batch execution gate."""

    structure: CommittedStructure
    metadata: PackMetadata
    valid: jax.Array
    error_code: jax.Array
    executed: jax.Array

    def tree_flatten(self):
        return (self.structure, self.metadata, self.valid, self.error_code, self.executed), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


class CategoricalCommitFactor(torx.AbstractReferenceFactor, torx.AbstractHasLogProbability):
    """Public-Torx categorical factor used by Adze-T structure commits."""

    input_ports: Mapping[str, Any]
    output_spec: Any

    def __init__(self, logits: jax.Array):
        if logits.ndim != 1:
            raise ValueError("categorical commit logits must be a vector")
        self.input_ports = {"logits": jax.ShapeDtypeStruct(logits.shape, logits.dtype)}
        self.output_spec = jax.ShapeDtypeStruct((), jnp.int32)

    def init_params(self, key: jax.Array) -> None:
        del key
        return None

    def sample(
        self,
        key: jax.Array,
        inputs: Mapping[str, Any],
        params: Any,
        info: Any = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Any:
        del params, site_info
        output = jax.random.categorical(key, inputs["logits"] / info["temperature"]).astype(
            jnp.int32
        )
        return (output, None) if return_aux else output

    def log_probability(
        self,
        inputs: Mapping[str, Any],
        outputs: jax.Array,
        params: Any,
        info: Any = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Any:
        del params, site_info
        log_probs = jax.nn.log_softmax(inputs["logits"] / info["temperature"])
        value = log_probs[outputs.astype(jnp.int32)]
        return (value, None) if return_aux else value


def closed_commit_masks(selected_sites: jax.Array) -> CommitMasks:
    """Close selected sites under outgoing-edge ownership."""
    selected = jnp.asarray(selected_sites, dtype=bool)
    boundary = selected & jnp.concatenate(
        (selected[:, 1:], jnp.zeros((selected.shape[0], 1), dtype=bool)), axis=1
    )
    return CommitMasks(boundary_edge=boundary, extent_site=selected)


def full_commit_masks(shape: tuple[int, int]) -> CommitMasks:
    return closed_commit_masks(jnp.ones(shape, dtype=bool))


def _committed_error_codes(structure: CommittedStructure, max_extent: int) -> jax.Array:
    if structure.length is None:
        raise ValueError("Phase G requires explicit committed lengths")
    dtype_valid = (
        jnp.issubdtype(structure.c_b.dtype, jnp.integer)
        and structure.activity.dtype == jnp.bool_
        and jnp.issubdtype(structure.length.dtype, jnp.integer)
    )
    boundary_binary = jnp.all((structure.c_b == 0) | (structure.c_b == 1), axis=1)
    activity_binary = jnp.all((structure.activity == 0) | (structure.activity == 1), axis=1)
    terminal = structure.c_b[:, -1] == 1
    extent_range = jnp.all((structure.length >= 0) & (structure.length <= max_extent), axis=1)
    activity_match = jnp.all((structure.length > 0) == structure.activity.astype(bool), axis=1)
    code = jnp.zeros((structure.c_b.shape[0],), dtype=jnp.int32)
    code = jnp.where(not dtype_valid, ROUTING_DTYPE, code)
    code = jnp.where((code == 0) & ~(boundary_binary & activity_binary), ROUTING_NONBINARY, code)
    code = jnp.where((code == 0) & ~terminal, ROUTING_TERMINAL, code)
    code = jnp.where((code == 0) & ~extent_range, ROUTING_EXTENT_RANGE, code)
    return jnp.where((code == 0) & ~activity_match, ROUTING_ACTIVITY_MISMATCH, code)


def _raise_invalid_committed(structure: CommittedStructure, *, max_extent: int) -> None:
    codes = _committed_error_codes(structure, max_extent)
    if bool(jnp.any(codes != ROUTING_OK)):
        raise ValueError(f"invalid committed structure; error codes={codes.tolist()}")


def _validate_eager_inputs(
    previous: CommittedStructure,
    predicted: PredictedStructure,
    masks: CommitMasks,
    *,
    max_extent: int,
) -> None:
    if previous.length is None:
        raise ValueError("Phase G requires explicit committed lengths")
    shape = previous.c_b.shape
    if len(shape) != 2:
        raise ValueError("committed structure must have shape [batch, carrier]")
    if previous.activity.shape != shape or previous.length.shape != shape:
        raise ValueError("committed structure fields must share [batch, carrier]")
    if predicted.b_logits.shape != (*shape, 2):
        raise ValueError("boundary logits must have shape [batch, carrier, 2]")
    if predicted.l_logits.shape != (*shape, max_extent + 1):
        raise ValueError("extent logits must have shape [batch, carrier, max_extent + 1]")
    if not jnp.issubdtype(predicted.b_logits.dtype, jnp.floating) or not jnp.issubdtype(
        predicted.l_logits.dtype, jnp.floating
    ):
        raise ValueError("structural logits must be floating point")
    for name, value in (
        ("boundary_edge mask", masks.boundary_edge),
        ("extent_site mask", masks.extent_site),
    ):
        if value.shape != shape or value.dtype != jnp.bool_:
            raise ValueError(f"{name} must be boolean with shape {shape}")
    if bool(jnp.any(masks.boundary_edge[:, -1])):
        raise ValueError("terminal boundary is forced and cannot be committed")
    if not bool(jnp.all(jnp.isfinite(predicted.b_logits))) or not bool(
        jnp.all(jnp.isfinite(predicted.l_logits))
    ):
        raise ValueError("structural logits must be finite")
    _raise_invalid_committed(previous, max_extent=max_extent)


def _sample_factor(
    logits: jax.Array, keys: jax.Array, temperature: jax.Array
) -> tuple[jax.Array, jax.Array]:
    factor = CategoricalCommitFactor(logits.reshape((-1, logits.shape[-1]))[0])
    flat_logits = logits.reshape((-1, logits.shape[-1]))
    flat_keys = keys.reshape((-1,))

    def one(value, key):
        inputs = {"logits": value}
        info = {"temperature": temperature}
        action = factor.sample(key, inputs, None, info)
        return action, factor.log_probability(inputs, action, None, info)

    action, log_prob = jax.vmap(one)(flat_logits, flat_keys)
    return action.reshape(logits.shape[:-1]), log_prob.reshape(logits.shape[:-1])


def _hysteresis_actions(
    previous: jax.Array,
    logits: jax.Array,
    last_switch: jax.Array,
    last_confidence: jax.Array,
    occurrence: jax.Array | int,
    config: HysteresisConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    probabilities = jax.nn.softmax(logits, axis=-1)
    candidate = jnp.argmax(probabilities, axis=-1).astype(jnp.int32)
    candidate_probability = jnp.take_along_axis(probabilities, candidate[..., None], axis=-1)[
        ..., 0
    ]
    current_probability = jnp.take_along_axis(
        probabilities, previous[..., None].astype(jnp.int32), axis=-1
    )[..., 0]
    dwell_ok = jnp.asarray(occurrence) - last_switch >= config.minimum_dwell
    accepted = (
        (candidate != previous)
        & (candidate_probability >= config.switch_threshold)
        & ((candidate_probability - current_probability) >= config.switch_margin)
        & dwell_ok
    )
    action = jnp.where(accepted, candidate, previous)
    switch = jnp.where(accepted, jnp.asarray(occurrence, last_switch.dtype), last_switch)
    confidence = jnp.where(accepted, candidate_probability, last_confidence)
    return action, switch, confidence


def commit_predicted_structure_core(
    previous: CommittedStructure,
    predicted: PredictedStructure,
    masks: CommitMasks,
    *,
    policy: CommitPolicy,
    max_extent: int,
    key_context: StructureKeyContext | None = None,
    temperature: jax.Array | float | None = None,
    estimator_owner: StructureEstimatorOwner = StructureEstimatorOwner.NONE,
    allow_reactivation: bool,
    hysteresis_config: HysteresisConfig | None = None,
    hysteresis_state: StructureHysteresisState | None = None,
) -> StructureCommitResult:
    """Array-first transition core. Eager callers use the validated wrapper."""
    del max_extent
    if previous.length is None:
        raise ValueError("Phase G requires explicit committed lengths")
    shape = previous.c_b.shape
    zero_b = jnp.zeros(shape, dtype=predicted.b_logits.dtype)
    zero_l = jnp.zeros(shape, dtype=predicted.l_logits.dtype)
    logits_finite = jnp.all(jnp.isfinite(predicted.b_logits), axis=(1, 2)) & jnp.all(
        jnp.isfinite(predicted.l_logits), axis=(1, 2)
    )
    safe_b_logits = jnp.where(jnp.isfinite(predicted.b_logits), predicted.b_logits, 0.0)
    safe_l_logits = jnp.where(jnp.isfinite(predicted.l_logits), predicted.l_logits, 0.0)
    temperature_valid = jnp.ones((shape[0],), dtype=bool)
    next_hysteresis = hysteresis_state

    if policy == "prediction_only":
        boundary, length = previous.c_b, previous.length
        boundary_lp, length_lp = zero_b, zero_l
    elif policy == "argmax":
        boundary = jax.lax.stop_gradient(jnp.argmax(safe_b_logits, axis=-1)).astype(jnp.int32)
        length = jax.lax.stop_gradient(jnp.argmax(safe_l_logits, axis=-1)).astype(jnp.int32)
        boundary_lp, length_lp = zero_b, zero_l
    elif policy == "sample":
        if key_context is None or temperature is None:
            raise ValueError("sample commit requires explicit key context and temperature")
        if estimator_owner not in (
            StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
            StructureEstimatorOwner.NATIVE_TORX,
        ):
            raise ValueError("sample commit requires an explicit score-gradient owner")
        raw_temperature = jnp.asarray(temperature)
        temperature_valid = jnp.broadcast_to(
            jnp.isfinite(raw_temperature) & (raw_temperature > 0), (shape[0],)
        )
        safe_temperature = jnp.where(jnp.all(temperature_valid), raw_temperature, 1.0)
        boundary_inner, boundary_inner_lp = _sample_factor(
            safe_b_logits[:, :-1],
            key_context.keys("boundary", shape[1] - 1),
            safe_temperature,
        )
        boundary = jnp.concatenate(
            (boundary_inner, jnp.ones((shape[0], 1), dtype=jnp.int32)), axis=1
        )
        boundary_lp = jnp.concatenate(
            (boundary_inner_lp, jnp.zeros((shape[0], 1), boundary_inner_lp.dtype)), axis=1
        )
        length, length_lp = _sample_factor(
            safe_l_logits,
            key_context.keys("extent", shape[1]),
            safe_temperature,
        )
    elif policy == "hysteresis":
        if hysteresis_config is None or hysteresis_state is None or key_context is None:
            raise ValueError("hysteresis requires explicit config, state, and occurrence context")
        boundary, b_switch, b_confidence = _hysteresis_actions(
            previous.c_b,
            safe_b_logits,
            hysteresis_state.boundary_last_switch,
            hysteresis_state.boundary_last_confidence,
            key_context.occurrence,
            hysteresis_config,
        )
        length, l_switch, l_confidence = _hysteresis_actions(
            previous.length,
            safe_l_logits,
            hysteresis_state.extent_last_switch,
            hysteresis_state.extent_last_confidence,
            key_context.occurrence,
            hysteresis_config,
        )
        next_hysteresis = StructureHysteresisState(
            jnp.where(masks.boundary_edge, b_switch, hysteresis_state.boundary_last_switch),
            jnp.where(masks.boundary_edge, b_confidence, hysteresis_state.boundary_last_confidence),
            jnp.where(masks.extent_site, l_switch, hysteresis_state.extent_last_switch),
            jnp.where(masks.extent_site, l_confidence, hysteresis_state.extent_last_confidence),
        )
        boundary_lp, length_lp = zero_b, zero_l
    else:
        raise ValueError(f"unknown structure commit policy: {policy}")

    boundary = jnp.where(masks.boundary_edge, boundary, previous.c_b).astype(jnp.int32)
    boundary = boundary.at[:, -1].set(1)
    length = jnp.where(masks.extent_site, length, previous.length).astype(jnp.int32)
    boundary_lp = jnp.where(masks.boundary_edge, boundary_lp, 0)
    length_lp = jnp.where(masks.extent_site, length_lp, 0)
    activity = length > 0
    committed = CommittedStructure(boundary, activity, length)
    previous_active = previous.length > 0
    reactivated = (~previous_active) & activity
    reactivation_violation = jnp.any(reactivated, axis=1) & jnp.logical_not(
        jnp.asarray(allow_reactivation)
    )
    error_code = jnp.where(
        ~logits_finite,
        COMMIT_LOGITS_NONFINITE,
        jnp.zeros((shape[0],), dtype=jnp.int32),
    )
    error_code = jnp.where(
        (error_code == 0) & ~temperature_valid,
        COMMIT_TEMPERATURE_INVALID,
        error_code,
    )
    error_code = jnp.where(
        (error_code == 0) & reactivation_violation,
        COMMIT_REACTIVATION_FORBIDDEN,
        error_code,
    )
    sampled = SampledStructure(
        boundary,
        length,
        boundary_lp,
        length_lp,
        jnp.sum(boundary_lp + length_lp, axis=1),
    )
    return StructureCommitResult(
        previous,
        committed,
        sampled,
        (boundary != previous.c_b) | (length != previous.length),
        reactivated,
        previous_active & (~activity),
        error_code == 0,
        error_code,
        next_hysteresis,
        policy,
        estimator_owner,
    )


def commit_predicted_structure(
    previous: CommittedStructure,
    predicted: PredictedStructure,
    masks: CommitMasks,
    *,
    policy: CommitPolicy,
    max_extent: int,
    key_context: StructureKeyContext | None = None,
    temperature: float | None = None,
    estimator_owner: StructureEstimatorOwner = StructureEstimatorOwner.NONE,
    allow_reactivation: bool,
    hysteresis_config: HysteresisConfig | None = None,
    hysteresis_state: StructureHysteresisState | None = None,
) -> StructureCommitResult:
    """Validated eager commit boundary."""
    _validate_eager_inputs(previous, predicted, masks, max_extent=max_extent)
    if policy == "sample" and (
        temperature is None or not jnp.isfinite(temperature) or temperature <= 0
    ):
        raise ValueError("sample temperature must be explicit, finite, and positive")
    if hysteresis_config is not None and (
        not 0 <= hysteresis_config.switch_threshold <= 1
        or hysteresis_config.switch_margin < 0
        or hysteresis_config.minimum_dwell < 0
    ):
        raise ValueError("invalid explicit hysteresis parameters")
    if policy == "sample" and estimator_owner == StructureEstimatorOwner.NATIVE_TORX:
        raise ValueError("Phase G sampled training requires local_score_bridge ownership")
    result = commit_predicted_structure_core(
        previous,
        predicted,
        masks,
        policy=policy,
        max_extent=max_extent,
        key_context=key_context,
        temperature=temperature,
        estimator_owner=estimator_owner,
        allow_reactivation=allow_reactivation,
        hysteresis_config=hysteresis_config,
        hysteresis_state=hysteresis_state,
    )
    _raise_invalid_committed(result.committed, max_extent=max_extent)
    if not bool(jnp.all(result.valid)):
        raise ValueError(
            f"structure commit policy failure; error codes={result.error_code.tolist()}"
        )
    return result


def _empty_metadata(batch: int, capacity: int, M_max: int, K: int) -> PackMetadata:
    bm, bmk = (batch, M_max), (batch, M_max, K)
    return PackMetadata(
        jnp.zeros(bm, bool),
        jnp.zeros(bmk, bool),
        jnp.zeros(bmk, bool),
        jnp.zeros(bmk, bool),
        jnp.zeros(bmk, bool),
        jnp.zeros(bmk, bool),
        jnp.full((batch, capacity), -1, jnp.int32),
        jnp.full((batch, capacity), -1, jnp.int32),
        jnp.full(bmk, -1, jnp.int32),
        jnp.zeros(bmk, jnp.int32),
        jnp.full(bmk, -1, jnp.int32),
        jnp.zeros(bmk, jnp.int32),
        jnp.zeros((batch,), bool),
        jnp.zeros((batch,), bool),
    )


def generated_routing_core(
    structure: CommittedStructure,
    *,
    M_max: int,
    K: int,
    max_extent: int,
    upstream_valid: jax.Array | None = None,
    upstream_error: jax.Array | None = None,
) -> GeneratedRoutingResult:
    """Compiled fail-closed routing; validity is decided before scatter."""
    if structure.length is None:
        raise ValueError("Phase G requires explicit committed lengths")
    batch, capacity = structure.c_b.shape
    code = _committed_error_codes(structure, max_extent)
    if upstream_valid is not None:
        if upstream_error is None:
            raise ValueError("upstream validity requires explicit error codes")
        code = jnp.where((code == 0) & ~upstream_valid, upstream_error, code)
    # Both lax.cond branches are traced. Use type-safe candidates so a static
    # dtype violation returns its error code rather than failing while tracing
    # an execution branch which will never be selected.
    traced_boundary = structure.c_b.astype(jnp.int32)
    traced_activity = structure.activity.astype(bool)
    layout = boundaries_to_blocks(traced_boundary)
    count_overflow = layout.n_blocks > M_max
    length_overflow = jnp.max(layout.block_lengths, axis=1) > K
    code = jnp.where((code == 0) & count_overflow, ROUTING_BLOCK_COUNT_OVERFLOW, code)
    code = jnp.where((code == 0) & length_overflow, ROUTING_BLOCK_LENGTH_OVERFLOW, code)
    valid = code == ROUTING_OK
    executed = jnp.all(valid)
    metadata = jax.lax.cond(
        executed,
        lambda _: build_pack_metadata_core(traced_boundary, traced_activity, M_max=M_max, K=K),
        lambda _: _empty_metadata(batch, capacity, M_max, K),
        operand=None,
    )
    return GeneratedRoutingResult(structure, metadata, valid, code, executed)


def generated_routing(
    structure: CommittedStructure,
    *,
    M_max: int,
    K: int,
    max_extent: int,
) -> GeneratedRoutingResult:
    """Eager wrapper rejecting malformed or overflowing generated routing."""
    if M_max <= 0 or K <= 0:
        raise ValueError("M_max and K must be positive")
    _raise_invalid_committed(structure, max_extent=max_extent)
    result = generated_routing_core(structure, M_max=M_max, K=K, max_extent=max_extent)
    if not bool(result.executed):
        raise ValueError(
            f"generated routing capacity error; error codes={result.error_code.tolist()}"
        )
    return result


def score_corrected_structure_loss(
    descendant_loss: jax.Array, commit: StructureCommitResult
) -> jax.Array:
    """Apply the local score bridge exactly once to sampled descendants."""
    if (
        commit.policy != "sample"
        or commit.estimator_owner != StructureEstimatorOwner.LOCAL_SCORE_BRIDGE
    ):
        raise ValueError("local score correction requires sampled local_score_bridge ownership")
    return score_corrected_loss(descendant_loss, commit.sampled.descendant_log_prob)


# Compatibility aliases for the first unvalidated engineering scaffold. New
# integration code must use the reviewed public V1 names above.
HysteresisState = StructureHysteresisState
StructureCommit = StructureCommitResult
GeneratedRouting = GeneratedRoutingResult
ScoreGradientOwner = StructureEstimatorOwner


def structure_churn(
    previous: CommittedStructure, current: CommittedStructure
) -> dict[str, jax.Array]:
    if previous.length is None or current.length is None:
        raise ValueError("Phase G churn requires explicit lengths")
    old_active, new_active = previous.length > 0, current.length > 0
    nonterminal = jnp.ones_like(previous.c_b, dtype=bool).at[:, -1].set(False)
    return {
        "boundary_change_rate": jnp.sum((previous.c_b != current.c_b) & nonterminal)
        / jnp.maximum(jnp.sum(nonterminal), 1),
        "extent_change_rate": jnp.mean(previous.length != current.length),
        "reactivation_rate": jnp.mean((~old_active) & new_active),
        "deactivation_rate": jnp.mean(old_active & (~new_active)),
        "block_count_delta_mean": jnp.mean(
            jnp.sum(current.c_b, axis=1) - jnp.sum(previous.c_b, axis=1)
        ),
    }


def categorical_calibration(
    logits: jax.Array, labels: jax.Array, mask: jax.Array, *, bins: int = 10
) -> dict[str, jax.Array]:
    """Stable accuracy/NLL/Brier/ECE with validated eager domains."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    if logits.ndim < 2 or labels.shape != logits.shape[:-1] or mask.shape != labels.shape:
        raise ValueError("calibration logits, labels, and mask shapes do not agree")
    if mask.dtype != jnp.bool_:
        raise ValueError("calibration mask must be boolean")
    if not bool(jnp.all(jnp.isfinite(logits))):
        raise ValueError("calibration logits must be finite")
    if not bool(jnp.all((labels >= 0) & (labels < logits.shape[-1]))):
        raise ValueError("calibration labels are outside the categorical domain")
    count = jnp.sum(mask)
    if int(count) == 0:
        raise ValueError("calibration mask has empty support")
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    predicted, confidence = jnp.argmax(probs, axis=-1), jnp.max(probs, axis=-1)
    correct, weights = predicted == labels, mask.astype(logits.dtype)
    selected = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    one_hot = jax.nn.one_hot(labels, logits.shape[-1], dtype=logits.dtype)
    ece = jnp.asarray(0.0, dtype=logits.dtype)
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        in_bin = (
            (confidence >= low)
            & (confidence < high if index + 1 < bins else confidence <= high)
            & mask
        )
        bin_count = jnp.sum(in_bin)
        bin_accuracy = jnp.sum(correct * in_bin) / jnp.maximum(bin_count, 1)
        bin_confidence = jnp.sum(confidence * in_bin) / jnp.maximum(bin_count, 1)
        ece += (bin_count / count) * jnp.abs(bin_accuracy - bin_confidence)
    return {
        "count": count,
        "accuracy": jnp.sum(correct * weights) / count,
        "nll": -jnp.sum(selected * weights) / count,
        "brier": jnp.sum(jnp.sum((probs - one_hot) ** 2, axis=-1) * weights) / count,
        "ece": ece,
    }


def structure_calibration(
    predicted: PredictedStructure,
    boundary_labels: jax.Array,
    length_labels: jax.Array,
    *,
    bins: int = 10,
) -> dict[str, dict[str, jax.Array]]:
    boundary_mask = jnp.ones_like(boundary_labels, dtype=bool).at[:, -1].set(False)
    return {
        "boundary": categorical_calibration(
            predicted.b_logits, boundary_labels, boundary_mask, bins=bins
        ),
        "extent": categorical_calibration(
            predicted.l_logits, length_labels, jnp.ones_like(length_labels, bool), bins=bins
        ),
    }
