"""Phase-F V0 continuous-content corruption and diffusion key contracts.

This is an explicit Phase-F reference design, not recovered original-Adze
behaviour and not an exact implementation of a named diffusion method.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp


PHASE_F_CORRUPTION_CONTRACT_V0 = "PHASE_F_CORRUPTION_CONTRACT_V0"
PHASE_F_CONTINUOUS_S_FIXED_STRUCTURE_V0 = "PHASE_F_CONTINUOUS_S_FIXED_STRUCTURE_V0"
PHASE_F_ROLLOUT_LOSS_V0 = "PHASE_F_ROLLOUT_LOSS_V0"

PHASE_F_NU_MIN = 0.0
PHASE_F_NU_MAX = 1.0
PHASE_F_TRAIN_NU_MIN = 0.025
PHASE_F_TRAIN_NU_MAX = 0.9
PHASE_F_EVAL_GRID = (0.10, 0.25, 0.50, 0.75, 0.90)
PHASE_F_SCHEDULE_MULTIPLIERS = (1.0, 0.75, 0.50, 0.25)
PHASE_F_MAX_DENOISE_STEPS = 4

# Fixed uint32 identities, deliberately separate from the Torx operator-key
# occurrence sequence. These values are part of the V0 diffusion-key contract.
_DIFFUSION_NAMESPACE_ID = 0x50464430  # ASCII-like marker: "PFD0"
_DIFFUSION_EXAMPLE_ID_MARKER = 0x4558414D  # "EXAM"
_DIFFUSION_STAGE_MARKER = 0x53544745  # "STGE"
_DIFFUSION_STEP_MARKER = 0x53544550  # "STEP"


class DiffusionStage(IntEnum):
    """Stable V0 identities for independent diffusion-noise stages."""

    INITIAL_CORRUPTION = 0x494E4954  # "INIT"
    RECORRUPTION = 0x5245434F  # "RECO"


class DiffusionEtaMode(IntEnum):
    """Authoritative V0 inter-step diffusion modes."""

    DETERMINISTIC_MEAN = 0
    FULL_RENOISE = 1


class CorruptionTrace(NamedTuple):
    """A sampled corrupted carrier and the Gaussian epsilon that produced it."""

    value: Array
    epsilon: Array


@dataclass(frozen=True)
class DiffusionKeyContext:
    """Identity needed to derive diffusion keys independently of operator noise."""

    root_key: Array
    global_example_id: int | Array

    def key_for(self, stage: DiffusionStage, denoise_step: int | Array) -> Array:
        return diffusion_key(
            self.root_key,
            global_example_id=self.global_example_id,
            stage=stage,
            denoise_step=denoise_step,
        )


def _validated_nu(nu: Array | float) -> Array:
    """Eagerly reject invalid levels; traced callers assume the declared domain."""
    value = jnp.asarray(nu)
    if not isinstance(value, jax.core.Tracer):  # pyright: ignore[reportAttributeAccessIssue]
        if bool(jnp.any(~jnp.isfinite(value))):
            raise ValueError("nu must be finite")
        if bool(jnp.any((value < PHASE_F_NU_MIN) | (value > PHASE_F_NU_MAX))):
            raise ValueError("nu must lie in [0, 1]")
    return value


def _validated_eta(eta_diff: Array | float | DiffusionEtaMode) -> Array:
    """Eagerly enforce the two authoritative V0 eta modes."""
    value = jnp.asarray(eta_diff)
    if not isinstance(value, jax.core.Tracer):  # pyright: ignore[reportAttributeAccessIssue]
        if bool(jnp.any((value != 0) & (value != 1))):
            raise ValueError("PHASE_F_CORRUPTION_CONTRACT_V0 supports eta_diff only in {0, 1}")
    return value


def _alpha_core(nu: Array) -> Array:
    return jnp.cos(jnp.pi * nu / 2.0)


def _sigma_core(nu: Array) -> Array:
    return jnp.sin(jnp.pi * nu / 2.0)


def alpha(nu: Array | float) -> Array:
    """Return ``cos(pi * nu / 2)`` for Phase-F levels in ``[0, 1]``."""
    return _alpha_core(_validated_nu(nu))


def sigma(nu: Array | float) -> Array:
    """Return ``sin(pi * nu / 2)`` for Phase-F levels in ``[0, 1]``."""
    return _sigma_core(_validated_nu(nu))


def _broadcast_coefficient(coefficient: Array, target: Array) -> Array:
    """Treat coefficient axes as leading target axes and append feature axes."""
    if coefficient.ndim > target.ndim:
        raise ValueError("nu/eta rank cannot exceed carrier rank")
    return coefficient.reshape((*coefficient.shape, *((1,) * (target.ndim - coefficient.ndim))))


def _validate_carrier_pair(carrier: Array, epsilon: Array) -> tuple[Array, Array]:
    carrier = jnp.asarray(carrier)
    epsilon = jnp.asarray(epsilon)
    if carrier.shape != epsilon.shape:
        raise ValueError("carrier and epsilon must have identical shapes")
    if not jnp.issubdtype(carrier.dtype, jnp.floating):
        raise TypeError("continuous carrier must have a floating dtype")
    if not jnp.issubdtype(epsilon.dtype, jnp.floating):
        raise TypeError("epsilon must have a floating dtype")
    return carrier, epsilon


def corrupt_h(h0: Array, nu: Array | float, epsilon: Array) -> Array:
    """Apply the exact V0 forward kernel ``alpha*h0 + sigma*epsilon``."""
    h0, epsilon = _validate_carrier_pair(h0, epsilon)
    nu_value = _validated_nu(nu)
    alpha_value = _broadcast_coefficient(_alpha_core(nu_value), h0)
    sigma_value = _broadcast_coefficient(_sigma_core(nu_value), h0)
    return alpha_value * h0 + sigma_value * epsilon


def recorrupt_h(
    h_hat_0: Array,
    nu_next: Array | float,
    epsilon: Array,
    eta_diff: Array | float | DiffusionEtaMode,
) -> Array:
    """Re-corrupt a model x0 prediction using the two authoritative V0 eta modes."""
    h_hat_0, epsilon = _validate_carrier_pair(h_hat_0, epsilon)
    nu_value = _validated_nu(nu_next)
    eta_value = _validated_eta(eta_diff)
    alpha_value = _broadcast_coefficient(_alpha_core(nu_value), h_hat_0)
    sigma_value = _broadcast_coefficient(_sigma_core(nu_value), h_hat_0)
    eta_value = _broadcast_coefficient(eta_value, h_hat_0)
    return alpha_value * h_hat_0 + eta_value * sigma_value * epsilon


def diffusion_key(
    root_key: Array,
    *,
    global_example_id: int | Array,
    stage: DiffusionStage,
    denoise_step: int | Array,
) -> Array:
    """Derive a V0 diffusion key without using the Torx operator namespace."""
    if not isinstance(stage, DiffusionStage):
        raise TypeError("stage must be a DiffusionStage")
    key = jax.random.fold_in(root_key, jnp.uint32(_DIFFUSION_NAMESPACE_ID))
    key = jax.random.fold_in(key, jnp.uint32(_DIFFUSION_EXAMPLE_ID_MARKER))
    key = jax.random.fold_in(key, jnp.asarray(global_example_id, dtype=jnp.uint32))
    key = jax.random.fold_in(key, jnp.uint32(_DIFFUSION_STAGE_MARKER))
    key = jax.random.fold_in(key, jnp.uint32(stage.value))
    key = jax.random.fold_in(key, jnp.uint32(_DIFFUSION_STEP_MARKER))
    return jax.random.fold_in(key, jnp.asarray(denoise_step, dtype=jnp.uint32))


def sample_initial_corruption(h0: Array, nu: Array | float, key: Array) -> CorruptionTrace:
    """Sample reproducible full-strength initial corruption and expose epsilon."""
    h0 = jnp.asarray(h0)
    if not jnp.issubdtype(h0.dtype, jnp.floating):
        raise TypeError("continuous carrier must have a floating dtype")
    epsilon = jax.random.normal(key, h0.shape, dtype=h0.dtype)
    return CorruptionTrace(value=corrupt_h(h0, nu, epsilon), epsilon=epsilon)


def sample_recorruption(
    h_hat_0: Array,
    nu_next: Array | float,
    key: Array,
    eta_diff: Array | float | DiffusionEtaMode,
) -> CorruptionTrace:
    """Sample reproducible inter-step re-corruption and expose epsilon."""
    h_hat_0 = jnp.asarray(h_hat_0)
    if not jnp.issubdtype(h_hat_0.dtype, jnp.floating):
        raise TypeError("continuous carrier must have a floating dtype")
    epsilon = jax.random.normal(key, h_hat_0.shape, dtype=h_hat_0.dtype)
    return CorruptionTrace(
        value=recorrupt_h(h_hat_0, nu_next, epsilon, eta_diff),
        epsilon=epsilon,
    )


def phase_f_schedule(nu0: Array | float, s_max: int = PHASE_F_MAX_DENOISE_STEPS) -> Array:
    """Return a strict prefix of the frozen four-step V0 schedule."""
    if not 1 <= s_max <= PHASE_F_MAX_DENOISE_STEPS:
        raise ValueError("s_max must lie in [1, 4]")
    nu_value = _validated_nu(nu0)
    multipliers = jnp.asarray(PHASE_F_SCHEDULE_MULTIPLIERS[:s_max], dtype=nu_value.dtype)
    return nu_value[..., None] * multipliers
