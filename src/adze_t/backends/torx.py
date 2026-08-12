"""Public-Torx learned operators for zero-noise Adze-D/Adze-T parity.

Each learned operation invokes a small public ``torx.AbstractReferenceFactor``.
The model graph never substitutes ``DeterministicOps`` at zero noise: the exact
mean branch lives inside the factor's public ``sample`` method.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
from typing import Any

import jax
from jax import Array
import jax.numpy as jnp
import torx


PHASE_D_NOISE_POLICY_V0 = "PHASE_D_NOISE_POLICY_V0"
PHASE_D_DIRECT_PARAMETER_NOISE_POLICY_V0_MEAN_ONLY = (
    "PHASE_D_DIRECT_PARAMETER_NOISE_POLICY_V0_MEAN_ONLY"
)
PHASE_D_INITIAL_SIGMA = 1.0e-3
PHASE_D_SIGMA_MIN = 1.0e-6
PHASE_D_SIGMA_MAX = 0.25


def stable_occurrence_id(name: str) -> int:
    """Return a process-independent uint32 identity for a static module path."""
    return int.from_bytes(hashlib.blake2s(name.encode("utf-8"), digest_size=4).digest(), "little")


@dataclass(frozen=True)
class TorxOperatorConfig:
    lambda_op: float | Array = 0.0
    operator_stochasticity: bool | Array = False
    sigma_min: float = PHASE_D_SIGMA_MIN
    sigma_max: float = PHASE_D_SIGMA_MAX


@dataclass(frozen=True)
class OccurrenceContext:
    root_key: Array
    scopes: tuple[str, ...] = ()
    evaluation_id: int | Array = 0
    global_example_id: int | Array = 0
    optimizer_step: int | Array = 0
    denoise_step: int | Array = 0
    refinement_step: int | Array = 0
    recurrence_cycle: int | Array = 0
    physical_layer: int | Array = 0
    site_coordinate: int | Array = 0

    def key_for(self, name: str) -> Array:
        """Derive a key from explicit identity fields in a frozen order."""
        key = self.root_key
        for value in (
            self.evaluation_id,
            self.global_example_id,
            self.optimizer_step,
        ):
            key = jax.random.fold_in(key, jnp.asarray(value, dtype=jnp.uint32))
        for scope in self.scopes:
            key = jax.random.fold_in(key, stable_occurrence_id(f"scope:{scope}"))
        key = jax.random.fold_in(key, stable_occurrence_id(f"module:{name}"))
        for value in (
            self.denoise_step,
            self.refinement_step,
            self.recurrence_cycle,
            self.physical_layer,
            self.site_coordinate,
        ):
            key = jax.random.fold_in(key, jnp.asarray(value, dtype=jnp.uint32))
        return key

    def with_scope(self, scope: str) -> "OccurrenceContext":
        if not scope:
            raise ValueError("occurrence scope must be a non-empty static string")
        return replace(self, scopes=(*self.scopes, scope))

    def occurrence_path(self, name: str) -> str:
        return "/".join((*self.scopes, name))


class GaussianAffineFactor(torx.AbstractReferenceFactor):
    """One local affine transformation with per-output-channel Gaussian noise."""

    input_ports: Mapping[str, Any]
    output_spec: Any

    def __init__(self, x: Array, out_dim: int):
        self.input_ports = {"x": jax.ShapeDtypeStruct(x.shape, x.dtype)}
        self.output_spec = jax.ShapeDtypeStruct((*x.shape[:-1], out_dim), x.dtype)

    def init_params(self, key: Array) -> None:
        del key
        return None

    def sample(
        self,
        key: Array,
        inputs: Mapping[str, Any],
        params: Mapping[str, Any],
        info: Any = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Any:
        del site_info
        mean_params = params["mean"]
        mean = inputs["x"] @ mean_params["weight"] + mean_params["bias"]
        output = _exact_or_noisy(key, mean, params["rho"], info)
        return (output, {"mean": mean}) if return_aux else output


class GaussianEmbeddingFactor(torx.AbstractReferenceFactor):
    """One learned categorical lookup with per-channel Gaussian noise."""

    input_ports: Mapping[str, Any]
    output_spec: Any

    def __init__(self, indices: Array, width: int, dtype: Any):
        self.input_ports = {"indices": jax.ShapeDtypeStruct(indices.shape, indices.dtype)}
        self.output_spec = jax.ShapeDtypeStruct((*indices.shape, width), dtype)

    def init_params(self, key: Array) -> None:
        del key
        return None

    def sample(
        self,
        key: Array,
        inputs: Mapping[str, Any],
        params: Mapping[str, Any],
        info: Any = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Any:
        del site_info
        mean = params["mean"][inputs["indices"].astype(jnp.int32)]
        output = _exact_or_noisy(key, mean, params["rho"], info)
        return (output, {"mean": mean}) if return_aux else output


class GaussianDepthwiseConv1DFactor(torx.AbstractReferenceFactor):
    """One local learned causal depthwise convolution factor."""

    input_ports: Mapping[str, Any]
    output_spec: Any

    def __init__(self, x: Array):
        self.input_ports = {"x": jax.ShapeDtypeStruct(x.shape, x.dtype)}
        self.output_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)

    def init_params(self, key: Array) -> None:
        del key
        return None

    def sample(
        self,
        key: Array,
        inputs: Mapping[str, Any],
        params: Mapping[str, Any],
        info: Any = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Any:
        del site_info
        x = inputs["x"]
        kernel = params["mean"]["kernel"][:, None, :]
        padded = jnp.pad(x, ((0, 0), (kernel.shape[0] - 1, 0), (0, 0)))
        mean = jax.lax.conv_general_dilated(
            padded,
            kernel,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=x.shape[-1],
        )
        mean = mean + params["mean"]["bias"]
        output = _exact_or_noisy(key, mean, params["rho"], info)
        return (output, {"mean": mean}) if return_aux else output


class MeanParameterFactor(torx.AbstractReferenceFactor):
    """Public-factor access to a learned mean with no Phase-C noise law."""

    input_ports: Mapping[str, Any]
    output_spec: Any

    def __init__(self, value: Array):
        self.input_ports = {}
        self.output_spec = jax.ShapeDtypeStruct(value.shape, value.dtype)

    def init_params(self, key: Array) -> None:
        del key
        return None

    def sample(
        self,
        key: Array,
        inputs: Mapping[str, Any],
        params: Mapping[str, Any],
        info: Any = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Any:
        del key, inputs, info, site_info
        output = params["mean"]
        return (output, {"mean": output}) if return_aux else output


def sigma_from_rho(
    rho: Array,
    *,
    sigma_min: float = PHASE_D_SIGMA_MIN,
    sigma_max: float = PHASE_D_SIGMA_MAX,
) -> Array:
    """Public Phase-D rho transform used by operators and statistical oracles."""
    return jnp.clip(jax.nn.softplus(rho), sigma_min, sigma_max)


def rho_from_sigma(sigma: float | Array) -> Array:
    """Return the unconstrained rho whose unclipped softplus is ``sigma``."""
    sigma_array = jnp.asarray(sigma, dtype=jnp.float32)
    return jnp.log(jnp.expm1(sigma_array))


def _exact_or_noisy(key: Array, mean: Array, rho: Array, config: TorxOperatorConfig) -> Array:
    """Trace both paths safely while preserving an exact zero-noise mean."""
    enabled = jnp.asarray(config.operator_stochasticity) & (jnp.asarray(config.lambda_op) != 0)

    def noisy(_: None) -> Array:
        sigma = sigma_from_rho(rho, sigma_min=config.sigma_min, sigma_max=config.sigma_max)
        return mean + jnp.asarray(config.lambda_op, mean.dtype) * sigma * jax.random.normal(
            key, mean.shape, dtype=mean.dtype
        )

    return jax.lax.cond(enabled, noisy, lambda _: mean, operand=None)


FactorObserver = Callable[[str, str], None]
OccurrenceObserver = Callable[[str, str, str, Array], None]
SampleObserver = Callable[[str, str, str, Array, Array, Array], None]


@dataclass(frozen=True)
class TorxOps:
    """Learned-operator backend executing every operation via public Torx factors."""

    context: OccurrenceContext
    config: TorxOperatorConfig = TorxOperatorConfig()
    observer: FactorObserver | None = None
    occurrence_observer: OccurrenceObserver | None = None
    sample_observer: SampleObserver | None = None

    @classmethod
    def create(
        cls,
        key: Array,
        *,
        config: TorxOperatorConfig | None = None,
        evaluation_id: int | Array = 0,
        global_example_id: int | Array = 0,
        optimizer_step: int | Array = 0,
        observer: FactorObserver | None = None,
        occurrence_observer: OccurrenceObserver | None = None,
        sample_observer: SampleObserver | None = None,
    ) -> "TorxOps":
        return cls(
            OccurrenceContext(
                key,
                evaluation_id=evaluation_id,
                global_example_id=global_example_id,
                optimizer_step=optimizer_step,
            ),
            config or TorxOperatorConfig(),
            observer,
            occurrence_observer,
            sample_observer,
        )

    def _sample(self, factor: Any, inputs: Mapping[str, Any], params: Any, name: str, kind: str):
        if self.observer is not None:
            self.observer(kind, name)
        key = self.context.key_for(name)
        if self.occurrence_observer is not None:
            self.occurrence_observer(kind, name, self.context.occurrence_path(name), key)
        if self.sample_observer is None:
            return factor.sample(key, inputs, params, self.config, return_aux=False)
        output, aux = factor.sample(key, inputs, params, self.config, return_aux=True)
        self.sample_observer(
            kind,
            name,
            self.context.occurrence_path(name),
            key,
            aux["mean"],
            output,
        )
        return output

    def linear(self, x: Array, params: Any, *, name: str = "linear") -> Array:
        return self._sample(
            GaussianAffineFactor(x, params["mean"]["bias"].shape[0]),
            {"x": x},
            params,
            name,
            "linear",
        )

    def categorical_logits(self, x: Array, params: Any, *, name: str) -> Array:
        return self._sample(
            GaussianAffineFactor(x, params["mean"]["bias"].shape[0]),
            {"x": x},
            params,
            name,
            "categorical_logits",
        )

    def embedding(self, indices: Array, params: Any, *, name: str) -> Array:
        return self._sample(
            GaussianEmbeddingFactor(indices, params["mean"].shape[-1], params["mean"].dtype),
            {"indices": indices},
            params,
            name,
            "embedding",
        )

    def depthwise_conv1d(self, x: Array, params: Any, *, name: str) -> Array:
        return self._sample(
            GaussianDepthwiseConv1DFactor(x), {"x": x}, params, name, "depthwise_conv1d"
        )

    def parameter(self, value: Any, *, name: str) -> Array:
        return self._sample(MeanParameterFactor(value["mean"]), {}, value, name, "parameter")

    def with_occurrence(
        self,
        *,
        recurrence_cycle: int | Array | None = None,
        physical_layer: int | Array | None = None,
        denoise_step: int | Array | None = None,
        refinement_step: int | Array | None = None,
        site_coordinate: int | Array | None = None,
    ) -> "TorxOps":
        updates = {
            name: value
            for name, value in {
                "recurrence_cycle": recurrence_cycle,
                "physical_layer": physical_layer,
                "denoise_step": denoise_step,
                "refinement_step": refinement_step,
                "site_coordinate": site_coordinate,
            }.items()
            if value is not None
        }
        return replace(self, context=replace(self.context, **updates))

    def with_scope(self, scope: str) -> "TorxOps":
        return replace(self, context=self.context.with_scope(scope))

    def init_linear(self, key: Array, in_dim: int, out_dim: int, *, scale: float = 1.0) -> Any:
        del key, in_dim, out_dim, scale
        raise NotImplementedError("initialise Adze-D once, then map its means into Torx parameters")

    def init_embedding(self, key: Array, size: int, width: int) -> Any:
        del key, size, width
        raise NotImplementedError("initialise Adze-D once, then map its means into Torx parameters")

    def init_depthwise_conv(self, key: Array, kernel_size: int, channels: int) -> Any:
        del key, kernel_size, channels
        raise NotImplementedError("initialise Adze-D once, then map its means into Torx parameters")
