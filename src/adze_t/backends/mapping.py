"""Semantic-path deterministic-to-Torx parameter correspondence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .torx import PHASE_D_INITIAL_SIGMA, rho_from_sigma


_DIRECT_MEAN_ONLY = {"a_log", "d_skip", "delta_bias", "layer_scale"}


@dataclass(frozen=True)
class ParameterMapEntry:
    deterministic_path: str | None
    torx_path: str
    role: str
    count: int


def _join(path: tuple[str, ...]) -> str:
    return "/".join(path)


def _rho(width: int, initial_sigma: float) -> jax.Array:
    raw = rho_from_sigma(initial_sigma)
    return jnp.full((width,), raw)


def deterministic_to_torx(
    params: Any, *, initial_sigma: float = PHASE_D_INITIAL_SIGMA
) -> tuple[Any, tuple[ParameterMapEntry, ...]]:
    """Map by semantic dictionary/list paths, never pytree leaf order."""
    entries: list[ParameterMapEntry] = []

    def convert(node: Any, path: tuple[str, ...]) -> Any:
        if isinstance(node, dict) and set(node) == {"weight", "bias"}:
            mean = {"weight": node["weight"], "bias": node["bias"]}
            entries.extend(
                [
                    ParameterMapEntry(
                        _join(path + ("weight",)),
                        _join(path + ("mean", "weight")),
                        "mean",
                        node["weight"].size,
                    ),
                    ParameterMapEntry(
                        _join(path + ("bias",)),
                        _join(path + ("mean", "bias")),
                        "mean",
                        node["bias"].size,
                    ),
                    ParameterMapEntry(
                        None, _join(path + ("rho",)), "stochastic", node["bias"].size
                    ),
                ]
            )
            return {"mean": mean, "rho": _rho(node["bias"].shape[0], initial_sigma)}
        if isinstance(node, dict) and set(node) == {"kernel", "bias"}:
            mean = {"kernel": node["kernel"], "bias": node["bias"]}
            entries.extend(
                [
                    ParameterMapEntry(
                        _join(path + ("kernel",)),
                        _join(path + ("mean", "kernel")),
                        "mean",
                        node["kernel"].size,
                    ),
                    ParameterMapEntry(
                        _join(path + ("bias",)),
                        _join(path + ("mean", "bias")),
                        "mean",
                        node["bias"].size,
                    ),
                    ParameterMapEntry(
                        None, _join(path + ("rho",)), "stochastic", node["bias"].size
                    ),
                ]
            )
            return {"mean": mean, "rho": _rho(node["bias"].shape[0], initial_sigma)}
        if isinstance(node, dict):
            return {key: convert(value, path + (str(key),)) for key, value in node.items()}
        if isinstance(node, (list, tuple)):
            converted = [convert(value, path + (str(index),)) for index, value in enumerate(node)]
            return type(node)(converted)
        if isinstance(node, (jax.Array, np.ndarray)):
            if not path:
                raise ValueError("unaddressed parameter leaf")
            name = path[-1]
            if name in _DIRECT_MEAN_ONLY:
                entries.append(
                    ParameterMapEntry(_join(path), _join(path + ("mean",)), "mean", node.size)
                )
                return {"mean": node}
            if node.ndim < 2:
                raise ValueError(f"unknown bare learned parameter at semantic path {_join(path)}")
            entries.extend(
                [
                    ParameterMapEntry(_join(path), _join(path + ("mean",)), "mean", node.size),
                    ParameterMapEntry(None, _join(path + ("rho",)), "stochastic", node.shape[-1]),
                ]
            )
            return {"mean": node, "rho": _rho(node.shape[-1], initial_sigma)}
        raise TypeError(f"unsupported parameter node at {_join(path)}: {type(node)!r}")

    mapped = convert(params, ())
    _validate_entries(params, mapped, entries)
    return mapped, tuple(entries)


def torx_means_to_deterministic(params: Any) -> Any:
    """Remove Torx stochastic wrappers without relying on traversal order."""
    if isinstance(params, dict) and set(params) == {"mean", "rho"}:
        return params["mean"]
    if isinstance(params, dict) and set(params) == {"mean"}:
        return params["mean"]
    if isinstance(params, dict):
        return {key: torx_means_to_deterministic(value) for key, value in params.items()}
    if isinstance(params, (list, tuple)):
        values = [torx_means_to_deterministic(value) for value in params]
        return type(params)(values)
    raise TypeError(f"unexpected unmapped Torx parameter node: {type(params)!r}")


def _validate_entries(
    deterministic: Any, torx_params: Any, entries: list[ParameterMapEntry]
) -> None:
    det_paths = {
        _join(tuple(str(key.key if hasattr(key, "key") else key.idx) for key in path))
        for path, _ in jax.tree_util.tree_leaves_with_path(deterministic)
    }
    mapped_paths = [entry.deterministic_path for entry in entries if entry.role == "mean"]
    if len(mapped_paths) != len(set(mapped_paths)) or set(mapped_paths) != det_paths:
        raise ValueError("deterministic-to-Torx mean mapping is not one-to-one and complete")
    round_trip = torx_means_to_deterministic(torx_params)
    if not all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(
            jax.tree_util.tree_leaves(deterministic),
            jax.tree_util.tree_leaves(round_trip),
            strict=True,
        )
    ):
        raise ValueError("Torx mean round-trip changed deterministic values")


def parameter_counts(entries: tuple[ParameterMapEntry, ...]) -> dict[str, int]:
    mean = sum(entry.count for entry in entries if entry.role == "mean")
    stochastic = sum(entry.count for entry in entries if entry.role == "stochastic")
    return {
        "deterministic": mean,
        "torx_mean": mean,
        "torx_stochastic": stochastic,
        "torx_total": mean + stochastic,
    }
