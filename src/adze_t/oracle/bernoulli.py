"""Tiny analytic Bernoulli oracle useful for estimator smoke tests."""

from __future__ import annotations

import jax.numpy as jnp


def probability(logit):
    return 1.0 / (1.0 + jnp.exp(-logit))


def linear_expectation(logit, a, b):
    p = probability(logit)
    return a * p + b


def linear_gradient(logit, a):
    p = probability(logit)
    return a * p * (1.0 - p)
