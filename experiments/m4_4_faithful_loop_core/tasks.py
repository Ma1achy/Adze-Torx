"""M4.3 task adapted to the explicit M4.4 92-D state layout."""

from __future__ import annotations

import jax.numpy as jnp

from experiments.m4_3_hard_task_depth_sweep.tasks import HardBatch
from experiments.m4_3_hard_task_depth_sweep.tasks import make_data as make_m43_data
from experiments.m4_4_faithful_loop_core.model import MASK, OPS, TASK, WIDTH


def make_data(key, n, k, train):
    source = make_m43_data(key, n, k, train, width=64)
    initial = (
        jnp.zeros((n, WIDTH), dtype=source.initial.dtype).at[:, :TASK].set(source.initial[:, :TASK])
    )
    initial = initial.at[:, TASK + 24 : TASK + 24 + OPS].set(source.initial[:, 8 : 8 + OPS])
    valid = jnp.zeros((n, MASK), dtype=source.initial.dtype).at[:, :k].set(1.0)
    initial = initial.at[:, TASK + 24 + OPS :].set(valid)
    target = jnp.zeros_like(initial).at[:, :TASK].set(source.target[:, :TASK])
    return HardBatch(initial, target, source.operators, source.intermediates)
