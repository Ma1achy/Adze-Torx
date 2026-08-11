"""Reproducible Phase-B target-codec and deterministic trainability runs."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.config import REFERENCE_SMALL_V0
from adze_t.model import apply_model, apply_target_codec
from adze_t.objectives import (
    codec_loss_components,
    emitted_metrics,
    loss_components,
    total_loss,
)
from adze_t.training import (
    codec_pretrain_step,
    initialise_training,
    make_fixed_structure_batch,
    train_step,
)

CONFIG = REFERENCE_SMALL_V0
RUN_ROOT = Path("results/phase_b/runs")
CHECKPOINT_ROOT = Path("results/phase_b/checkpoints")
B3_CHECKPOINTS = (
    100,
    250,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    40_000,
    60_000,
    120_000,
    240_000,
)
CODEC_CHECKPOINTS = (
    1,
    100,
    250,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    40_000,
    60_000,
)


def dataset(task: str, count: int, seed: int) -> tuple[jax.Array, jax.Array]:
    """Fixed, deterministic eight-byte data; zero remains reserved."""
    source = jax.random.randint(
        jax.random.PRNGKey(seed), (count, 8), minval=1, maxval=33, dtype=jnp.int32
    )
    if task == "copy":
        target = source
    elif task == "reverse":
        target = source[:, ::-1]
    else:
        raise ValueError(f"unsupported generated task: {task}")
    return source, target


def save_checkpoint(path: Path, params: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(jax.device_get(params), stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: Path) -> Any:
    with path.open("rb") as stream:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(stream))


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def serialise(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)):
            return value
        array = jax.device_get(value)
        return float(array) if getattr(array, "ndim", 0) == 0 else array.tolist()

    with path.open("a") as stream:
        stream.write(json.dumps({key: serialise(value) for key, value in row.items()}) + "\n")


def _codec_eval(params: Any, target: jax.Array) -> dict[str, jax.Array]:
    mask = jnp.ones_like(target, dtype=bool)
    outputs = apply_target_codec(params, target, mask, config=CONFIG)
    components = codec_loss_components(outputs)
    byte, sequence = emitted_metrics(
        outputs["codec_logits"],
        outputs["target"]["teacher"].slot_bytes,
        outputs["target"]["teacher"].slot_mask,
    )
    return {
        "loss": jnp.sum(jnp.stack(list(components.values()))),
        **components,
        "byte_accuracy": byte,
        "sequence_accuracy": sequence,
    }


def _model_eval(params: Any, prompt: jax.Array, target: jax.Array) -> dict[str, jax.Array]:
    prompt_mask = jnp.ones_like(prompt, dtype=bool)
    target_mask = jnp.ones_like(target, dtype=bool)
    outputs = apply_model(params, prompt, prompt_mask, target, target_mask, config=CONFIG)
    components = loss_components(outputs)
    teacher = outputs["target"]["teacher"]
    byte, sequence = emitted_metrics(outputs["byte_logits"], teacher.slot_bytes, teacher.slot_mask)
    _, b_logits, l_logits = outputs["prediction"]
    return {
        "loss": total_loss(components, CONFIG),
        **components,
        "byte_accuracy": byte,
        "sequence_accuracy": sequence,
        "boundary_accuracy": jnp.mean(
            jnp.argmax(b_logits[:, :-1], -1) == teacher.boundaries[:, :-1]
        ),
        "length_accuracy": jnp.mean(jnp.argmax(l_logits, -1) == teacher.length),
        "activation_packed_input": outputs["activation_rms"]["packed_input"],
        "activation_unpooled_carrier": outputs["activation_rms"]["unpooled_carrier"],
        "activation_block_rms": outputs["dit_aux"]["block_rms"],
        "activation_cycle_rms": outputs["dit_aux"]["cycle_rms"],
    }


def averaged_eval(
    function: Any,
    params: Any,
    first: jax.Array,
    second: jax.Array | None = None,
    chunk: int = 4,
) -> dict[str, jax.Array]:
    totals: dict[str, jax.Array] = {}
    count = 0
    for start in range(0, first.shape[0], chunk):
        left = first[start : start + chunk]
        result = (
            function(params, left)
            if second is None
            else function(params, left, second[start : start + chunk])
        )
        weight = left.shape[0]
        for name, value in result.items():
            totals[name] = totals.get(name, jnp.zeros_like(value)) + value * weight
        count += weight
    return {name: value / count for name, value in totals.items()}


def run_codec(max_steps: int) -> None:
    train = dataset("copy", 1_024, 710)[1]
    validation = dataset("copy", 256, 711)[1]
    params, moments = initialise_training(jax.random.PRNGKey(700), CONFIG)
    update = jax.jit(codec_pretrain_step, static_argnames=("config",))
    evaluate = jax.jit(_codec_eval)
    metrics_path = RUN_ROOT / "target_codec_b1.jsonl"
    metrics_path.unlink(missing_ok=True)
    started = time.monotonic()
    batch_size = 32
    for step in range(1, max_steps + 1):
        offset = ((step - 1) * batch_size) % train.shape[0]
        target = train[offset : offset + batch_size]
        batch = make_fixed_structure_batch(target, target, config=CONFIG)
        params, moments, metrics = update(params, moments, step, batch, config=CONFIG)
        if step in CODEC_CHECKPOINTS or step == max_steps:
            jax.block_until_ready(metrics["loss"])
            validation_metrics = averaged_eval(evaluate, params, validation)
            elapsed = time.monotonic() - started
            row = {
                "stage": "target_codec",
                "step": step,
                "total_loss": metrics["loss"],
                "train_emitted_byte_accuracy": metrics["byte_accuracy"],
                "train_exact_sequence_accuracy": metrics["sequence_accuracy"],
                "validation_emitted_byte_accuracy": validation_metrics["byte_accuracy"],
                "validation_exact_sequence_accuracy": validation_metrics["sequence_accuracy"],
                "validation_codec_byte_loss": validation_metrics["codec_byte"],
                "validation_codec_boundary_loss": validation_metrics["codec_b"],
                "validation_codec_extent_loss": validation_metrics["codec_l"],
                "gradient_norm": metrics["grad_norm"],
                "learning_rate": CONFIG.training.learning_rate,
                "wall_clock_seconds": elapsed,
                "steps_per_second": step / elapsed,
            }
            append_row(metrics_path, row)
            print(
                json.dumps(
                    {
                        key: (float(value) if hasattr(value, "ndim") and value.ndim == 0 else value)
                        for key, value in row.items()
                    }
                ),
                flush=True,
            )
            save_checkpoint(CHECKPOINT_ROOT / "target_codec_b1.pkl", params)
            if (
                step >= 500
                and float(validation_metrics["byte_accuracy"]) >= 0.99
                and float(validation_metrics["sequence_accuracy"]) >= 0.95
            ):
                break


def _stage_data(stage: str) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
    if stage == "single":
        source, target = dataset("reverse", 1, 800)
        return source, target, None, None
    if stage == "overfit":
        source, target = dataset("reverse", 16, 810)
        return source, target, None, None
    if stage in ("copy", "reverse"):
        train_source, train_target = dataset(stage, 65_536, 820 if stage == "copy" else 830)
        valid_source, valid_target = dataset(stage, 256, 821 if stage == "copy" else 831)
        return train_source, train_target, valid_source, valid_target
    raise ValueError(stage)


def run_b3(stage: str, max_steps: int) -> None:
    codec_path = CHECKPOINT_ROOT / "target_codec_b1.pkl"
    if not codec_path.exists():
        raise FileNotFoundError("run target-codec pretraining before B3")
    params = load_checkpoint(codec_path)
    zero, _ = initialise_training(jax.random.PRNGKey(0), CONFIG)
    del zero
    from adze_t.objectives import adamw_init

    moments = (adamw_init(params), adamw_init(params))
    train_source, train_target, valid_source, valid_target = _stage_data(stage)
    update = jax.jit(train_step, static_argnames=("config",))
    evaluate = jax.jit(_model_eval)
    metrics_path = RUN_ROOT / f"{stage}.jsonl"
    metrics_path.unlink(missing_ok=True)
    started = time.monotonic()
    checkpoints = B3_CHECKPOINTS
    if stage == "single":
        checkpoints = tuple(step for step in B3_CHECKPOINTS if step <= 1_000)
    elif stage == "overfit":
        checkpoints = tuple(step for step in B3_CHECKPOINTS if step <= 20_000)
    majority = jnp.max(jnp.bincount(train_target.reshape(-1), length=256)) / train_target.size
    batch_size = 1 if stage == "single" else min(CONFIG.training.batch_size, train_source.shape[0])
    initial_count = min(train_source.shape[0], 128)
    initial_metrics = averaged_eval(
        evaluate, params, train_source[:initial_count], train_target[:initial_count]
    )
    initial_validation = (
        averaged_eval(evaluate, params, valid_source, valid_target)
        if valid_source is not None and valid_target is not None
        else initial_metrics
    )
    initial_h = float(initial_metrics["h"])
    append_row(
        metrics_path,
        {
            "stage": stage,
            "step": 0,
            "total_loss": initial_metrics["loss"],
            "h_loss": initial_metrics["h"],
            "boundary_loss": initial_metrics["b"],
            "extent_loss": initial_metrics["l"],
            "byte_loss": initial_metrics["byte"],
            "proposal_loss": initial_metrics["proposal"],
            "train_emitted_byte_accuracy": initial_metrics["byte_accuracy"],
            "train_exact_sequence_accuracy": initial_metrics["sequence_accuracy"],
            "validation_emitted_byte_accuracy": initial_validation["byte_accuracy"],
            "validation_exact_sequence_accuracy": initial_validation["sequence_accuracy"],
            "boundary_accuracy": initial_metrics["boundary_accuracy"],
            "extent_accuracy": initial_metrics["length_accuracy"],
            "majority_byte_baseline": majority,
            "gradient_norm": 0.0,
            "learning_rate": CONFIG.training.learning_rate,
            "wall_clock_seconds": 0.0,
            "steps_per_second": 0.0,
            "activation_packed_input": initial_metrics["activation_packed_input"],
            "activation_unpooled_carrier": initial_metrics["activation_unpooled_carrier"],
            "activation_block_rms": initial_metrics["activation_block_rms"],
            "activation_cycle_rms": initial_metrics["activation_cycle_rms"],
        },
    )
    for step in range(1, max_steps + 1):
        index = ((step - 1) * batch_size) % train_source.shape[0]
        source = train_source[index : index + batch_size]
        target = train_target[index : index + batch_size]
        batch = make_fixed_structure_batch(source, target, config=CONFIG)
        params, moments, metrics = update(params, moments, step, batch, config=CONFIG)
        if step in checkpoints or step == max_steps:
            jax.block_until_ready(metrics["loss"])
            train_eval_count = min(train_source.shape[0], 128)
            train_metrics = averaged_eval(
                evaluate, params, train_source[:train_eval_count], train_target[:train_eval_count]
            )
            validation_metrics = (
                averaged_eval(evaluate, params, valid_source, valid_target)
                if valid_source is not None and valid_target is not None
                else train_metrics
            )
            elapsed = time.monotonic() - started
            row = {
                "stage": stage,
                "step": step,
                "total_loss": train_metrics["loss"],
                "h_loss": train_metrics["h"],
                "boundary_loss": train_metrics["b"],
                "extent_loss": train_metrics["l"],
                "byte_loss": train_metrics["byte"],
                "proposal_loss": train_metrics["proposal"],
                "train_emitted_byte_accuracy": train_metrics["byte_accuracy"],
                "train_exact_sequence_accuracy": train_metrics["sequence_accuracy"],
                "validation_emitted_byte_accuracy": validation_metrics["byte_accuracy"],
                "validation_exact_sequence_accuracy": validation_metrics["sequence_accuracy"],
                "boundary_accuracy": train_metrics["boundary_accuracy"],
                "extent_accuracy": train_metrics["length_accuracy"],
                "majority_byte_baseline": majority,
                "gradient_norm": metrics["grad_norm"],
                "learning_rate": CONFIG.training.learning_rate,
                "wall_clock_seconds": elapsed,
                "steps_per_second": step / elapsed,
                "activation_packed_input": train_metrics["activation_packed_input"],
                "activation_unpooled_carrier": train_metrics["activation_unpooled_carrier"],
                "activation_block_rms": train_metrics["activation_block_rms"],
                "activation_cycle_rms": train_metrics["activation_cycle_rms"],
                **{name: value for name, value in metrics.items() if name.startswith("grad_")},
            }
            append_row(metrics_path, row)
            print(
                f"{stage} step={step} train={float(train_metrics['byte_accuracy']):.4f} validation={float(validation_metrics['byte_accuracy']):.4f} loss={float(train_metrics['loss']):.5f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            save_checkpoint(CHECKPOINT_ROOT / f"{stage}.pkl", params)
            h_reduction = 1.0 - float(train_metrics["h"]) / max(initial_h, 1.0e-12)
            if (
                stage == "single"
                and float(train_metrics["byte_accuracy"]) >= 0.99
                and h_reduction >= 0.9
                and float(train_metrics["boundary_accuracy"]) == 1.0
                and float(train_metrics["length_accuracy"]) == 1.0
            ):
                break
            if stage == "overfit" and float(train_metrics["byte_accuracy"]) >= 0.95:
                break
            if (
                stage in ("copy", "reverse")
                and float(validation_metrics["byte_accuracy"]) >= 0.9
                and float(validation_metrics["byte_accuracy"] - majority) >= 0.2
            ):
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("codec", "single", "overfit", "copy", "reverse"))
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    defaults = {
        "codec": 60_000,
        "single": 1_000,
        "overfit": 20_000,
        "copy": 240_000,
        "reverse": 240_000,
    }
    max_steps = args.max_steps or defaults[args.stage]
    if args.stage == "codec":
        run_codec(max_steps)
    else:
        run_b3(args.stage, max_steps)


if __name__ == "__main__":
    main()
