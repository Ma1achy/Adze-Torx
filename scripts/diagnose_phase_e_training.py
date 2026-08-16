"""Small, explicit Phase-E.0.1 stochastic-training failure reproducer."""

from __future__ import annotations

import argparse
import os
import sys


def mark(label: str) -> None:
    print(label, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("eager", "forward", "loss", "grad", "step"), default="eager"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--phase-d-control", action="store_true")
    args = parser.parse_args()
    mark("A process start")
    mark(f"environment JAX_TRACEBACK_FILTERING={os.environ.get('JAX_TRACEBACK_FILTERING')}")
    import jax

    mark("B imports complete")
    mark(f"C JAX devices={jax.devices()} backend={jax.default_backend()}")
    sys.path.insert(0, "scripts")
    from run_phase_b import dataset
    from run_phase_e import configs, initialise
    from adze_t.backends.torx import stable_occurrence_id
    from adze_t.objectives import adamw_init, loss_components, total_loss
    from adze_t.training import make_fixed_structure_batch, stochastic_train_step
    from adze_t.model import apply_model
    from adze_t.backends.torx import TorxOperatorConfig, TorxOps

    config = configs()["E_Q1"] if not args.phase_d_control else configs()["E_REF"]
    params = initialise(config, 0)
    mark("D model initialized")
    prompt, target = dataset("copy", args.batch_size, 820)
    batch = make_fixed_structure_batch(prompt, target, config=config)
    mark("E batch loaded")
    zeros = adamw_init(params)
    moments = (zeros, zeros)
    mark("F optimizer initialized")
    root = jax.random.fold_in(jax.random.PRNGKey(0), stable_occurrence_id("phase_e_0_1"))

    def objective(p):
        ops = TorxOps.create(
            root,
            config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=1.0),
            optimizer_step=1,
        )
        clean = TorxOps.create(
            root,
            config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
            optimizer_step=1,
        )
        output = apply_model(
            p,
            batch["prompt"],
            batch["prompt_mask"],
            batch["target"],
            batch["target_mask"],
            config=config,
            ops=ops,
            target_ops=clean,
        )
        return total_loss(loss_components(output), config)

    mark("G before loss/grad tracing")
    if args.mode == "eager":
        with jax.disable_jit():
            value = objective(params)
            mark(f"I after forward value={float(value)}")
            grads = jax.grad(objective)(params)
            mark(f"J after backward/grad leaves={len(jax.tree.leaves(grads))}")
    elif args.mode == "forward":
        compiled = jax.jit(lambda p: objective(p))
        value = compiled(params)
        jax.block_until_ready(value)
        mark("H after tracing/compilation")
        mark(f"I after forward value={float(value)}")
    elif args.mode == "loss":
        compiled = jax.jit(objective)
        value = compiled(params)
        jax.block_until_ready(value)
        mark("H after tracing/compilation")
        mark(f"I after forward value={float(value)}")
    elif args.mode == "grad":
        compiled = jax.jit(jax.value_and_grad(objective))
        value, grads = compiled(params)
        jax.block_until_ready(value)
        mark("H after tracing/compilation")
        mark(f"J after backward/grad value={float(value)} leaves={len(jax.tree.leaves(grads))}")
    else:
        compiled = jax.jit(stochastic_train_step, static_argnames=("config",))
        params, moments, metrics = compiled(params, moments, 1, batch, root, config=config)
        jax.block_until_ready(metrics["loss"])
        mark("H after tracing/compilation")
        mark(f"K after optimizer update loss={float(metrics['loss'])}")
    mark("L process exit")


if __name__ == "__main__":
    main()
