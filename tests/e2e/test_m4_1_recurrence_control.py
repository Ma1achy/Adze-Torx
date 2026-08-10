import jax

from experiments.m3_carrier_structure.data import make_data
from experiments.m4_1_recurrence_control.model import M4Config, initialise_params
from experiments.m4_1_recurrence_control.train import TrainConfig, run


def test_m4_1_train_curve_and_optimizer_diagnostics_are_recorded():
    config = M4Config(core=M4Config().core.__class__(width=18, q=2, family="q_normalized_residual"))
    train_data = make_data(jax.random.key(60), 64, config.structure)
    val_data = make_data(jax.random.key(61), 32, config.structure)
    _, metrics = run(
        config,
        TrainConfig(steps=4, batch_size=16, eval_interval=2, seed=4),
        train_data,
        val_data,
        initialise_params,
    )
    assert len(metrics["train_curve"]) == 4
    assert len(metrics["validation_curve"]) == 2
    assert metrics["train_curve"][0]["delta_update_parameter_ratio"] is None
    assert metrics["compile_seconds_estimate"] >= 0.0
