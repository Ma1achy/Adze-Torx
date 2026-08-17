from pathlib import Path

from adze_t.phase_e_1_paths import checkpoint_path, evidence_path, resolve_run_state


def test_primary_checkpoint_never_resumes_calibration_state(tmp_path: Path):
    calibration = checkpoint_path(
        tmp_path,
        benchmark="pointer",
        stage="calibration",
        arm="E_REF",
        init_seed=0,
        stochastic_training_seed=0,
    )
    primary = checkpoint_path(
        tmp_path,
        benchmark="pointer",
        stage="primary",
        arm="E_REF",
        init_seed=0,
        stochastic_training_seed=0,
    )
    calibration.parent.mkdir(parents=True)
    calibration.write_bytes(b"synthetic calibration step N")

    assert calibration.exists()
    assert calibration != primary
    assert not primary.exists()
    params, moments, step = resolve_run_state(
        primary,
        load_state=lambda _: (_ for _ in ()).throw(AssertionError("must not load calibration")),
        initialize_state=lambda: ("frozen scratch params", "fresh moments", 0),
    )
    assert (params, moments, step) == ("frozen scratch params", "fresh moments", 0)


def test_seed_specific_checkpoint_and_evidence_paths_do_not_collide(tmp_path: Path):
    checkpoint_zero = checkpoint_path(
        tmp_path,
        benchmark="fixed_transition",
        stage="primary",
        arm="T_REF",
        init_seed=0,
        stochastic_training_seed=0,
    )
    checkpoint_one = checkpoint_path(
        tmp_path,
        benchmark="fixed_transition",
        stage="primary",
        arm="T_REF",
        init_seed=1,
        stochastic_training_seed=1,
    )
    evidence_zero = evidence_path(
        tmp_path,
        benchmark="fixed_transition",
        stage="primary",
        stem="training_ref",
        init_seed=0,
        stochastic_training_seed=0,
        suffix=".jsonl",
    )
    evidence_one = evidence_path(
        tmp_path,
        benchmark="fixed_transition",
        stage="primary",
        stem="training_ref",
        init_seed=1,
        stochastic_training_seed=1,
        suffix=".jsonl",
    )

    assert checkpoint_zero != checkpoint_one
    assert evidence_zero != evidence_one
    assert checkpoint_zero.name == "init0_stoch0.pkl"
    assert evidence_one.name == "training_ref_init1_stoch1.jsonl"
