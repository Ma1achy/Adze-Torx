import jax.numpy as jnp

from adze_t.phase_e_1_pointer import (
    POINTER_V0,
    audit_pointer_dataset,
    generate_pointer_dataset,
    pointer_intermediate_states,
    pointer_oracle,
)


def test_pointer_v0_layout_oracle_and_permutations():
    prompt, target, depths, audit = generate_pointer_dataset(128, 920)
    checks = audit_pointer_dataset(prompt, target, depths)
    assert prompt.shape == (128, 120)
    assert target.shape == (128, 8)
    assert depths.shape == (128,)
    assert audit["prompt_bytes"] == 120
    assert all(checks.values())
    assert jnp.all(pointer_oracle(prompt) == target)
    trajectory = pointer_intermediate_states(prompt)
    assert trajectory.shape == (128, 11, 8)
    assert jnp.all(trajectory[jnp.arange(128), depths - 1] == target)


def test_pointer_generation_is_reproducible_and_seeded():
    first = generate_pointer_dataset(32, 921)
    second = generate_pointer_dataset(32, 921)
    other = generate_pointer_dataset(32, 922)
    for left, right in zip(first[:3], second[:3], strict=True):
        assert jnp.array_equal(left, right)
    assert not jnp.array_equal(first[0], other[0])


def test_pointer_spec_reports_chance_baselines():
    assert POINTER_V0.prompt_bytes == 120
    assert POINTER_V0.target_bytes == 8
    assert POINTER_V0.target_bytes == POINTER_V0.queries
    assert POINTER_V0.n_states == 10
