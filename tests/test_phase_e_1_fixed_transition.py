import jax
import jax.numpy as jnp

from adze_t.phase_e_1_fixed_transition import (
    FIXED_TRANSITION_V0,
    audit_fixed_transition_dataset,
    balanced_transition_indices,
    bytes_to_little_endian_bits,
    fixed_transition_example_hashes,
    fixed_transition_oracle,
    generate_fixed_transition_dataset,
    iterate_rule30,
    little_endian_bits_to_bytes,
    rule30_step,
    transition_quality_audit,
)


def _reference_step(state_bytes: list[int]) -> list[int]:
    state = int.from_bytes(bytes(state_bytes), "little")
    output = 0
    for index in range(64):
        left = (state >> ((index - 1) % 64)) & 1
        center = (state >> index) & 1
        right = (state >> ((index + 1) % 64)) & 1
        output |= (left ^ (center | right)) << index
    return list(output.to_bytes(8, "little"))


def _reference_iterate(state_bytes: list[int], depth: int) -> list[int]:
    current = state_bytes
    for _ in range(depth):
        current = _reference_step(current)
    return current


def test_rule30_hand_cases_and_periodic_boundary():
    zero = jnp.zeros((1, 8), dtype=jnp.int32)
    ones = jnp.full((1, 8), 255, dtype=jnp.int32)
    single = jnp.asarray([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)
    assert rule30_step(zero).tolist() == [[0] * 8]
    assert rule30_step(ones).tolist() == [[0] * 8]
    assert rule30_step(single).tolist() == [[3, 0, 0, 0, 0, 0, 0, 128]]
    assert rule30_step(single)[0].tolist() == _reference_step(single[0].tolist())


def test_bit_packing_round_trip_and_one_step_reference_parity():
    states = jnp.asarray(
        [[0, 1, 2, 3, 127, 128, 254, 255], [91, 42, 7, 200, 1, 8, 64, 129]],
        dtype=jnp.int32,
    )
    bits = bytes_to_little_endian_bits(states)
    assert bits.shape == (2, 64)
    assert jnp.array_equal(little_endian_bits_to_bytes(bits), states)
    expected = jnp.asarray([_reference_step(row.tolist()) for row in states], dtype=jnp.int32)
    assert jnp.array_equal(rule30_step(states), expected)


def test_repeated_iteration_and_prompt_oracle_match_independent_reference():
    prompt, target, depths, audit = generate_fixed_transition_dataset(64, 930)
    expected = jnp.asarray(
        [
            _reference_iterate(state.tolist(), int(depth))
            for state, depth in zip(prompt[:, 2:], depths, strict=True)
        ],
        dtype=jnp.int32,
    )
    assert audit["version"] == "FIXED_STATE_TRANSITION_V0"
    assert jnp.array_equal(iterate_rule30(prompt[:, 2:], depths), expected)
    assert jnp.array_equal(fixed_transition_oracle(prompt), expected)
    assert jnp.array_equal(target, expected)


def test_fixed_transition_dataset_is_balanced_reproducible_and_task_relevant():
    first = generate_fixed_transition_dataset(1_024, 931)
    repeated = generate_fixed_transition_dataset(1_024, 931)
    other = generate_fixed_transition_dataset(1_024, 932)
    for left, right in zip(first[:3], repeated[:3], strict=True):
        assert jnp.array_equal(left, right)
    assert not jnp.array_equal(first[0], other[0])
    prompt, target, depths, _ = first
    checks = audit_fixed_transition_dataset(prompt, target, depths)
    assert all(checks.values())
    assert prompt.shape == (1_024, 10)
    assert target.shape == (1_024, 8)
    assert jnp.all(
        jnp.asarray([jnp.sum(depths == depth) for depth in FIXED_TRANSITION_V0.depths]) == 128
    )
    assert jnp.array_equal(target, iterate_rule30(prompt[:, 2:], depths))
    selected = balanced_transition_indices(depths, 64)
    assert selected.shape == (512,)
    assert jnp.all(
        jnp.asarray([jnp.sum(depths[selected] == depth) for depth in FIXED_TRANSITION_V0.depths])
        == 64
    )


def test_fixed_transition_splits_have_no_complete_example_overlap():
    train = generate_fixed_transition_dataset(2_048, 930)
    validation = generate_fixed_transition_dataset(1_024, 931)
    test = generate_fixed_transition_dataset(1_024, 932)
    train_hashes = fixed_transition_example_hashes(*train[:3])
    validation_hashes = fixed_transition_example_hashes(*validation[:3])
    test_hashes = fixed_transition_example_hashes(*test[:3])
    assert len(train_hashes) == 2_048
    assert len(validation_hashes) == 1_024
    assert len(test_hashes) == 1_024
    assert not train_hashes & validation_hashes
    assert not train_hashes & test_hashes
    assert not validation_hashes & test_hashes


def test_predeclared_rule_passes_model_independent_quality_gate():
    initial = jax.random.randint(jax.random.PRNGKey(933), (4_096, 8), 0, 256)
    audit = transition_quality_audit(initial)
    assert all(audit["quality_gate"].values())
