import jax
import jax.numpy as jnp

from experiments.m4_5_algorithmic_reasoning.arithmetic import (
    DIGITS,
    RESULT_DIGITS,
    exact_add,
    make_data,
)
from experiments.m4_5_algorithmic_reasoning.model import (
    AlgorithmicConfig,
    apply_state,
    initialise_params,
)
from experiments.m4_5_algorithmic_reasoning.programs import (
    MODULUS,
    OPS,
    apply_instruction,
    execute_program,
    mask_all_conditioning,
)
from experiments.m4_5_algorithmic_reasoning.programs import (
    make_data as make_program,
)


def test_arithmetic_is_complete_and_deterministic_with_requested_carry_depths():
    for depth in (1, 2, 4, 8, 12):
        first = make_data(jax.random.PRNGKey(depth), 32, depth)
        second = make_data(jax.random.PRNGKey(depth), 32, depth)
        assert jnp.array_equal(first.result, second.result)
        assert first.a.shape == (32, DIGITS)
        assert first.result.shape == (32, RESULT_DIGITS)
        assert jnp.all(first.carry_depth == depth)
        oracle, _ = exact_add(first.a, first.b)
        assert jnp.array_equal(oracle, first.result)


def test_arithmetic_cursor_has_exact_completion_and_done_tail():
    batch = make_data(jax.random.PRNGKey(11), 2, 12)
    assert batch.cursor_conditioning.shape == (2, DIGITS + 1, 5)
    assert jnp.all(batch.cursor_conditioning[:, :DIGITS, 3] == 1)
    assert jnp.all(batch.cursor_conditioning[:, DIGITS, 4] == 1)
    assert batch.result.shape[-1] == 13


def test_register_instructions_are_bijective():
    states = jnp.asarray([(a, b) for a in range(MODULUS) for b in range(MODULUS)])
    for instruction in range(OPS):
        outputs = jax.vmap(apply_instruction, in_axes=(0, None))(states, jnp.asarray(instruction))
        assert jnp.array_equal(jnp.unique(outputs, axis=0), states)


def test_program_oracle_and_padding_are_fixed_shape():
    batch = make_program(jax.random.PRNGKey(12), 16, 4)
    assert batch.programs.shape == (16, 12)
    assert batch.all_conditioning.shape[-1] == 12 * OPS + 12
    assert jnp.all(batch.valid[:, 4:] == 0)
    intermediates, final = execute_program(batch.initial_registers[0], batch.programs[0, :4])
    assert intermediates.shape == (5, 2)
    assert jnp.array_equal(final, batch.final_registers[0])


def test_program_padding_is_masked_but_valid_instruction_is_not():
    batch = make_program(jax.random.PRNGKey(121), 1, 4)
    original = batch.all_conditioning[0]
    padded = original.at[4 * OPS].set(1.0).at[4 * OPS + 1].set(1.0)
    assert jnp.array_equal(mask_all_conditioning(original), mask_all_conditioning(padded))
    valid_changed = original.at[0].set(1.0 - original[0])
    assert not jnp.array_equal(
        mask_all_conditioning(original), mask_all_conditioning(valid_changed)
    )


def test_recurrent_state_is_fixed_shape_and_conditioning_is_read_only():
    batch = make_program(jax.random.PRNGKey(13), 1, 4)
    config = AlgorithmicConfig(
        q=12, conditioning_width=batch.cursor_conditioning.shape[-1], output_shape=(2, 16)
    )
    params = initialise_params(config, jax.random.PRNGKey(14))
    schedule = batch.cursor_conditioning[0]
    output, trajectory = (
        apply_state(
            config, params, batch.initial_dynamic[0], schedule, jax.random.PRNGKey(15), True
        ),
        None,
    )
    assert output.shape == (13, 32)
    assert trajectory is None


def test_q12_gradients_are_finite():
    batch = make_data(jax.random.PRNGKey(16), 2, 8)
    config = AlgorithmicConfig(q=12, conditioning_width=48, output_shape=(13, 10))
    params = initialise_params(config, jax.random.PRNGKey(17))
    schedules = jnp.broadcast_to(batch.all_conditioning[:, None, :], (2, 12, 48))
    keys = jax.random.split(jax.random.PRNGKey(18), 2)

    def loss(candidate):
        states = jax.vmap(lambda d, c, k: apply_state(config, candidate, d, c, k))(
            batch.initial_dynamic, schedules, keys
        )
        return jnp.mean(states**2)

    value, gradients = jax.value_and_grad(loss)(params)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(gradients))
