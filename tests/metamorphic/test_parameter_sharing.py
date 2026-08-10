import jax
import jax.numpy as jnp

from experiments.m1_trainability.discrete import exact
from experiments.m1_trainability.oracles import markov_objective


def test_tied_gradient_equals_sum_of_untied_occurrence_gradients():
    theta = 0.35
    depth = 4
    tied = jax.grad(lambda t: markov_objective(t, depth))(jnp.array(theta))

    def objective(ts):
        state = jnp.array([1.0, 0.0])
        for t in ts:
            state = state @ jnp.array(
                [
                    [1 - jax.nn.sigmoid(t), jax.nn.sigmoid(t)],
                    [jax.nn.sigmoid(t), 1 - jax.nn.sigmoid(t)],
                ]
            )
        return state[1]

    untied = jax.grad(objective)(jnp.full((depth,), theta))
    assert jnp.allclose(tied, jnp.sum(untied))
    assert jnp.allclose(tied, exact(depth, theta)[1])
