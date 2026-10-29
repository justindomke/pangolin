"""Compare a nonlinear vector-carry recurrence and full-trajectory pullbacks with JAX Scan."""

import jax
import jax.numpy as jnp

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_states(matrix: ir.RV, start: ir.RV, inputs: ir.RV) -> ir.RV:
    def step(previous: ir.RV, increment: ir.RV) -> ir.RV:
        return 0.2 * previous + pi.tanh(matrix @ previous + increment)

    return pi.scan(pi.composite(step, simplify=False), 3)(start, inputs)


def jax_states(matrix: jax.Array, start: jax.Array, inputs: jax.Array) -> jax.Array:
    def step(previous, increment):
        current = 0.2 * previous + jnp.tanh(matrix @ previous + increment)
        return current, current

    return jax.lax.scan(step, start, inputs)[1]


def test_coupled_recurrence():
    m = jnp.array([[0.3, -0.5], [0.6, 0.2]])
    initial = jnp.array([-0.4, 0.7])
    increments = jnp.array([[0.1, -0.2], [0.3, 0.5], [-0.6, 0.2]])
    weights = jnp.array([[0.2, -0.5], [0.7, 0.4], [-0.3, 0.9]])
    matrix, start, inputs = pi.constant(m), pi.constant(initial), pi.constant(increments)
    cotangent = pi.constant(weights)
    results = (
        pangolin_states(matrix, start, inputs),
        pi.vjp(pangolin_states)(cotangent, matrix, start, inputs),
        pi.vjp(pangolin_states, 1)(cotangent, matrix, start, inputs),
        pi.vjp(pangolin_states, 2)(cotangent, matrix, start, inputs),
    )

    def evaluate(m: jax.Array, x: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {matrix: m, inputs: x})

    compiled = jax.jit(evaluate)
    for matrix_value, input_value in ((m, increments), (-0.7 * m, increments + 0.15)):
        value, pullback = jax.vjp(jax_states, matrix_value, initial, input_value)
        expected = (value, *pullback(weights))
        assert_matches(evaluate(matrix_value, input_value), expected)
        assert_matches(compiled(matrix_value, input_value), expected)
