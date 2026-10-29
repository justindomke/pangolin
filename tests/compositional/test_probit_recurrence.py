"""Compare batched CDF recurrences and shared-parameter derivatives using Scan, loops, and JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import ndtr

from pangolin import interface, ir, jax_backend

from .helpers import assert_matches

pi = interface.Interface(broadcasting="off")


def pangolin_scan(coefficient: ir.RV, start: ir.RV, inputs: ir.RV) -> ir.RV:
    weights = pi.constant([0.4, -0.7, 1.1, 0.2])

    def step(previous: ir.RV, increment: ir.RV) -> ir.RV:
        return pi.cdf(pi.normal)(coefficient * previous + increment, 0, 1)

    states = pi.scan(step, 4)(start, inputs)
    return pi.sum(pi.vmap(lambda x, w: x * w)(states, weights), 0)


def pangolin_sfor(coefficient: ir.RV, start: ir.RV, inputs: ir.RV) -> ir.RV:
    weights = pi.constant([0.4, -0.7, 1.1, 0.2])

    def step(previous: ir.RV, t: ir.RV) -> ir.RV:
        return pi.cdf(pi.normal)(coefficient * previous + inputs[t], 0, 1)

    states = pi.sfor(step, init=start, t=4)
    return pi.sum(pi.vmap(lambda x, w: x * w)(states, weights), 0)


def pangolin_python(coefficient: ir.RV, start: ir.RV, inputs: ir.RV) -> ir.RV:
    weights = pi.constant([0.4, -0.7, 1.1, 0.2])
    states = []
    previous = start
    for t in range(4):
        previous = pi.cdf(pi.normal)(coefficient * previous + inputs[t], 0, 1)
        states.append(previous)
    return sum((state * weights[t] for t, state in enumerate(states)), pi.constant(0))


def jax_recurrence(coefficient: jax.Array, start: jax.Array, inputs: jax.Array) -> jax.Array:
    states = []
    previous = start
    for increment in inputs:
        previous = ndtr(coefficient * previous + increment)
        states.append(previous)
    return jnp.dot(jnp.stack(states), jnp.array([0.4, -0.7, 1.1, 0.2]))


@pytest.mark.parametrize("objective", [pangolin_scan, pangolin_sfor, pangolin_python])
def test_batched_probit_recurrence(objective: Callable[[ir.RV, ir.RV, ir.RV], ir.RV]):
    batched = pi.vmap(objective, (None, 0, 0))
    coefficient = pi.normal(0, 1)
    starts = pi.constant([-0.4, 0.6])
    inputs = pi.constant([[0.2, -0.7, 1.1, -0.3], [-0.5, 0.8, -0.2, 0.4]])
    cotangent = pi.constant([1.3, -0.6])
    results = (
        batched(coefficient, starts, inputs),
        pi.vmap(pi.grad(objective), (None, 0, 0))(coefficient, starts, inputs),
        pi.vjp(batched)(cotangent, coefficient, starts, inputs),
        pi.vjp(batched, 2)(cotangent, coefficient, starts, inputs),
    )

    def evaluate(a: jax.Array, initial: jax.Array, increments: jax.Array):
        return jax_backend.ancestor_sample_graph(
            results, key=None, boundary={coefficient: a, starts: initial, inputs: increments}
        )

    reference_batch = jax.vmap(jax_recurrence, (None, 0, 0))
    compiled = jax.jit(evaluate)
    for a in (0.8, -0.5):
        args = (
            jnp.asarray(a),
            jnp.array([-0.4, 0.6]),
            jnp.array([[0.2, -0.7, 1.1, -0.3], [-0.5, 0.8, -0.2, 0.4]]),
        )
        _, pullback = jax.vjp(reference_batch, *args)
        derivatives = pullback(jnp.array([1.3, -0.6]))
        expected = (
            reference_batch(*args),
            jax.vmap(jax.grad(jax_recurrence), (None, 0, 0))(*args),
            derivatives[0],
            derivatives[2],
        )
        assert_matches(evaluate(*args), expected, rtol=3e-5)
        assert_matches(compiled(*args), expected, rtol=3e-5)
