"""Compare a lognormal autoregression's joint density and derivatives with explicit JAX factors."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.stats import norm

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_scan(values: ir.RV, coefficient: ir.RV, log_scale: ir.RV) -> ir.RV:
    increments = pi.constant([0.2, -0.4, 0.1, 0.3])
    start = pi.constant(1.4)

    def step(previous: ir.RV, increment: ir.RV) -> ir.RV:
        return pi.tforms.exp(pi.normal)(
            coefficient * pi.log(previous) + increment, pi.exp(log_scale)
        )

    return pi.log_prob(pi.scan(step, 4))(values, start, increments)


def pangolin_sfor(values: ir.RV, coefficient: ir.RV, log_scale: ir.RV) -> ir.RV:
    increments = pi.constant([0.2, -0.4, 0.1, 0.3])
    start = pi.constant(1.4)

    def step(previous: ir.RV, t: ir.RV) -> ir.RV:
        return pi.tforms.exp(pi.normal)(
            coefficient * pi.log(previous) + increments[t], pi.exp(log_scale)
        )

    def recurrence(initial: ir.RV) -> ir.RV:
        return pi.sfor(step, init=initial, t=4)

    return pi.log_prob(recurrence)(values, start)


def pangolin_python(values: ir.RV, coefficient: ir.RV, log_scale: ir.RV) -> ir.RV:
    increments = pi.constant([0.2, -0.4, 0.1, 0.3])
    start = pi.constant(1.4)

    def step(previous: ir.RV, increment: ir.RV) -> ir.RV:
        return pi.tforms.exp(pi.normal)(
            coefficient * pi.log(previous) + increment, pi.exp(log_scale)
        )

    factors = []
    previous = start
    for t in range(4):
        factors.append(pi.log_prob(step)(values[t], previous, increments[t]))
        previous = values[t]
    return sum(factors, pi.constant(0))


def jax_density(values: jax.Array, coefficient: jax.Array, log_scale: jax.Array) -> jax.Array:
    previous = jnp.concatenate((jnp.array([1.4]), values[:-1]))
    means = coefficient * jnp.log(previous) + jnp.array([0.2, -0.4, 0.1, 0.3])
    return jnp.sum(norm.logpdf(jnp.log(values), means, jnp.exp(log_scale)) - jnp.log(values))


@pytest.mark.parametrize("density", [pangolin_scan, pangolin_sfor, pangolin_python])
def test_lognormal_recurrence_likelihood(density: Callable[[ir.RV, ir.RV, ir.RV], ir.RV]):
    values = pi.constant([1.1, 0.7, 1.5, 2.0])
    coefficient = pi.normal(0, 1)
    log_scale = pi.normal(0, 1)
    results = (
        density(values, coefficient, log_scale),
        pi.grad(density)(values, coefficient, log_scale),
        pi.grad(density, 1)(values, coefficient, log_scale),
        pi.grad(pi.grad(density, 1), 1)(values, coefficient, log_scale),
    )

    def evaluate(y: jax.Array, a: jax.Array, log_sigma: jax.Array):
        return jax_backend.ancestor_sample_graph(
            results, key=None, boundary={values: y, coefficient: a, log_scale: log_sigma}
        )

    compiled = jax.jit(evaluate)
    for y, a, log_sigma in [([1.1, 0.7, 1.5, 2.0], 0.6, -0.3), ([0.8, 1.7, 0.9, 1.2], -0.4, 0.2)]:
        args = (jnp.array(y), jnp.asarray(a), jnp.asarray(log_sigma))
        expected = (
            jax_density(*args),
            jax.grad(jax_density)(*args),
            jax.grad(jax_density, 1)(*args),
            jax.grad(jax.grad(jax_density, 1), 1)(*args),
        )
        assert_matches(evaluate(*args), expected, rtol=3e-5)
        assert_matches(compiled(*args), expected, rtol=3e-5)
