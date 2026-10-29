"""A nonlinear two-axis recurrence, its pullbacks, and its curvature versus nested JAX scans."""

import jax
import jax.numpy as jnp

from pangolin import interface as pi, ir, jax_backend
from pangolin.functional import loop

from .helpers import assert_matches


def pangolin_field(theta: ir.RV, forcing: ir.RV) -> ir.RV:
    def cell(prev, t, s):
        if t is None:
            return theta[0] + theta[1] * s
        if s is None:
            return theta[1] - 0.2 * theta[0] * t
        a, b = prev[t - 1, s], prev[t, s - 1]
        return pi.tanh(theta[0] * a + theta[1] * b + forcing[t, s]) + 0.1 * a

    return loop(
        cell,
        interface=pi.default,
        t=[2],
        s=[3],
    )


def jax_field(theta, forcing):
    def row(old_t, inputs):
        t, increments = inputs

        def cell(old_s, inputs):
            a, increment = inputs
            current = jnp.tanh(theta[0] * a + theta[1] * old_s + increment) + 0.1 * a
            return current, current

        current = jax.lax.scan(cell, theta[1] - 0.2 * theta[0] * t, (old_t, increments))[1]
        return current, current

    initial = theta[0] + theta[1] * jnp.arange(3)
    return jax.lax.scan(row, initial, (jnp.arange(2), forcing))[1]


def pangolin_loss(theta: ir.RV, forcing: ir.RV) -> ir.RV:
    x = pangolin_field(theta, forcing)
    return pi.sum(pi.sum(x * x, 0), 0)


def jax_loss(theta, forcing):
    return jnp.sum(jax_field(theta, forcing) ** 2)


def test_values_pullbacks_gradients_and_hessian_vector_products():
    theta_values = jnp.array([0.2, -0.3])
    forcing_values = jnp.array([[0.1, -0.2, 0.3], [0.2, 0.15, -0.4]])
    weights = jnp.array([[0.4, -0.2, 0.1], [0.3, 0.5, -0.3]])
    direction = jnp.array([0.6, -0.4])
    theta, forcing = pi.constant(theta_values), pi.constant(forcing_values)
    results = (
        pangolin_field(theta, forcing),
        pi.composite(pangolin_field, simplify=False)(theta, forcing),
        pi.vjp(pangolin_field)(pi.constant(weights), theta, forcing),
        pi.vjp(pangolin_field, 1)(pi.constant(weights), theta, forcing),
        pi.grad(pangolin_loss)(theta, forcing),
        pi.grad(pangolin_loss, 1)(theta, forcing),
        pi.vjp(pi.grad(pangolin_loss))(pi.constant(direction), theta, forcing),
    )

    def evaluate(a, b):
        return jax_backend.ancestor_sample_graph(results, None, {theta: a, forcing: b})

    compiled = jax.jit(evaluate)
    for a, b in ((theta_values, forcing_values), (-theta_values, forcing_values + 0.25)):
        value, pullback = jax.vjp(jax_field, a, b)
        _, curvature = jax.vjp(lambda t: jax.grad(jax_loss)(t, b), a)
        expected = (
            value,
            value,
            *pullback(weights),
            jax.grad(jax_loss)(a, b),
            jax.grad(jax_loss, 1)(a, b),
            curvature(direction)[0],
        )
        assert_matches(evaluate(a, b), expected, rtol=4e-5, atol=4e-6)
        assert_matches(compiled(a, b), expected, rtol=4e-5, atol=4e-6)
