"""Compare a batched heteroscedastic likelihood, parameter gradient, and Hessian product with JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.stats import norm

from pangolin import functional, interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_vmap(theta: ir.RV, design: ir.RV, values: ir.RV) -> ir.RV:
    def observation(row: ir.RV) -> ir.RV:
        return pi.normal(row @ theta[0, :], pi.exp(row @ theta[1, :]))

    return pi.log_prob(pi.vmap(observation))(values, design)


def pangolin_vfor(theta: ir.RV, design: ir.RV, values: ir.RV) -> ir.RV:
    def observation(i: ir.RV) -> ir.RV:
        return pi.normal(design[i, :] @ theta[0, :], pi.exp(design[i, :] @ theta[1, :]))

    def observations() -> ir.RV:
        return pi.vfor(observation, i=3)

    return pi.log_prob(observations)(values)


def pangolin_loop(theta: ir.RV, design: ir.RV, values: ir.RV) -> ir.RV:
    def observation(i: ir.RV) -> ir.RV:
        return pi.normal(design[i, :] @ theta[0, :], pi.exp(design[i, :] @ theta[1, :]))

    def observations() -> ir.RV:
        return functional.loop(observation, interface=pi.default, i=3)

    return pi.log_prob(observations)(values)


def jax_likelihood(theta: jax.Array, design: jax.Array, values: jax.Array) -> jax.Array:
    return jnp.sum(norm.logpdf(values, design @ theta[0], jnp.exp(design @ theta[1])))


@pytest.mark.parametrize("likelihood", [pangolin_vmap, pangolin_vfor, pangolin_loop])
def test_heteroscedastic_regression(likelihood: Callable[[ir.RV, ir.RV, ir.RV], ir.RV]):
    theta = pi.constant([[0.2, -0.3], [-0.4, 0.1]])
    design = pi.constant([[1.0, -0.7], [1.0, 0.2], [1.0, 1.3]])
    values = pi.constant([-0.2, 0.5, 1.1])
    direction = pi.constant([[0.3, -0.8], [0.2, 0.6]])
    score = pi.grad(likelihood)
    results = (
        likelihood(theta, design, values),
        score(theta, design, values),
        pi.vjp(score)(direction, theta, design, values),
    )

    def evaluate(t: jax.Array, x: jax.Array, y: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {theta: t, design: x, values: y})

    compiled = jax.jit(evaluate)
    x = jnp.array([[1.0, -0.7], [1.0, 0.2], [1.0, 1.3]])
    y = jnp.array([-0.2, 0.5, 1.1])
    direction_value = jnp.array([[0.3, -0.8], [0.2, 0.6]])
    for t in (jnp.array([[0.2, -0.3], [-0.4, 0.1]]), jnp.array([[-0.5, 0.8], [0.2, -0.2]])):
        _, pullback = jax.vjp(lambda t: jax.grad(jax_likelihood)(t, x, y), t)
        expected = (
            jax_likelihood(t, x, y),
            jax.grad(jax_likelihood)(t, x, y),
            pullback(direction_value)[0],
        )
        assert_matches(evaluate(t, x, y), expected)
        assert_matches(compiled(t, x, y), expected)
