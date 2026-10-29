"""Compare a Gram matrix of per-observation scores and derivatives through it with JAX."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_gram(theta: ir.RV, values: ir.RV) -> ir.RV:
    def density(parameter: ir.RV, value: ir.RV) -> ir.RV:
        return pi.log_prob(pi.normal)(value, parameter[0], pi.exp(parameter[1]))

    scores = pi.vmap(pi.grad(density), (None, 0))(theta, values)
    return scores.T @ scores / 4


def jax_gram(theta: jax.Array, values: jax.Array) -> jax.Array:
    def density(parameter: jax.Array, value: jax.Array) -> jax.Array:
        return norm.logpdf(value, parameter[0], jnp.exp(parameter[1]))

    scores = jax.vmap(jax.grad(density), (None, 0))(theta, values)
    return scores.T @ scores / 4


def test_score_gram():
    observations = jnp.array([-0.8, 0.1, 0.7, 1.3])
    weights = jnp.array([[0.4, -0.7], [0.2, 0.9]])
    theta, values = pi.constant([0.2, -0.1]), pi.constant(observations)
    results = (
        pangolin_gram(theta, values),
        pi.vjp(pangolin_gram)(pi.constant(weights), theta, values),
        pi.vjp(pangolin_gram, 1)(pi.constant(weights), theta, values),
    )

    def evaluate(parameter: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {theta: parameter})

    compiled = jax.jit(evaluate)
    for parameter in (jnp.array([0.2, -0.1]), jnp.array([-0.4, 0.3])):
        value, pullback = jax.vjp(jax_gram, parameter, observations)
        expected = (value, *pullback(weights))
        assert_matches(evaluate(parameter), expected, rtol=4e-5)
        assert_matches(compiled(parameter), expected, rtol=4e-5)
