"""Compare multivariate Gaussian likelihoods through a learned positive-definite covariance with JAX."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_likelihood(factor: ir.RV, values: ir.RV) -> ir.RV:
    covariance = factor @ factor.T + pi.diag_matrix(pi.exp(pi.diag(factor)))
    mean = pi.sum(factor, 0)

    def observation() -> ir.RV:
        return pi.multi_normal(mean, covariance)

    return pi.log_prob(pi.vmap(observation, axis_size=3))(values)


def jax_likelihood(factor: jax.Array, values: jax.Array) -> jax.Array:
    covariance = factor @ factor.T + jnp.diag(jnp.exp(jnp.diag(factor)))
    mean = jnp.sum(factor, axis=0)
    return jnp.sum(multivariate_normal.logpdf(values, mean, covariance))


def test_covariance_regression():
    matrix = jnp.array([[0.3, -0.4], [0.6, 0.2]])
    observations = jnp.array([[-0.3, 0.7], [1.1, -0.2], [0.4, 0.5]])
    direction = jnp.array([[0.2, -0.8], [0.3, 0.5]])
    factor, values = pi.constant(matrix), pi.constant(observations)
    score = pi.grad(pangolin_likelihood)
    results = (
        pangolin_likelihood(factor, values),
        score(factor, values),
        pi.grad(pangolin_likelihood, 1)(factor, values),
        pi.vjp(score)(pi.constant(direction), factor, values),
    )

    def evaluate(m: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {factor: m})

    compiled = jax.jit(evaluate)
    for m in (matrix, matrix + 0.4):
        _, pullback = jax.vjp(lambda a: jax.grad(jax_likelihood)(a, observations), m)
        expected = (
            jax_likelihood(m, observations),
            jax.grad(jax_likelihood)(m, observations),
            jax.grad(jax_likelihood, 1)(m, observations),
            pullback(direction)[0],
        )
        assert_matches(evaluate(m), expected, rtol=4e-5, atol=3e-6)
        assert_matches(compiled(m), expected, rtol=4e-5, atol=3e-6)
