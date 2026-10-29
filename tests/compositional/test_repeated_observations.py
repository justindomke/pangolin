"""Compare likelihood derivatives through repeated vectorized indices and a nonleading mapped axis with JAX."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_likelihood(matrix: ir.RV, values: ir.RV) -> ir.RV:
    rows = pi.constant([0, 1, 0, 0])
    cols = pi.constant([2, 0, 2, 1])
    means = matrix.vindex[rows, cols]

    def replicate(observations: ir.RV) -> ir.RV:
        return pi.sum(pi.vmap(pi.log_prob(pi.normal), (0, 0, None))(observations, means, 0.7), 0)

    return pi.sum(pi.vmap(replicate, 1)(values), 0)


def jax_likelihood(matrix: jax.Array, values: jax.Array) -> jax.Array:
    means = matrix[jnp.array([0, 1, 0, 0]), jnp.array([2, 0, 2, 1])]
    return jnp.sum(norm.logpdf(values, means[:, None], 0.7))


def test_repeated_observations():
    m = jnp.array([[0.2, -0.4, 0.7], [-0.3, 0.6, 0.1]])
    y = jnp.array([[0.5, 1.0], [-0.4, 0.2], [0.8, 0.3], [-0.2, -0.7]])
    direction = jnp.array([[0.3, -0.8, 0.2], [0.5, 0.6, -0.4]])
    matrix, values = pi.constant(m), pi.constant(y)
    score = pi.grad(pangolin_likelihood)
    results = (
        pangolin_likelihood(matrix, values),
        score(matrix, values),
        pi.vjp(score)(pi.constant(direction), matrix, values),
    )

    def evaluate(parameter: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {matrix: parameter})

    compiled = jax.jit(evaluate)
    for parameter in (m, m - 0.3):
        _, pullback = jax.vjp(lambda a: jax.grad(jax_likelihood)(a, y), parameter)
        expected = (
            jax_likelihood(parameter, y),
            jax.grad(jax_likelihood)(parameter, y),
            pullback(direction)[0],
        )
        assert_matches(evaluate(parameter), expected)
        assert_matches(compiled(parameter), expected)
