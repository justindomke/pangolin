"""Compare CDF-driven Bernoulli regression and mixed derivatives with an explicit JAX likelihood."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import ndtr

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_vmap(beta: ir.RV, design: ir.RV, values: ir.RV) -> ir.RV:
    def outcome(row: ir.RV) -> ir.RV:
        probability = pi.cdf(pi.normal)(row @ beta, 0, 1)
        return pi.bernoulli(probability)

    return pi.log_prob(pi.vmap(outcome))(values, design)


def pangolin_python(beta: ir.RV, design: ir.RV, values: ir.RV) -> ir.RV:
    def outcome(row: ir.RV) -> ir.RV:
        return pi.bernoulli(pi.cdf(pi.normal)(row @ beta, 0, 1))

    factors = [pi.log_prob(outcome)(values[i], design[i, :]) for i in range(4)]
    return sum(factors, pi.constant(0))


def jax_likelihood(beta: jax.Array, design: jax.Array, values: jax.Array) -> jax.Array:
    probabilities = ndtr(design @ beta)
    return jnp.sum(jnp.where(values == 1, jnp.log(probabilities), jnp.log1p(-probabilities)))


@pytest.mark.parametrize("likelihood", [pangolin_vmap, pangolin_python])
def test_probit_regression(likelihood: Callable[[ir.RV, ir.RV, ir.RV], ir.RV]):
    beta = pi.constant([0.3, -0.2])
    design_value = jnp.array([[1.0, -1.2], [1.0, -0.1], [1.0, 0.8], [1.0, 1.5]])
    observations = jnp.array([0, 1, 0, 1])
    design, values = pi.constant(design_value), pi.constant(observations)
    direction_value = jnp.array([0.4, -0.7])
    score = pi.grad(likelihood)
    results = (
        likelihood(beta, design, values),
        score(beta, design, values),
        pi.vjp(score, 1)(pi.constant(direction_value), beta, design, values),
    )

    def evaluate(b: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {beta: b})

    compiled = jax.jit(evaluate)
    for b in (jnp.array([0.3, -0.2]), jnp.array([-0.6, 0.5])):
        _, pullback = jax.vjp(lambda x: jax.grad(jax_likelihood)(b, x, observations), design_value)
        expected = (
            jax_likelihood(b, design_value, observations),
            jax.grad(jax_likelihood)(b, design_value, observations),
            pullback(direction_value)[0],
        )
        assert_matches(evaluate(b), expected)
        assert_matches(compiled(b), expected)
