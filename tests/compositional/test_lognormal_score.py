"""Compare a lognormal likelihood, its derivatives, and a score-driven model with JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.stats import norm

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_lognormal(mean: ir.RV, log_scale: ir.RV) -> ir.RV:
    def normal_with_log_scale(mean: ir.RV, log_scale: ir.RV) -> ir.RV:
        return pi.normal(mean, pi.exp(log_scale))

    return pi.tforms.exp(normal_with_log_scale)(mean, log_scale)


def pangolin_composite(mean: ir.RV, log_scale: ir.RV) -> ir.RV:
    return pi.composite(pangolin_lognormal, simplify=False)(mean, log_scale)


def jax_density(value: jax.Array, mean: jax.Array, log_scale: jax.Array) -> jax.Array:
    return norm.logpdf(jnp.log(value), mean, jnp.exp(log_scale)) - jnp.log(value)


@pytest.mark.parametrize("lognormal", [pangolin_lognormal, pangolin_composite])
def test_lognormal_score_and_curvature(lognormal: Callable[[ir.RV, ir.RV], ir.RV]):
    density = pi.log_prob(lognormal)
    score = pi.grad(density)
    curvature = pi.grad(score)

    def score_model(value: ir.RV, mean: ir.RV, log_scale: ir.RV) -> ir.RV:
        return pi.normal(score(value, mean, log_scale), 0.7)

    value, mean, log_scale = (pi.normal(0, 1) for _ in range(3))
    results = (
        density(value, mean, log_scale),
        score(value, mean, log_scale),
        curvature(value, mean, log_scale),
        pi.grad(density, 2)(value, mean, log_scale),
        pi.log_prob(score_model)(pi.constant(-0.4), value, mean, log_scale),
    )

    def evaluate(y: jax.Array, mu: jax.Array, log_sigma: jax.Array):
        return jax_backend.ancestor_sample_graph(
            results, key=None, boundary={value: y, mean: mu, log_scale: log_sigma}
        )

    compiled = jax.jit(evaluate)
    for y, mu, log_sigma in [(0.7, -0.2, 0.3), (2.4, 0.5, -0.4)]:
        args = tuple(jnp.asarray(x) for x in (y, mu, log_sigma))
        expected_score = jax.grad(jax_density)(*args)
        expected = (
            jax_density(*args),
            expected_score,
            jax.grad(jax.grad(jax_density))(*args),
            jax.grad(jax_density, 2)(*args),
            norm.logpdf(-0.4, expected_score, 0.7),
        )
        assert_matches(evaluate(*args), expected)
        assert_matches(compiled(*args), expected)
