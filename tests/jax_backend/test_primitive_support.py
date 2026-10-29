"""Primitive lowerings own support checks, including safe inactive density evaluations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pangolin import ir
from pangolin.jax_backend import distribution_constructors, lower_deterministic, lower_random


def test_uniform_quantile_endpoints_evaluate_from_runtime_parents():
    def bounds(alpha, beta):
        parents = (alpha, beta)
        return (
            lower_deterministic(ir.Quantile(ir.Uniform()), (0.0, *parents)),
            lower_deterministic(ir.Quantile(ir.Uniform()), (1.0, *parents)),
        )

    alpha = jnp.array([-3.0, 0.0, 2.0])
    beta = jnp.array([-1.0, 4.0, 10.0])
    actual = jax.jit(jax.vmap(bounds))(alpha, beta)
    np.testing.assert_array_equal(actual[0], alpha)
    np.testing.assert_array_equal(actual[1], beta)
    np.testing.assert_array_equal(
        jax.jacfwd(bounds, (0, 1))(alpha[0], beta[0]), ((1.0, 0.0), (0.0, 1.0))
    )


@pytest.mark.parametrize(
    "op,parents,valid,invalid",
    [
        (ir.Exponential(), (2.0,), 0.5, -1.0),
        (ir.Gamma(), (2.0, 3.0), 0.5, -1.0),
        (ir.Lognormal(), (0.0, 1.0), 0.5, -1.0),
        (ir.Beta(), (2.0, 3.0), 0.5, 1.5),
        (ir.Uniform(), (-1.0, 2.0), 0.5, 3.0),
        (ir.Bernoulli(), (0.3,), 1.0, 2.0),
        (ir.BernoulliLogit(), (0.3,), 0.0, -1.0),
        (ir.Binomial(), (5, 0.3), 2.0, 6.0),
        (ir.BetaBinomial(), (5, 2.0, 3.0), 2.0, -1.0),
        (ir.Categorical(), (jnp.array([0.2, 0.8]),), 1, 2),
        (ir.Poisson(), (2.0,), 2.0, -1.0),
        (ir.Dirichlet(), (jnp.array([2.0, 3.0]),), jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3])),
        (
            ir.Multinomial(),
            (5, jnp.array([0.2, 0.8])),
            jnp.array([1.0, 4.0]),
            jnp.array([1.0, 3.0]),
        ),
        (ir.Wishart(), (5.0, jnp.eye(2)), jnp.eye(2), jnp.diag(jnp.array([-1.0, 1.0]))),
    ],
)
def test_support_is_enforced_without_truncation(op, parents, valid, invalid):
    distribution = lower_random(op, parents)
    reference = distribution_constructors[type(op)](*parents)
    expected = reference.log_prob(jnp.asarray(valid))
    np.testing.assert_allclose(jax.jit(distribution.log_prob)(valid), expected, atol=1e-6)
    assert distribution.log_prob(invalid) == -jnp.inf
    assert jax.jit(distribution.log_prob)(invalid) == -jnp.inf
    values = jnp.stack([jnp.asarray(valid), jnp.asarray(invalid)])
    result = jax.jit(jax.vmap(distribution.log_prob))(values)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, jnp.array([expected, -jnp.inf]), atol=1e-6)
    if jnp.issubdtype(jnp.asarray(valid).dtype, jnp.inexact):
        assert jnp.isnan(distribution.log_prob(jnp.full(jnp.shape(valid), jnp.nan)))
    key = jax.random.key(0)
    np.testing.assert_array_equal(distribution.sample(key), reference.sample(key))


@pytest.mark.parametrize(
    "op,parents",
    [
        (ir.Poisson(), (2.0,)),
        (ir.Binomial(), (5, 0.3)),
    ],
)
def test_discrete_support_rejects_nonintegers(op, parents):
    assert lower_random(op, parents).log_prob(0.5) == -jnp.inf


@pytest.mark.parametrize(
    "op,parents,invalid",
    [
        (ir.Gamma(), (2.0, 3.0), -1.0),
        (ir.Lognormal(), (0.0, 1.0), -1.0),
        (ir.Beta(), (2.0, 3.0), 1.5),
        (ir.Dirichlet(), (jnp.array([2.0, 3.0]),), jnp.array([-1.0, 2.0])),
        (ir.Wishart(), (5.0, jnp.eye(2)), jnp.diag(jnp.array([-1.0, 1.0]))),
    ],
)
def test_outside_support_value_gradients_are_zero_not_nan(op, parents, invalid):
    distribution = lower_random(op, parents)
    actual = jax.jit(jax.grad(distribution.log_prob))(jnp.asarray(invalid))
    np.testing.assert_array_equal(actual, jnp.zeros_like(invalid))


def test_outside_support_parameter_gradients_are_zero_under_vmap():
    def log_prob(value, rate):
        return lower_random(ir.Exponential(), (rate,)).log_prob(value)

    values = jnp.array([-1.0, 0.5, 1.5])
    gradient = jax.jit(jax.vmap(jax.grad(log_prob, 1), (0, None)))(values, 2.0)
    np.testing.assert_allclose(gradient, jnp.array([0.0, 0.0, -1.0]))
