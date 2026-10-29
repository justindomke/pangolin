"""Two-axis truncated lognormal recurrences with likelihoods, scores, and likelihood curvature."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr
from jax.scipy.stats import norm

from pangolin import interface as pi, ir, jax_backend
from pangolin.functional import loop

from .helpers import assert_matches


def pangolin_field(coefficient: ir.RV) -> ir.RV:
    positive = pi.truncated(pi.tforms.exp(pi.normal), 0.5, 3.0)

    def cell(prev, t, s):
        if t is None:
            return 0.8 + 0.1 * s
        if s is None:
            return 1.0 + 0.2 * t
        mean = coefficient * pi.log(prev[t - 1, s]) + 0.3 * pi.log(prev[t, s - 1])
        return positive(mean, 0.6)

    return loop(cell, interface=pi.default, t=[2], s=[3])


def pangolin_likelihood(coefficient: ir.RV, values: ir.RV) -> ir.RV:
    return pi.log_prob(pangolin_field)(values, coefficient)


def jax_likelihood(coefficient, values):
    old_t = jnp.concatenate(((0.8 + 0.1 * jnp.arange(3))[None, :], values[:-1]), axis=0)
    old_s = jnp.concatenate(((1.0 + 0.2 * jnp.arange(2))[:, None], values[:, :-1]), axis=1)
    means = coefficient * jnp.log(old_t) + 0.3 * jnp.log(old_s)
    mass = ndtr((jnp.log(3.0) - means) / 0.6) - ndtr((jnp.log(0.5) - means) / 0.6)
    return jnp.sum(norm.logpdf(jnp.log(values), means, 0.6) - jnp.log(values) - jnp.log(mass))


def test_likelihood_parameter_and_value_gradients_and_curvature():
    coefficient = pi.constant(0.4)
    values = jnp.array([[0.7, 1.2, 2.1], [1.4, 0.9, 1.8]])
    observed = pi.constant(values)
    results = (
        pangolin_likelihood(coefficient, observed),
        pi.grad(pangolin_likelihood)(coefficient, observed),
        pi.grad(pangolin_likelihood, 1)(coefficient, observed),
        pi.grad(pi.grad(pangolin_likelihood))(coefficient, observed),
    )

    def evaluate(a):
        return jax_backend.ancestor_sample_graph(results, None, {coefficient: a})

    compiled = jax.jit(evaluate)
    for a in (jnp.array(0.4), jnp.array(-0.2)):
        expected = (
            jax_likelihood(a, values),
            jax.grad(jax_likelihood)(a, values),
            jax.grad(jax_likelihood, 1)(a, values),
            jax.grad(jax.grad(jax_likelihood))(a, values),
        )
        assert_matches(evaluate(a), expected, rtol=8e-5, atol=8e-6)
        assert_matches(compiled(a), expected, rtol=8e-5, atol=8e-6)


def test_samples_stay_in_interval_and_have_correct_joint_density():
    output = pangolin_field(pi.constant(0.4))
    sample = jax_backend.ancestor_sampler(output)
    values = jax.jit(jax.vmap(sample))(jax.random.split(jax.random.key(13), 128))
    assert values.shape == (128, 2, 3)
    assert np.all((values > 0.5) & (values < 3.0))
    actual = jax.jit(jax.vmap(jax_backend.ancestor_log_prob(output)))(values)
    expected = jax.vmap(lambda x: jax_likelihood(0.4, x))(values)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
