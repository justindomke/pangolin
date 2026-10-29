"""Batched truncated random walks with moving bounds, likelihoods, and graph derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr
from jax.scipy.stats import norm
from scipy import stats

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_paths(coefficient: ir.RV, starts: ir.RV, shifts: ir.RV) -> ir.RV:
    def trajectory(start: ir.RV, increments: ir.RV) -> ir.RV:
        def step(previous: ir.RV, shift: ir.RV) -> ir.RV:
            return pi.truncated(pi.normal, previous - 0.4, previous + 0.9)(
                coefficient * previous + shift, 0.6
            )

        return pi.scan(pi.composite(step, simplify=False), 3)(start, increments)

    return pi.vmap(trajectory)(starts, shifts)


def pangolin_likelihood(coefficient: ir.RV, values: ir.RV, starts: ir.RV, shifts: ir.RV):
    return pi.log_prob(pangolin_paths)(values, coefficient, starts, shifts)


def jax_likelihood(coefficient, values, starts, shifts):
    previous = jnp.concatenate((starts[:, None], values[:, :-1]), axis=1)
    means = coefficient * previous + shifts
    lo, hi = previous - 0.4, previous + 0.9
    mass = ndtr((hi - means) / 0.6) - ndtr((lo - means) / 0.6)
    return jnp.sum(norm.logpdf(values, means, 0.6) - jnp.log(mass))


def test_truncated_walk_likelihood_and_derivatives():
    initial = jnp.array([-0.1, 0.3])
    increments = jnp.array([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]])
    observations = jnp.array([[0.1, 0.3, 0.15], [0.4, 0.2, 0.5]])
    coefficient = pi.constant(0.4)
    args = (coefficient, pi.constant(observations), pi.constant(initial), pi.constant(increments))
    results = (
        pangolin_likelihood(*args),
        *(pi.grad(pangolin_likelihood, i)(*args) for i in range(4)),
        pi.grad(pi.grad(pangolin_likelihood))(*args),
    )

    def evaluate(a):
        return jax_backend.ancestor_sample_graph(results, None, {coefficient: a})

    compiled = jax.jit(evaluate)
    for a in (jnp.asarray(0.4), jnp.asarray(-0.3)):
        numeric_args = (a, observations, initial, increments)
        expected = (
            jax_likelihood(*numeric_args),
            *(jax.grad(jax_likelihood, i)(*numeric_args) for i in range(4)),
            jax.grad(jax.grad(jax_likelihood))(*numeric_args),
        )
        assert_matches(evaluate(a), expected, rtol=4e-5)
        assert_matches(compiled(a), expected, rtol=4e-5)


def test_truncated_walk_samples_and_joint_density():
    initial = jnp.array([-0.1, 0.3])
    increments = jnp.array([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]])
    paths = pangolin_paths(pi.constant(0.4), pi.constant(initial), pi.constant(increments))
    sample = jax.jit(jax_backend.ancestor_sampler(paths))
    values = np.asarray(jax.jit(jax.vmap(sample))(jax.random.split(jax.random.key(29), 2048)))
    assert values.shape == (2048, 2, 3)
    previous = np.concatenate(
        (np.broadcast_to(initial, (2048, 2))[..., None], values[..., :-1]), axis=2
    )
    means = 0.4 * previous + np.asarray(increments)
    lo, hi = previous - 0.4, previous + 0.9
    reference = stats.truncnorm((lo - means) / 0.6, (hi - means) / 0.6, loc=means, scale=0.6)
    assert np.all((values >= lo) & (values <= hi))
    probabilities = reference.cdf(values)
    for i in range(2):
        for t in range(3):
            assert stats.kstest(probabilities[:, i, t], "uniform").statistic < 0.045
    np.testing.assert_allclose(
        np.corrcoef(probabilities.reshape(2048, 6), rowvar=False), np.eye(6), atol=0.09
    )

    log_prob = jax.jit(jax.vmap(jax_backend.ancestor_log_prob(paths)))
    expected = stats.truncnorm.logpdf(
        values, (lo - means) / 0.6, (hi - means) / 0.6, loc=means, scale=0.6
    ).sum(axis=(1, 2))
    np.testing.assert_allclose(log_prob(values), expected, atol=8e-6)
