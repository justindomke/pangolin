import math

import jax
import numpy as np

from pangolin import functional, interface as pi, jax_backend


def test_parameter_graph_matches_normal_log_density():
    def fun(mean, log_scale):
        return pi.normal(mean, pi.exp(log_scale))

    result = functional.log_prob(fun)(pi.constant(2.0), 0.5, math.log(1.5))
    expected = -0.5 - math.log(1.5) - 0.5 * math.log(2 * math.pi)
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_vmap_of_log_prob_is_componentwise():
    fun = functional.vmap(functional.log_prob(pi.normal), (0, None, None))
    result = fun(pi.constant([0.0, 1.0]), 0.0, 1.0)
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, -0.5 * np.array([0.0, 1.0]) - 0.5 * math.log(2 * math.pi))


def test_log_prob_of_vmap_is_joint():
    fun = functional.log_prob(functional.vmap(pi.normal, (0, None)))
    result = fun(pi.constant([0.0, 1.0]), pi.constant([0.0, 0.0]), 1.0)
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, -0.5 - math.log(2 * math.pi), rtol=1e-6)
