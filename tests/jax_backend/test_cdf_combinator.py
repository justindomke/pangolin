import math

import jax
import numpy as np

from pangolin import interface as pi, jax_backend


def test_normal_cdf_with_deterministic_parameter_graph():
    def fun(mean, log_scale):
        return pi.normal(mean, pi.exp(log_scale))

    result = pi.cdf(fun)(pi.constant(2.0), 0.5, math.log(1.5))
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    expected = (1 + math.erf(1 / math.sqrt(2))) / 2
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_poisson_cdf_includes_evaluation_value():
    result = pi.cdf(pi.poisson)(pi.constant(1), 2.0)
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, 3 * math.exp(-2), rtol=1e-6)
