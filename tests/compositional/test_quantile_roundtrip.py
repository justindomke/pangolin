"""Quantiles of a transformed user-defined distribution compose with CDF, VMap, and grad."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri

from pangolin import interface as pi, ir, jax_backend


def lognormal(mean: ir.RV, log_scale: ir.RV) -> ir.RV:
    return pi.tforms.exp(pi.normal)(mean, pi.exp(log_scale))


def test_quantile_roundtrip_and_probability_derivative():
    probabilities = pi.constant([0.1, 0.3, 0.7, 0.9])
    mean, log_scale = pi.constant(0.3), pi.constant(-0.4)
    quantile = pi.quantile(pi.composite(lognormal, simplify=False))
    values = pi.vmap(quantile, (0, None, None))(probabilities, mean, log_scale)
    recovered = pi.vmap(pi.cdf(lognormal), (0, None, None))(values, mean, log_scale)
    slopes = pi.vmap(pi.grad(quantile), (0, None, None))(probabilities, mean, log_scale)
    result = jax.jit(lambda: jax_backend.ancestor_sample_graph((values, recovered, slopes), None))()
    p = jnp.array([0.1, 0.3, 0.7, 0.9])

    def reference(probability):
        return jnp.exp(0.3 + jnp.exp(-0.4) * ndtri(probability))

    for actual, expected in zip(
        result, (reference(p), p, jax.vmap(jax.grad(reference))(p)), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
