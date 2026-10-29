"""Compare joint and component CDFs of negative lognormals and their derivatives with JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import ndtr

from pangolin import functional, interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_vmap_distribution(means: ir.RV, scale: ir.RV) -> ir.RV:
    reflected = pi.tforms.scale_by(-2.0)(pi.tforms.exp(pi.normal))
    return pi.vmap(reflected, (0, None))(means, scale)


def pangolin_vmap_components(values: ir.RV, means: ir.RV, scale: ir.RV) -> ir.RV:
    reflected = pi.tforms.scale_by(-2.0)(pi.tforms.exp(pi.normal))
    return pi.vmap(pi.cdf(reflected), (0, 0, None))(values, means, scale)


def pangolin_vfor_distribution(means: ir.RV, scale: ir.RV) -> ir.RV:
    reflected = pi.tforms.scale_by(-2.0)(pi.tforms.exp(pi.normal))
    return pi.vfor(lambda i: reflected(means[i], scale), i=3)


def pangolin_vfor_components(values: ir.RV, means: ir.RV, scale: ir.RV) -> ir.RV:
    reflected = pi.tforms.scale_by(-2.0)(pi.tforms.exp(pi.normal))
    return pi.vfor(lambda i: pi.cdf(reflected)(values[i], means[i], scale), i=3)


def pangolin_loop_distribution(means: ir.RV, scale: ir.RV) -> ir.RV:
    reflected = pi.tforms.scale_by(-2.0)(pi.tforms.exp(pi.normal))
    return functional.loop(lambda i: reflected(means[i], scale), interface=pi.default, i=3)


def pangolin_loop_components(values: ir.RV, means: ir.RV, scale: ir.RV) -> ir.RV:
    reflected = pi.tforms.scale_by(-2.0)(pi.tforms.exp(pi.normal))
    return functional.loop(
        lambda i: pi.cdf(reflected)(values[i], means[i], scale), interface=pi.default, i=3
    )


def jax_components(values: jax.Array, means: jax.Array, scale: jax.Array) -> jax.Array:
    safe_values = jnp.where(values < 0, values, -1.0)
    probabilities = ndtr((means - jnp.log(-safe_values / 2)) / scale)
    return jnp.where(values < 0, probabilities, 1.0)


def jax_joint(values: jax.Array, means: jax.Array, scale: jax.Array) -> jax.Array:
    return jnp.prod(jax_components(values, means, scale))


@pytest.mark.parametrize(
    "distribution,components",
    [
        (pangolin_vmap_distribution, pangolin_vmap_components),
        (pangolin_vfor_distribution, pangolin_vfor_components),
        (pangolin_loop_distribution, pangolin_loop_components),
    ],
    ids=["vmap", "vfor", "loop"],
)
def test_reflected_lognormal_cdf(
    distribution: Callable[[ir.RV, ir.RV], ir.RV],
    components: Callable[[ir.RV, ir.RV, ir.RV], ir.RV],
):
    values = pi.constant([-1.0, -2.0, -3.0])
    means = pi.constant([-0.2, 0.3, 0.6])
    scale = pi.normal(0, 1)
    weights = pi.constant([0.7, -1.3, 2.1])
    joint_cdf = pi.cdf(distribution)
    results = (
        joint_cdf(values, means, scale),
        components(values, means, scale),
        pi.grad(joint_cdf)(values, means, scale),
        pi.grad(joint_cdf, 2)(values, means, scale),
        pi.vjp(components, 2)(weights, values, means, scale),
    )

    def evaluate(y: jax.Array, mu: jax.Array, sigma: jax.Array):
        return jax_backend.ancestor_sample_graph(
            results, key=None, boundary={values: y, means: mu, scale: sigma}
        )

    compiled = jax.jit(evaluate)
    for y, sigma in [([-0.7, -1.5, -2.3], 0.8), ([-0.4, -1.5, -0.1], 1.2)]:
        args = (jnp.array(y), jnp.array([-0.2, 0.3, 0.6]), jnp.asarray(sigma))
        _, pullback = jax.vjp(lambda s: jax_components(args[0], args[1], s), args[2])
        expected = (
            jax_joint(*args),
            jax_components(*args),
            jax.grad(jax_joint)(*args),
            jax.grad(jax_joint, 2)(*args),
            pullback(jnp.array([0.7, -1.3, 2.1]))[0],
        )
        assert_matches(evaluate(*args), expected)
        assert_matches(compiled(*args), expected)
