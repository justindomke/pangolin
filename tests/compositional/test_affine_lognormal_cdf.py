"""Compare nested transformed CDFs when a captured scale changes sign and support moves with parameters."""

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

from pangolin import interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_cdf(value: ir.RV, scale: ir.RV, shift: ir.RV) -> ir.RV:
    def distribution() -> ir.RV:
        positive = pi.tforms.exp(pi.normal)
        scaled = pi.tforms.scale_by(scale)(positive)
        return pi.tforms.shift_by(shift)(scaled)(0.3, 0.8)

    return pi.cdf(distribution)(value)


def jax_cdf(value: jax.Array, scale: jax.Array, shift: jax.Array) -> jax.Array:
    x = (value - shift) / scale
    safe_x = jnp.where(x > 0, x, 1.0)
    base = ndtr((jnp.log(safe_x) - 0.3) / 0.8)
    base = jnp.where(x > 0, base, 0.0)
    return jnp.where(scale > 0, base, 1.0 - base)


def test_affine_lognormal_cdf():
    value, scale, shift = pi.constant(1.4), pi.constant(0.7), pi.constant(-0.2)
    results = (
        pangolin_cdf(value, scale, shift),
        pi.grad(pangolin_cdf)(value, scale, shift),
        pi.grad(pangolin_cdf, 1)(value, scale, shift),
        pi.grad(pangolin_cdf, 2)(value, scale, shift),
    )

    def evaluate(y: jax.Array, a: jax.Array, b: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {value: y, scale: a, shift: b})

    compiled = jax.jit(evaluate)
    for y, a, b in [(1.4, 0.7, -0.2), (-1.5, -0.9, 0.4), (0.3, 0.7, 0.2), (-0.2, -0.9, 0.1)]:
        args = (jnp.asarray(y), jnp.asarray(a), jnp.asarray(b))
        expected = (jax_cdf(*args), *(jax.grad(jax_cdf, i)(*args) for i in range(3)))
        assert_matches(evaluate(*args), expected)
        assert_matches(compiled(*args), expected)
