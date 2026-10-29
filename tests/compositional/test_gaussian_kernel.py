"""Compare nested vectorization of a Gaussian kernel and gradients through shared inputs with JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from pangolin import functional, interface as pi, ir, jax_backend

from .helpers import assert_matches


def pangolin_vfor(points: ir.RV, log_length: ir.RV) -> ir.RV:
    def entry(i: ir.RV, j: ir.RV) -> ir.RV:
        difference = points[i, :] - points[j, :]
        return pi.exp(-0.5 * pi.sum(difference * difference, 0) / pi.exp(2 * log_length))

    return pi.vfor(entry, i=3, j=3)


def pangolin_vmap(points: ir.RV, log_length: ir.RV) -> ir.RV:
    def entry(left: ir.RV, right: ir.RV) -> ir.RV:
        difference = left - right
        return pi.exp(-0.5 * pi.sum(difference * difference, 0) / pi.exp(2 * log_length))

    return pi.vmap(lambda left: pi.vmap(entry, (None, 0))(left, points))(points)


def pangolin_loop(points: ir.RV, log_length: ir.RV) -> ir.RV:
    def entry(i: ir.RV, j: ir.RV) -> ir.RV:
        difference = points[i, :] - points[j, :]
        return pi.exp(-0.5 * pi.sum(difference * difference, 0) / pi.exp(2 * log_length))

    return functional.loop(entry, interface=pi.default, i=3, j=3)


def jax_kernel(points: jax.Array, log_length: jax.Array) -> jax.Array:
    difference = points[:, None, :] - points[None, :, :]
    return jnp.exp(-0.5 * jnp.sum(difference**2, axis=-1) / jnp.exp(2 * log_length))


@pytest.mark.parametrize("kernel", [pangolin_vfor, pangolin_vmap, pangolin_loop])
def test_gaussian_kernel(kernel: Callable[[ir.RV, ir.RV], ir.RV]):
    x = jnp.array([[-0.4, 0.7], [0.3, -0.2], [1.1, 0.5]])
    cotangent = jnp.array([[0.2, -0.7, 0.4], [1.2, 0.3, -0.1], [-0.5, 0.8, 0.6]])
    points, log_length = pi.constant(x), pi.constant(0.1)
    results = (
        kernel(points, log_length),
        pi.vjp(kernel)(pi.constant(cotangent), points, log_length),
        pi.vjp(kernel, 1)(pi.constant(cotangent), points, log_length),
    )

    def evaluate(length: jax.Array):
        return jax_backend.ancestor_sample_graph(results, None, {log_length: length})

    compiled = jax.jit(evaluate)
    for length in (jnp.asarray(-0.3), jnp.asarray(0.6)):
        value, pullback = jax.vjp(jax_kernel, x, length)
        dx, dl = pullback(cotangent)
        expected = (value, dx, dl)
        assert_matches(evaluate(length), expected)
        assert_matches(compiled(length), expected)
