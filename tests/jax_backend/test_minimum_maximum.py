"""Minimum/maximum values, infinite bounds, NaNs, and derivatives through graph combinators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pangolin import interface as pi, ir, jax_backend


@pytest.mark.parametrize("op,reference", [(ir.Minimum(), np.minimum), (ir.Maximum(), np.maximum)])
def test_values_infinities_and_nans(op, reference):
    values = np.array([-np.inf, -2.0, 0.0, 3.0, np.inf, np.nan])
    left, right = np.meshgrid(values, values)

    def evaluate(a, b):
        return jax_backend.lower_deterministic(op, (a, b))

    mapped = jax.vmap(evaluate)
    for fun in [mapped, jax.jit(mapped)]:
        actual = fun(left.ravel(), right.ravel())
        np.testing.assert_allclose(actual, reference(left, right).ravel(), equal_nan=True)
    for a, b in [(2, -1), (-1, 2), (3, 3)]:
        np.testing.assert_array_equal(evaluate(a, b), reference(a, b))


@pytest.mark.parametrize("name,reference", [("minimum", np.minimum), ("maximum", np.maximum)])
def test_numpy_broadcasting_compiles_to_elementwise_function(name, reference):
    configured = pi.Interface(broadcasting="numpy")
    a, b = configured.constant([[0], [0]]), configured.constant([0, 0, 0])
    result = getattr(configured, name)(a, b)

    def evaluate(left, right):
        return jax_backend.ancestor_sample_graph(result, None, {a: left, b: right})

    left, right = jnp.array([[-1.0], [2.0]]), jnp.array([0.0, 1.0, 3.0])
    np.testing.assert_array_equal(jax.jit(evaluate)(left, right), reference(left, right))


@pytest.mark.parametrize(
    "fun,expected_a,expected_b",
    [
        (pi.minimum, [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]),
        (pi.maximum, [0.0, 1.0, 0.5], [1.0, 0.0, 0.5]),
    ],
)
def test_gradients_under_vmap_including_jax_tie_convention(fun, expected_a, expected_b):
    a, b = pi.constant([-1.0, 3.0, 2.0]), pi.constant([2.0, 1.0, 2.0])
    gradients = pi.vmap(pi.grad(fun))(a, b), pi.vmap(pi.grad(fun, 1))(a, b)
    actual = jax.jit(lambda: jax_backend.ancestor_sample_graph(gradients, None))()
    np.testing.assert_array_equal(actual[0], expected_a)
    np.testing.assert_array_equal(actual[1], expected_b)


@pytest.mark.parametrize(
    "fun,start,expected",
    [(pi.minimum, np.inf, [2.0, 1.0, 1.0]), (pi.maximum, -np.inf, [2.0, 2.0, 3.0])],
)
def test_scan_with_infinite_initial_bound(fun, start, expected):
    path = pi.scan(fun, 3)(pi.constant(start), pi.constant([2.0, 1.0, 3.0]))
    actual = jax.jit(lambda: jax_backend.ancestor_sample_graph(path, None))()
    np.testing.assert_array_equal(actual, expected)


def test_symbolic_support_intersection_with_infinite_endpoints():
    lo, hi = pi.constant(0.0), pi.constant(1.0)
    bounds = pi.maximum(lo, 0.0), pi.minimum(hi, np.inf)

    def evaluate(lower, upper):
        return jax_backend.ancestor_sample_graph(bounds, None, {lo: lower, hi: upper})

    for lower, upper in [(-np.inf, np.inf), (-1.0, 2.0), (0.5, 3.0)]:
        actual = jax.jit(evaluate)(jnp.asarray(lower), jnp.asarray(upper))
        np.testing.assert_array_equal(actual, [max(lower, 0), upper])
