import jax
import numpy as np
import pytest

from pangolin import interface as pi, jax_backend


@pytest.mark.parametrize(
    "fun, x, expected",
    [
        (lambda x: x * x, 3.0, 6.0),
        (lambda x: pi.exp(x * x), 2.0, 4 * np.exp(4)),
        (lambda x: pi.sum(x * x, 0), [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        (lambda x: pi.sum(pi.sum(x * x, 0), 0), [[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]),
        (lambda x: x, 3.0, 1.0),
        (lambda x: pi.constant(5), [1.0, 2.0], [0.0, 0.0]),
    ],
)
def test_numerical_gradients(fun, x, expected):
    result = pi.grad(fun)(pi.constant(x))
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_second_argument_and_aliasing():
    x = pi.constant(3.0)
    result = pi.grad(lambda a, b: a * b + b, 1)(x, x)
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, 4.0)


def test_grad_log_prob():
    result = pi.grad(pi.log_prob(pi.normal))(pi.constant(2.0), 0.0, 2.0)
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, -0.5)


def test_second_derivative():
    result = pi.grad(pi.grad(lambda x: x * x * x))(pi.constant(2.0))
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, 12.0)


def test_vjp_array_output_and_rectangular_jacobian():
    matrix = pi.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = pi.vjp(lambda x: matrix @ x)(pi.constant([2.0, -1.0]), pi.constant([0.0, 0.0, 0.0]))
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, [-2.0, -1.0, 0.0])


def test_vjp_does_not_differentiate_cotangent_dependencies():
    x = pi.constant(3.0)
    result = pi.vjp(lambda x: x * x)(x * x, x)
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, 54.0)


def test_vjp_array_contributions_accumulate():
    result = pi.vjp(lambda x: x * x)(pi.constant([3.0, 4.0]), pi.constant([1.0, 2.0]))
    actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, [6.0, 16.0])
