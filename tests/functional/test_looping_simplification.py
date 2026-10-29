"""Indexing regressions adapted from vfor, checked against both IR and NumPy."""

import jax
import numpy as np
import pytest

from pangolin import dag, interface as pi, ir, jax_backend
from pangolin.functional import loop


def evaluate(output):
    return jax.jit(jax_backend.ancestor_sampler(output))(jax.random.key(0))


def op_contains_index(op: ir.Op) -> bool:
    if isinstance(op, ir.Index):
        return True
    if isinstance(op, ir.VMap):
        return op_contains_index(op.base_op)
    if isinstance(op, ir.Scan):
        return op_contains_index(op.base_op)
    if isinstance(op, ir.Composite):
        return any((op_contains_index(component) for component in op.ops))
    return False


def contains_index(x) -> bool:
    return any((op_contains_index(node.op) for node in dag.upstream_nodes(x)))


def test_indexed_exp():
    a = pi.constant([1, 2, 3])
    b = loop(lambda i: pi.exp(a[i]), i=3, interface=pi.default)
    assert b.op == ir.VMap(ir.Exp(), [0], 3)
    assert b.parents == (a,)
    np.testing.assert_allclose(evaluate(b), np.exp([1, 2, 3]))


def test_unused_axis_exp():
    a = pi.constant(1.5)
    b = loop(lambda i: pi.exp(a), i=3, interface=pi.default)
    assert b.op == ir.VMap(ir.Exp(), [None], 3)
    np.testing.assert_allclose(evaluate(b), np.full(3, np.exp(1.5)))


def test_outer_product():
    a = pi.constant([1, 2, 3])
    b = pi.constant([4, 5])
    c = loop(lambda i, j: a[i] * b[j], i=3, j=2, interface=pi.default)
    assert c.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, None], 3)
    assert c.parents == (a, b)
    np.testing.assert_array_equal(evaluate(c), np.outer([1, 2, 3], [4, 5]))


def test_mapped_left_operand():
    a = pi.constant([1, 2, 3])
    b = pi.constant(4)
    c = loop(lambda i: a[i] * b, i=3, interface=pi.default)
    assert c.op == ir.VMap(ir.Mul(), [0, None], 3)
    assert c.parents == (a, b)
    np.testing.assert_array_equal(evaluate(c), [4, 8, 12])


def test_mapped_right_operand():
    a = pi.constant(1)
    b = pi.constant([2, 3, 4])
    c = loop(lambda i: a * b[i], i=3, interface=pi.default)
    assert c.op == ir.VMap(ir.Mul(), [None, 0], 3)
    assert c.parents == (a, b)
    np.testing.assert_array_equal(evaluate(c), [2, 3, 4])


def test_rectangular_indexing():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    y = loop(lambda i, j: pi.exp(x[i, j]), i=2, j=3, interface=pi.default)
    assert y.shape == (2, 3)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), [0], 3), [0], 2)
    assert y.parents == (x,)
    np.testing.assert_allclose(evaluate(y), np.exp([[1, 2, 3], [4, 5, 6]]))


def test_transposed_indexing():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    y = loop(lambda i, j: pi.exp(x[j, i]), i=3, j=2, interface=pi.default)
    assert y.shape == (3, 2)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), [0], 2), [1], 3)
    assert y.parents == (x,)
    np.testing.assert_allclose(evaluate(y), np.exp([[1, 4], [2, 5], [3, 6]]))


def test_parameter_order_overrides_size_keyword_order():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    y = loop(lambda j, i: pi.exp(x[i, j]), i=2, j=3, interface=pi.default)
    assert y.shape == (3, 2)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), [0], 2), [1], 3)
    assert y.parents == (x,)
    np.testing.assert_allclose(evaluate(y), np.exp([[1, 4], [2, 5], [3, 6]]))


def test_operands_with_different_axis_orders():
    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    b = pi.constant([[1, 2], [3, 4], [5, 6]])
    c = loop(lambda i, j: a[i, j] * b[j, i], i=2, j=3, interface=pi.default)
    assert c.shape == (2, 3)
    assert c.op == ir.VMap(ir.VMap(ir.Mul(), [0, 0], 3), [0, 1], 2)
    assert c.parents == (a, b)
    np.testing.assert_array_equal(evaluate(c), [[1, 6, 15], [8, 20, 36]])


def test_three_dimensional_identity_index():
    x = pi.constant([[[0.0] * 5] * 3] * 2)
    y = loop(lambda i, j, k: pi.exp(x[i, j, k]), i=2, j=3, k=5, interface=pi.default)
    assert y.shape == (2, 3, 5)
    assert y.op == ir.VMap(ir.VMap(ir.VMap(ir.Exp(), [0], 5), [0], 3), [0], 2)
    assert y.parents == (x,)
    np.testing.assert_array_equal(evaluate(y), np.ones((2, 3, 5)))


def test_operands_skip_different_axes():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    z = pi.constant([7, 8, 9, 10])
    y = loop(lambda i, j, k: x[i, k] * z[j], i=2, j=4, k=3, interface=pi.default)
    assert y.shape == (2, 4, 3)
    assert y.op == ir.VMap(ir.VMap(ir.VMap(ir.Mul(), [0, None], 3), [None, 0], 4), [0, None], 2)
    assert y.parents == (x, z)
    np.testing.assert_array_equal(
        evaluate(y),
        np.array([[1, 2, 3], [4, 5, 6]])[:, None, :] * np.array([7, 8, 9, 10])[None, :, None],
    )


def test_unused_inner_axis_with_indexed_capture():
    a = pi.constant([0, 1, 2])
    x = loop(lambda i, j: pi.exp(a[i]), i=3, j=4, interface=pi.default)
    assert x.op == ir.VMap(ir.VMap(ir.Exp(), [None], 4), [0], 3)
    assert x.parents == (a,)
    np.testing.assert_allclose(evaluate(x), np.broadcast_to(np.exp(np.arange(3))[:, None], (3, 4)))


def test_unused_inner_axis_with_coordinate():
    x = loop(lambda i, j: pi.exp(i), i=3, j=4, interface=pi.default)
    assert x.op == ir.VMap(ir.VMap(ir.Exp(), [None], 4), [0], 3)
    assert x.parents[0].op == ir.Constant([0, 1, 2])
    np.testing.assert_allclose(evaluate(x), np.broadcast_to(np.exp(np.arange(3))[:, None], (3, 4)))


def test_nested_index_multiplication_retains_both_dimensions():
    values = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    data = pi.constant(values)
    result = loop(lambda i, j: data[i, j] * 2, interface=pi.default, i=2, j=3)
    assert result.shape == (2, 3)
    assert result.op == ir.VMap(ir.VMap(ir.Mul(), (0, None), 3), (0, None), 2)
    assert result.parents[0] is data
    assert result.parents[1].op == ir.Constant(2)
    assert not contains_index(result)
    np.testing.assert_allclose(evaluate(result), values * 2)


def test_partial_index_then_reduce():
    data = pi.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = loop(lambda j: pi.sum(data[:, j], 0), interface=pi.default, j=3)
    assert result.shape == (3,)
    assert result.op == ir.VMap(ir.Sum(0), (1,), 3)
    assert result.parents == (data,)
    assert not contains_index(result)
    np.testing.assert_array_equal(evaluate(result), [5.0, 7.0, 9.0])


@pytest.mark.parametrize(
    "axes,index_remains",
    [
        ((0, 1, 2), False),
        ((0, 2, 1), True),
        ((1, 0, 2), False),
        ((1, 2, 0), True),
        ((2, 0, 1), False),
        ((2, 1, 0), True),
    ],
)
def test_all_three_dimensional_axis_permutations(axes, index_remains):
    values = np.arange(30).reshape(2, 3, 5) / 10
    data = pi.constant(values)

    def model(i, j, k):
        coordinates = (i, j, k)
        indices = tuple(coordinates[axes.index(axis)] for axis in range(3))
        return pi.exp(data[indices])

    shape = tuple(values.shape[axis] for axis in axes)
    result = loop(model, interface=pi.default, i=shape[0], j=shape[1], k=shape[2])
    assert result.shape == shape
    assert contains_index(result) is index_remains
    np.testing.assert_allclose(evaluate(result), np.exp(values.transpose(axes)), rtol=1e-6)


def test_nonidentity_coordinate_index_is_not_removed():
    data = pi.constant([2.0, 5.0, 11.0, 23.0])
    result = loop(lambda i: data[3 - i] * 2, interface=pi.default, i=4)
    assert contains_index(result)
    np.testing.assert_array_equal(evaluate(result), [46.0, 22.0, 10.0, 4.0])


def test_repeated_coordinate_selects_diagonal():
    values = np.arange(9).reshape(3, 3)
    data = pi.constant(values)
    result = loop(lambda i: data[i, i] + 1, interface=pi.default, i=3)
    assert result.shape == (3,)
    np.testing.assert_array_equal(evaluate(result), np.diag(values) + 1)
