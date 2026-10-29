from pangolin import interface as pi, ir


from pangolin import dag

import numpy as np
import pytest

from pangolin.functional.vfor_looping import vfor_simple


@pytest.fixture
def evaluate(backend):
    def call(*rvs):
        values = backend.ancestor_sample_flat(list(rvs), None)
        return tuple(np.asarray(value) for value in values)

    return call


def op_contains_index(op: ir.Op) -> bool:
    if isinstance(op, ir.Index):
        return True

    if isinstance(op, ir.VMap):
        return op_contains_index(op.base_op)

    if isinstance(op, ir.Scan):
        return op_contains_index(op.base_op)

    if isinstance(op, ir.Composite):
        return any(op_contains_index(component) for component in op.ops)

    return False


def contains_index(x) -> bool:
    return any(op_contains_index(node.op) for node in dag.upstream_nodes(x))


def test_vfor_simple_coordinate_values(evaluate):
    x = vfor_simple(lambda i: i * 4.4, 3, interface=pi.default)

    assert x.shape == (3,)

    [value] = evaluate(x)
    np.testing.assert_allclose(
        value,
        np.array([0.0, 4.4, 8.8]),
    )


def test_vfor_simple_unused_coordinate(evaluate):
    x = vfor_simple(lambda i: pi.constant(7) * 2, 4, interface=pi.default)

    assert x.shape == (4,)

    [value] = evaluate(x)
    np.testing.assert_array_equal(
        value,
        np.full(4, 14),
    )


def test_vfor_simple_index_array(evaluate):
    data = pi.constant([1.1, 2.2, 3.3])

    x = vfor_simple(lambda i: data[i] * 2, 3, interface=pi.default)

    assert x.shape == (3,)
    assert not contains_index(x)

    [value] = evaluate(x)
    np.testing.assert_allclose(
        value,
        np.array([2.2, 4.4, 6.6]),
    )


def test_vfor_simple_outer_product(evaluate):
    x = vfor_simple(lambda i, j: i * j, 2, 3, interface=pi.default)

    assert x.shape == (2, 3)

    [value] = evaluate(x)
    np.testing.assert_array_equal(
        value,
        np.outer(
            np.arange(2),
            np.arange(3),
        ),
    )


def test_vfor_simple_axis_order_is_argument_order(evaluate):
    x = vfor_simple(lambda i, j: 10 * i + j, 2, 3, interface=pi.default)

    assert x.shape == (2, 3)

    [value] = evaluate(x)
    np.testing.assert_array_equal(
        value,
        np.array(
            [
                [0, 1, 2],
                [10, 11, 12],
            ]
        ),
    )


def test_vfor_simple_three_axis_order(evaluate):
    x = vfor_simple(lambda i, j, k: 100 * i + 10 * j + k, 2, 3, 4, interface=pi.default)

    assert x.shape == (2, 3, 4)

    [value] = evaluate(x)

    expected = np.empty((2, 3, 4), dtype=int)

    for i in range(2):
        for j in range(3):
            for k in range(4):
                expected[i, j, k] = 100 * i + 10 * j + k

    np.testing.assert_array_equal(value, expected)


def test_vfor_simple_multiple_outputs(evaluate):
    x, y = vfor_simple(lambda i: (i + 2, i * 3), 4, interface=pi.default)

    assert x.shape == (4,)
    assert y.shape == (4,)

    x_value, y_value = evaluate(x, y)

    np.testing.assert_array_equal(
        x_value,
        np.array([2, 3, 4, 5]),
    )
    np.testing.assert_array_equal(
        y_value,
        np.array([0, 3, 6, 9]),
    )


def test_vfor_simple_nested_array_index_regression(evaluate):
    """Both mapped dimensions must survive index-view elimination."""

    data = pi.constant(
        [
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
        ]
    )

    x = vfor_simple(lambda i, j: data[i, j] * 2, 2, 3, interface=pi.default)

    # This is the critical regression assertion. A result of shape (3,)
    # means the outer i dimension was incorrectly eliminated.
    assert x.shape == (2, 3)

    [value] = evaluate(x)
    np.testing.assert_allclose(
        value,
        np.array(
            [
                [2.2, 4.4, 6.6],
                [8.8, 11.0, 13.2],
            ]
        ),
    )


def test_vfor_simple_three_dimensional_array(evaluate):
    values = np.arange(24).reshape(2, 3, 4)
    data = pi.constant(values)

    x = vfor_simple(lambda i, j, k: data[i, j, k] + 1, 2, 3, 4, interface=pi.default)

    assert x.shape == (2, 3, 4)
    assert not contains_index(x)

    [value] = evaluate(x)
    np.testing.assert_array_equal(
        value,
        values + 1,
    )


def test_vfor_simple_partial_array_index(evaluate):
    data = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    x = vfor_simple(lambda j: pi.sum(data[:, j], 0), 3, interface=pi.default)

    assert x.shape == (3,)

    [value] = evaluate(x)
    np.testing.assert_allclose(
        value,
        np.array([5.0, 7.0, 9.0]),
    )


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
def test_all_three_dimensional_axis_permutations(evaluate, axes, index_remains):
    """Every permutation is correct even when the conservative optimization declines it."""

    values = np.arange(30).reshape(2, 3, 5)
    x = pi.constant(values)

    def fun(i, j, k):
        coordinates = (i, j, k)
        indices = tuple(coordinates[axes.index(axis)] for axis in range(3))
        return pi.exp(x[indices])

    y = vfor_simple(fun, *(values.shape[axis] for axis in axes), interface=pi.default)

    assert y.shape == tuple(values.shape[axis] for axis in axes)
    assert contains_index(y) is index_remains

    [actual] = evaluate(y)
    expected = np.exp(np.transpose(values, axes))
    np.testing.assert_allclose(actual, expected)


# def test_raw_index_simple():
#     x = vfor(lambda i: i, i=3)
#     assert x.op == pi.Constant([0, 1, 2])
