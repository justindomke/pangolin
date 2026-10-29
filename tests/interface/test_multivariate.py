from pangolin import interface as pi, ir


from typing import assert_type

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("fun", "args", "op_type", "expected_shape"),
    [
        (pi.matmul, ([1, 2], [3, 4]), ir.Matmul, ()),
        (pi.transpose, ([[1, 2, 3], [4, 5, 6]],), ir.Transpose, (3, 2)),
        (pi.diag, ([[1, 0], [0, 2]],), ir.Diag, (2,)),
        (pi.diag_matrix, ([1, 2, 3],), ir.DiagMatrix, (3, 3)),
        (pi.inv, ([[1, 0], [0, 2]],), ir.Inv, (2, 2)),
        (pi.cholesky, ([[2, 0], [0, 3]],), ir.Cholesky, (2, 2)),
        (pi.softmax, ([1, 2, 3],), ir.Softmax, (3,)),
        (lambda x: pi.sum(x, axis=0), ([[1, 2], [3, 4]],), ir.Sum, (2,)),
        (
            pi.concatenate,
            ([[1, 2]], [[3, 4], [5, 6]]),
            ir.Concatenate,
            (3, 2),
        ),
        (
            pi.multi_normal,
            ([0, 0], [[1, 0], [0, 1]]),
            ir.MultiNormal,
            (2,),
        ),
        (pi.categorical, ([0.2, 0.3, 0.5],), ir.Categorical, ()),
        (pi.multinomial, (10, [0.2, 0.3, 0.5]), ir.Multinomial, (3,)),
        (pi.dirichlet, ([1, 2, 3],), ir.Dirichlet, (3,)),
        (pi.wishart, (5, [[1, 0], [0, 1]]), ir.Wishart, (2, 2)),
    ],
)
def test_array_operations_and_distributions_coerce_parents(fun, args, op_type, expected_shape):
    rv = fun(*args)

    assert isinstance(rv, ir.RV)
    assert type(rv.op) is op_type
    assert rv.shape == expected_shape
    assert all(isinstance(parent.op, ir.Constant) for parent in rv.parents)


def test_array_operation_preserves_existing_rv_parent():
    x = pi.constant([1, 2, 3])

    y = pi.softmax(x)

    assert y.parents == (x,)
    assert y.parents[0] is x


def test_multivariate_distribution_preserves_rvs_and_coerces_other_parents():
    mean = pi.constant([0, 0])

    x = pi.multi_normal(mean, [[1, 0], [0, 1]])

    assert x.parents[0] is mean
    assert isinstance(x.parents[1].op, ir.Constant)


def test_sum_accepts_rvlike_input_and_returns_an_interface_rv():
    x = pi.sum([[1, 2, 3], [4, 5, 6]], axis=-1)

    assert_type(x, ir.RV)
    assert x.shape == (2,)
    assert isinstance(x.parents[0].op, ir.Constant)


def test_sum_delegates_axis_normalization_to_ir_sum():
    x = pi.sum(np.ones((2, 3)), axis=np.int64(-1))  # type: ignore[arg-type]

    assert x.op == ir.Sum(-1)
    assert x.shape == (2,)


@pytest.mark.parametrize("axis", [True, None, 1.5, "1"])
def test_sum_rejects_noninteger_axis(axis):
    with pytest.raises(TypeError, match="axis for Sum must be an integer"):
        pi.sum([1, 2, 3], axis=axis)


def test_sum_rejects_out_of_bounds_axis():
    with pytest.raises(ValueError, match="axis 2 is out of bounds"):
        pi.sum([[1, 2], [3, 4]], axis=2)


def test_concatenate_is_variadic_and_preserves_existing_rv_parents():
    first = pi.constant([[1], [2]])

    result = pi.concatenate(first, [[3, 4], [5, 6]], axis=-1)

    assert_type(result, ir.RV)
    assert result.op == ir.Concatenate(-1)
    assert result.shape == (2, 3)
    assert result.parents[0] is first
    assert isinstance(result.parents[1].op, ir.Constant)


def test_concatenate_performs_no_broadcasting():
    with pytest.raises(ValueError, match="must match except along axis"):
        pi.concatenate(np.ones((2, 1)), np.ones((3, 2)), axis=1)


def test_interface_operations_delegate_shape_validation_to_the_ir():
    with pytest.raises(ValueError, match="transpose only applies to 2d arrays"):
        pi.transpose([1, 2, 3])

    with pytest.raises(ValueError, match="second parameter must be matrix with size matching"):
        pi.multi_normal([0, 0], np.eye(3))


def test_wishart_scale_keyword():
    x = pi.wishart(5, scale=np.eye(2))

    assert_type(x, ir.RV)
    assert x.shape == (2, 2)
