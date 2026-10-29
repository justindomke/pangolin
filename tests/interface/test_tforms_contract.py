from pangolin import interface as pi, ir
from pangolin.ir import RV


import numpy as np
import pytest

from pangolin.interface import Transform
from pangolin.interface import constant
from pangolin.interface import multi_normal
from pangolin.interface import normal
from pangolin.interface import tforms
from pangolin.interface import uniform


def zero_log_det(x: RV, y: RV) -> RV:
    return constant(0.0)


def test_scale_by_creates_a_parameterized_scalar_bijector():
    factor = constant(2)

    result = tforms.scale_by(factor)(normal)(0, 1)

    assert result.parents[0] is factor
    assert result.op.bijector == ir.Bijector(
        ir.Mul(),
        ir.Div(),
        ir.Composite(
            3,
            (ir.Abs(), ir.Log()),
            ((2,), (3,)),
        ),
        n_biject_params=1,
        sign=ir.Composite(1, (ir.Abs(), ir.Div()), ((0,), (0, 1))),
    )


def test_scale_by_rejects_a_non_scalar_factor():
    with pytest.raises(ValueError, match="factor must be scalar"):
        tforms.scale_by([1, 2])


def test_scale_by_rejects_a_non_scalar_value():
    with pytest.raises(ValueError, match="supports only scalar values"):
        tforms.scale_by(2)(multi_normal)(np.zeros(2), np.eye(2))


def test_shift_by_uses_its_offset_as_one_bijector_parameter():
    offset = constant(2)

    result = tforms.shift_by(offset)(normal)(0, 1)

    assert result.parents[0] is offset
    assert result.op.bijector == ir.Bijector(
        ir.Add(),
        ir.Sub(),
        ir.Composite(3, (ir.Constant(0.0),), ((),)),
        n_biject_params=1,
        sign=ir.Composite(1, (ir.Constant(1),), ((),)),
    )


def test_tforms_module_contains_every_standard_transform_and_factory():
    assert tforms is pi.default.tforms

    transform_names = [
        "exp",
        "log",
        "logit",
        "inv_logit",
        "cholesky",
        "fill_tril",
        "extract_tril",
        "exp_diagonal",
        "log_diagonal",
        "unconstrain_spd",
    ]
    for name in transform_names:
        assert isinstance(getattr(tforms, name), Transform)

    assert isinstance(tforms.scaled_logit(1, 5), Transform)
    assert isinstance(tforms.scale_by(2), Transform)
    assert isinstance(tforms.shift_by(3), Transform)


@pytest.mark.parametrize(
    ("transform", "forward_op", "inverse_op"),
    [
        (tforms.exp, ir.Exp, ir.Log),
        (tforms.log, ir.Log, ir.Exp),
        (tforms.logit, ir.Logit, ir.InvLogit),
        (tforms.inv_logit, ir.InvLogit, ir.Logit),
    ],
)
def test_standard_scalar_transforms_have_the_expected_directions(transform, forward_op, inverse_op):
    x = constant(0.25)
    y = transform.forward(x)
    reconstructed = transform.inverse(y)
    log_det = transform.log_det_jac(x, y)

    assert isinstance(y.op, forward_op)
    assert isinstance(reconstructed.op, inverse_op)
    assert y.shape == reconstructed.shape == log_det.shape == ()


def test_scaled_logit_uses_the_correct_interval_orientation_and_forward_jacobian():
    lower = constant(1.0)
    upper = constant(5.0)

    result = tforms.scaled_logit(lower, upper)(uniform)(lower, upper)

    assert result.parents[:2] == (lower, upper)
    assert result.op.bijector.n_biject_params == 2
    assert result.op.bijector.forward == ir.Composite(
        3,
        (ir.Sub(), ir.Sub(), ir.Div(), ir.Logit()),
        ((0, 1), (2, 1), (3, 4), (5,)),
    )
    assert result.op.bijector.inverse == ir.Composite(
        3,
        (ir.Sub(), ir.InvLogit(), ir.Mul(), ir.Add()),
        ((2, 1), (0,), (3, 4), (1, 5)),
    )
    assert result.op.bijector.log_det_jac == ir.Composite(
        4,
        (
            ir.Sub(),
            ir.Abs(),
            ir.Log(),
            ir.Sub(),
            ir.Abs(),
            ir.Log(),
            ir.Sub(),
            ir.Abs(),
            ir.Log(),
            ir.Sub(),
            ir.Sub(),
        ),
        ((3, 2), (4,), (5,), (0, 2), (7,), (8,), (3, 0), (10,), (11,), (6, 9), (13, 12)),
    )


@pytest.mark.parametrize("bounds", [([1, 2], 5), (1, [5, 6])])
def test_scaled_logit_requires_scalar_bounds(bounds):
    with pytest.raises(ValueError, match="bounds must be scalar"):
        tforms.scaled_logit(*bounds)


def test_cholesky_transform_preserves_matrix_shapes_and_has_a_scalar_jacobian():
    x = constant([[4.0, 1.0], [1.0, 3.0]])

    y = tforms.cholesky.forward(x)
    reconstructed = tforms.cholesky.inverse(y)
    log_det = tforms.cholesky.log_det_jac(x, y)

    assert y.shape == reconstructed.shape == (2, 2)
    assert log_det.shape == ()
    assert isinstance(y.op, ir.Cholesky)
    assert isinstance(reconstructed.op, ir.Matmul)


def test_fill_and_extract_tril_are_opposite_shape_transformations():
    vector = constant([1.0, 2.0, 3.0])

    matrix = tforms.fill_tril.forward(vector)
    reconstructed_vector = tforms.fill_tril.inverse(matrix)
    extracted = tforms.extract_tril.forward(matrix)
    reconstructed_matrix = tforms.extract_tril.inverse(extracted)

    assert matrix.shape == reconstructed_matrix.shape == (2, 2)
    assert reconstructed_vector.shape == extracted.shape == (3,)
    assert tforms.fill_tril.log_det_jac(vector, matrix).shape == ()


def test_exp_and_log_diagonal_preserve_matrix_shapes():
    x = constant([[1.0, 0.0], [2.0, 3.0]])

    y = tforms.exp_diagonal.forward(x)
    reconstructed = tforms.exp_diagonal.inverse(y)
    log_det = tforms.exp_diagonal.log_det_jac(x, y)

    assert y.shape == reconstructed.shape == (2, 2)
    assert log_det.shape == ()
    assert isinstance(log_det.op, ir.Sum)
    assert tforms.log_diagonal.forward(y).shape == (2, 2)


def test_unconstrain_spd_maps_a_matrix_to_a_vector_and_back():
    x = constant([[4.0, 1.0], [1.0, 3.0]])

    y = tforms.unconstrain_spd.forward(x)
    reconstructed = tforms.unconstrain_spd.inverse(y)
    log_det = tforms.unconstrain_spd.log_det_jac(x, y)

    assert y.shape == (3,)
    assert reconstructed.shape == (2, 2)
    assert log_det.shape == ()
