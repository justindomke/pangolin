from typing import Any

import pytest

from pangolin import functional, ir
from pangolin.functional import tforms
from pangolin.functional.transforming import Transform


def normal(mean: ir.RV, scale: ir.RV) -> ir.RV:
    return new_like(mean, ir.Normal(), mean, scale)


def new_like(prototype: ir.RV, op: ir.Op, *parents: ir.RV) -> ir.RV:
    return ir.RV(op, *parents, interface=prototype.interface)


def test_public_api_contains_the_standard_transform_catalog():
    assert functional.tforms is tforms
    assert tforms.__all__ == [
        "exp",
        "log",
        "logit",
        "inv_logit",
        "stick_breaking",
        "softmax_centered",
        "softmax_centered_ilr",
        "scaled_logit",
        "scale_by",
        "shift_by",
        "cholesky",
        "fill_tril",
        "extract_tril",
        "exp_diagonal",
        "log_diagonal",
        "unconstrain_spd",
    ]
    assert all(isinstance(getattr(tforms, name), Transform) for name in tforms.__all__[:4])


@pytest.mark.parametrize(
    ("transform", "forward_type", "inverse_type"),
    [
        (tforms.exp, ir.Exp, ir.Log),
        (tforms.log, ir.Log, ir.Exp),
        (tforms.logit, ir.Logit, ir.InvLogit),
        (tforms.inv_logit, ir.InvLogit, ir.Logit),
    ],
)
def test_scalar_transforms_have_the_expected_directions(transform, forward_type, inverse_type):
    x = ir.RV(ir.Constant(0.25))
    y = transform.forward(x)
    reconstructed = transform.inverse(y)

    assert isinstance(y.op, forward_type)
    assert isinstance(reconstructed.op, inverse_type)
    assert transform.log_det_jac(x, y).shape == ()


def test_scalar_transforms_construct_transformed_distributions():
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    result = tforms.exp(normal)(mean, scale)

    assert isinstance(result.op, ir.Bijected)
    assert result.op.base_op == ir.Normal()
    assert result.op.bijector.forward == ir.Exp()
    assert result.parents == (mean, scale)


def test_parameterized_transforms_capture_ir_rvs_without_coercion():
    factor = ir.RV(ir.Constant(2))
    offset = ir.RV(ir.Constant(3))
    lower = ir.RV(ir.Constant(1))
    upper = ir.RV(ir.Constant(5))
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    scaled = tforms.scale_by(factor)(normal)(mean, scale)
    shifted = tforms.shift_by(offset)(normal)(mean, scale)

    def uniform(low: ir.RV, high: ir.RV) -> ir.RV:
        return ir.RV(ir.Uniform(), low, high)

    unconstrained = tforms.scaled_logit(lower, upper)(uniform)(lower, upper)

    assert scaled.parents == (factor, mean, scale)
    assert shifted.parents == (offset, mean, scale)
    assert unconstrained.parents == (lower, upper, lower, upper)
    assert isinstance(scaled.op, ir.Bijected)
    assert isinstance(shifted.op, ir.Bijected)
    assert isinstance(unconstrained.op, ir.Bijected)
    assert scaled.op.bijector.n_biject_params == 1
    assert shifted.op.bijector.n_biject_params == 1
    assert unconstrained.op.bijector.n_biject_params == 2


def test_shift_by_is_elementwise_but_does_not_broadcast():
    offset = ir.RV(ir.Constant([1, 2]))
    x = ir.RV(ir.Constant([3, 4]))

    result = tforms.shift_by(offset).forward(x)

    assert result.op == ir.VMap(ir.Add(), (0, 0), 2)
    assert result.shape == (2,)
    with pytest.raises(ValueError, match="equal shapes"):
        tforms.shift_by(offset).forward(ir.RV(ir.Constant(1)))


def test_parameterized_transforms_require_ir_rvs_and_valid_shapes():
    not_an_rv: Any = 2
    with pytest.raises(TypeError, match="factor must be an RV"):
        tforms.scale_by(not_an_rv)
    with pytest.raises(TypeError, match="offset must be an RV"):
        tforms.shift_by(not_an_rv)
    with pytest.raises(TypeError, match="bounds must be RVs"):
        tforms.scaled_logit(ir.RV(ir.Constant(0)), not_an_rv)
    with pytest.raises(ValueError, match="factor must be scalar"):
        tforms.scale_by(ir.RV(ir.Constant([1, 2])))
    with pytest.raises(ValueError, match="bounds must be scalar"):
        tforms.scaled_logit(ir.RV(ir.Constant([0, 1])), ir.RV(ir.Constant(2)))


def test_matrix_transform_callbacks_have_the_expected_shapes():
    matrix = ir.RV(ir.Constant([[4.0, 0.0], [0.0, 9.0]]))
    vector = ir.RV(ir.Constant([1.0, 2.0, 3.0]))

    factor = tforms.cholesky.forward(matrix)
    reconstructed_matrix = tforms.cholesky.inverse(factor)
    filled = tforms.fill_tril.forward(vector)
    extracted = tforms.fill_tril.inverse(filled)
    exponentiated = tforms.exp_diagonal.forward(filled)
    logged = tforms.exp_diagonal.inverse(exponentiated)

    assert factor.shape == reconstructed_matrix.shape == (2, 2)
    assert filled.shape == exponentiated.shape == logged.shape == (2, 2)
    assert extracted.shape == (3,)
    assert tforms.cholesky.log_det_jac(matrix, factor).shape == ()
    assert tforms.fill_tril.log_det_jac(vector, filled).shape == ()
    assert tforms.exp_diagonal.log_det_jac(filled, exponentiated).shape == ()


def test_unconstrain_spd_maps_matrices_to_vectors_and_back():
    matrix = ir.RV(ir.Constant([[4.0, 0.0], [0.0, 9.0]]))

    vector = tforms.unconstrain_spd.forward(matrix)
    reconstructed = tforms.unconstrain_spd.inverse(vector)
    log_det = tforms.unconstrain_spd.log_det_jac(matrix, vector)

    assert vector.shape == (3,)
    assert reconstructed.shape == (2, 2)
    assert log_det.shape == ()


def test_unconstrain_spd_constructs_a_transformed_wishart_distribution():
    degrees = ir.RV(ir.Constant(4))
    scale = ir.RV(ir.Constant([[1.0, 0.0], [0.0, 1.0]]))

    def wishart(nu: ir.RV, matrix: ir.RV) -> ir.RV:
        return ir.RV(ir.Wishart(), nu, matrix)

    result = tforms.unconstrain_spd(wishart)(degrees, scale)

    assert isinstance(result.op, ir.Bijected)
    assert result.op.base_op == ir.Wishart()
    assert result.shape == (3,)
    assert result.parents == (degrees, scale)


def test_softmax_centered_ilr_maps_a_simplex_to_one_fewer_coordinates():
    simplex = ir.RV(ir.Constant([0.2, 0.3, 0.5]))

    coordinates = tforms.softmax_centered_ilr.forward(simplex)
    reconstructed = tforms.softmax_centered_ilr.inverse(coordinates)
    log_det = tforms.softmax_centered_ilr.log_det_jac(simplex, coordinates)

    assert coordinates.shape == (2,)
    assert reconstructed.shape == (3,)
    assert reconstructed.op == ir.Softmax()
    assert log_det.shape == ()


@pytest.mark.parametrize("transform", [tforms.stick_breaking, tforms.softmax_centered])
def test_alternative_simplex_transforms_map_to_one_fewer_coordinates(transform):
    simplex = ir.RV(ir.Constant([0.2, 0.3, 0.5]))

    coordinates = transform.forward(simplex)
    reconstructed = transform.inverse(coordinates)
    log_det = transform.log_det_jac(simplex, coordinates)

    assert coordinates.shape == (2,)
    assert reconstructed.shape == (3,)
    assert log_det.shape == ()


@pytest.mark.parametrize("transform", [tforms.stick_breaking, tforms.softmax_centered])
def test_alternative_simplex_transforms_construct_transformed_dirichlet(transform):
    concentration = ir.RV(ir.Constant([1.0, 2.0, 3.0]))

    def dirichlet(alpha: ir.RV) -> ir.RV:
        return ir.RV(ir.Dirichlet(), alpha)

    result = transform(dirichlet)(concentration)

    assert isinstance(result.op, ir.Bijected)
    assert result.op.base_op == ir.Dirichlet()
    assert result.shape == (2,)
    assert result.parents == (concentration,)


def test_softmax_centered_ilr_constructs_a_transformed_dirichlet_distribution():
    concentration = ir.RV(ir.Constant([1.0, 2.0, 3.0]))

    def dirichlet(alpha: ir.RV) -> ir.RV:
        return ir.RV(ir.Dirichlet(), alpha)

    result = tforms.softmax_centered_ilr(dirichlet)(concentration)

    assert isinstance(result.op, ir.Bijected)
    assert result.op.base_op == ir.Dirichlet()
    assert result.shape == (2,)
    assert result.parents == (concentration,)


def test_softmax_centered_ilr_validates_shapes():
    with pytest.raises(ValueError, match="requires a vector"):
        tforms.softmax_centered_ilr.forward(ir.RV(ir.Constant([[0.5, 0.5]])))
    with pytest.raises(ValueError, match="at least two simplex components"):
        tforms.softmax_centered_ilr.forward(ir.RV(ir.Constant([1.0])))
    with pytest.raises(ValueError, match="output shape"):
        tforms.softmax_centered_ilr.log_det_jac(
            ir.RV(ir.Constant([0.2, 0.3, 0.5])),
            ir.RV(ir.Constant([0.0, 0.0, 0.0])),
        )


@pytest.mark.parametrize("transform", [tforms.stick_breaking, tforms.softmax_centered])
def test_alternative_simplex_transforms_validate_shapes(transform):
    with pytest.raises(ValueError, match="requires a vector"):
        transform.forward(ir.RV(ir.Constant([[0.5, 0.5]])))
    with pytest.raises(ValueError, match="at least two simplex components"):
        transform.forward(ir.RV(ir.Constant([1.0])))
    with pytest.raises(ValueError, match="output shape"):
        transform.log_det_jac(
            ir.RV(ir.Constant([0.2, 0.3, 0.5])),
            ir.RV(ir.Constant([0.0, 0.0, 0.0])),
        )


def test_matrix_transforms_validate_shapes():
    with pytest.raises(ValueError, match="vector"):
        tforms.fill_tril.forward(ir.RV(ir.Constant([[1, 2], [3, 4]])))
    with pytest.raises(ValueError, match="triangular"):
        tforms.fill_tril.forward(ir.RV(ir.Constant([1, 2])))
    with pytest.raises(ValueError, match="square matrix"):
        tforms.extract_tril.forward(ir.RV(ir.Constant([[1, 2, 3], [4, 5, 6]])))


def test_transform_catalog_preserves_interfaces():
    interface = ir.RVInterface()
    mean = ir.RV(ir.Constant(0), interface=interface)
    scale = ir.RV(ir.Constant(1), interface=interface)
    vector = ir.RV(ir.Constant([1.0, 2.0, 3.0]), interface=interface)

    scalar_result = tforms.exp(normal)(mean, scale)
    matrix = tforms.fill_tril.forward(vector)
    vector_result = tforms.extract_tril.forward(matrix)

    assert scalar_result.interface is interface
    assert matrix.interface is interface
    assert vector_result.interface is interface
