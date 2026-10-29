import pytest

from pangolin.ir import (
    Bijector,
    Composite,
    Constant,
    DiagMatrix,
    Exp,
    Identity,
    Log,
    Normal,
    Op,
    Bijected,
)


def zero_log_det() -> Composite:
    return Composite(2, (Constant(0),), [[]])


def exp_bijector() -> Bijector:
    return Bijector(Exp(), Log(), Composite(2, (Identity(),), [[0]]))


def parameterized_identity_bijector() -> Bijector:
    identity = Composite(2, (Identity(),), [[0]])
    return Bijector(identity, identity, Composite(3, (Constant(0),), [[]]), n_biject_params=1)


def test_bijector_stores_its_operations_and_parameter_count():
    forward = Exp()
    inverse = Log()
    log_det_jac = Composite(2, (Identity(),), [[0]])

    bijector = Bijector(forward, inverse, log_det_jac, n_biject_params=2)

    assert bijector.forward is forward
    assert bijector.inverse is inverse
    assert bijector.log_det_jac is log_det_jac
    assert bijector.n_biject_params == 2


def test_bijector_is_a_descriptor_not_an_op():
    bijector = exp_bijector()

    assert not isinstance(bijector, Op)
    assert isinstance(Bijected(Normal(), bijector), Op)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("forward", Identity()),
        ("inverse", Identity()),
        ("log_det_jac", zero_log_det()),
        ("n_biject_params", 1),
        ("sign", Constant(1)),
    ],
)
def test_bijector_is_immutable(attribute, value):
    bijector = exp_bijector()

    with pytest.raises(TypeError, match="Bijectors are immutable"):
        setattr(bijector, attribute, value)


@pytest.mark.parametrize(
    ("forward", "inverse", "log_det_jac", "message"),
    [
        (Normal(), Log(), zero_log_det(), "forward"),
        (Exp(), Normal(), zero_log_det(), "inverse"),
        (Exp(), Log(), Normal(), "log_det_jac"),
    ],
)
def test_bijector_rejects_random_operations(forward, inverse, log_det_jac, message):
    with pytest.raises(ValueError, match=rf"{message} op cannot be random"):
        Bijector(forward, inverse, log_det_jac)


@pytest.mark.parametrize(
    ("forward", "inverse", "log_det_jac", "message"),
    [
        ("exp", Log(), zero_log_det(), "forward must be an Op"),
        (Exp(), "log", zero_log_det(), "inverse must be an Op"),
        (Exp(), Log(), 0, "log_det_jac must be an Op"),
    ],
)
def test_bijector_rejects_non_op_operations(forward, inverse, log_det_jac, message):
    with pytest.raises(TypeError, match=message):
        Bijector(forward, inverse, log_det_jac)


@pytest.mark.parametrize("n_biject_params", [True, 1.5, "1", None])
def test_bijector_rejects_non_integer_parameter_count(n_biject_params):
    with pytest.raises(TypeError, match="n_biject_params must be an integer"):
        Bijector(Exp(), Log(), zero_log_det(), n_biject_params)


def test_bijector_rejects_negative_parameter_count():
    with pytest.raises(ValueError, match="n_biject_params must be non-negative"):
        Bijector(Exp(), Log(), zero_log_det(), -1)


def test_bijector_get_shape():
    assert exp_bijector().get_shape(()) == ()


def test_bijector_get_shape_with_parameters():
    assert parameterized_identity_bijector().get_shape((), ()) == ()


def test_bijector_get_shape_accepts_scalar_log_det_for_vector_transform():
    bijector = Bijector(Identity(), Identity(), zero_log_det())

    assert bijector.get_shape((2,)) == (2,)


def test_bijector_get_shape_rejects_non_scalar_log_det():
    vector_log_det = Composite(2, (Identity(),), [[0]])
    bijector = Bijector(Identity(), Identity(), vector_log_det)

    with pytest.raises(ValueError, match=r"log_det_jac must return a scalar, got shape \(2,\)"):
        bijector.get_shape((2,))


@pytest.mark.parametrize("param_shapes", [(), ((), ())])
def test_bijector_get_shape_requires_exact_parameter_count(param_shapes):
    bijector = parameterized_identity_bijector()

    with pytest.raises(ValueError, match="Number of bijection parameters"):
        bijector.get_shape((), *param_shapes)


def test_bijector_get_shape_rejects_incompatible_inverse_shape():
    bijector = Bijector(DiagMatrix(), Identity(), zero_log_det())

    with pytest.raises(ValueError, match="forward and inverse ops have incompatible shapes"):
        bijector.get_shape((2,))


def test_bijector_equality_includes_all_fields():
    bijector = exp_bijector()

    assert bijector == exp_bijector()
    assert bijector != Bijector(Identity(), Log(), Composite(2, (Identity(),), [[0]]))
    assert bijector != Bijector(Exp(), Identity(), Composite(2, (Identity(),), [[0]]))
    assert bijector != Bijector(Exp(), Log(), Composite(2, (Identity(),), [[1]]))
    assert bijector != Bijector(Exp(), Log(), Composite(2, (Identity(),), [[0]]), n_biject_params=1)
    assert bijector != Exp()


def test_equal_bijectors_have_equal_hashes_and_can_be_dictionary_keys():
    bijector = exp_bijector()
    equivalent_bijector = exp_bijector()

    assert hash(bijector) == hash(equivalent_bijector)
    assert {bijector: "value"}[equivalent_bijector] == "value"


def test_bijector_representations():
    bijector = exp_bijector()

    assert repr(bijector) == "Bijector(Exp(), Log(), Composite(2, [Identity()], [[0]]))"
    assert str(bijector) == "bijector(exp, log, composite(2, [identity], [[0]]))"


def test_parameterized_bijector_representations_include_parameter_count():
    bijector = parameterized_identity_bijector()

    assert repr(bijector).endswith(", 1)")
    assert str(bijector).endswith(", 1)")
