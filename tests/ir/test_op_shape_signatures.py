import inspect

from pangolin.ir import Constant, Identity, Matmul, Normal, VMap


def parameter_kinds(callable_):
    return {
        name: parameter.kind
        for name, parameter in inspect.signature(callable_).parameters.items()
    }


def test_zero_arity_op_exposes_concrete_shape_signature():
    assert parameter_kinds(Constant(1).get_shape) == {}


def test_unary_op_exposes_concrete_shape_signature():
    assert parameter_kinds(Identity().get_shape) == {
        "par_shape": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }


def test_binary_op_exposes_concrete_shape_signature():
    assert parameter_kinds(Matmul().get_shape) == {
        "a_shape": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "b_shape": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }


def test_variadic_op_exposes_concrete_shape_signature():
    op = VMap(Normal(), (0, 0))

    assert parameter_kinds(op.get_shape) == {
        "parents_shapes": inspect.Parameter.VAR_POSITIONAL,
    }


def test_concrete_get_shape_exposes_concrete_documentation():
    doc = inspect.getdoc(Matmul.get_shape)
    assert doc is not None
    assert "a_shape" in doc
    assert "b_shape" in doc
    assert "Returns" in doc
