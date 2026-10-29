import doctest
import inspect
from typing import assert_type

import pytest

from pangolin import functional, ir
import pangolin.interface as interface
from pangolin.interface import Config, Interface


def test_interface_constructs_its_config():
    assert Interface(broadcasting="rank")._effective_config == Config("rank")


@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
def test_constructor_accepts_broadcasting_modes(mode):
    assert Interface(broadcasting=mode)._effective_config.broadcasting == mode


@pytest.mark.parametrize("value,error", [("invalid", ValueError), (None, TypeError), (1, TypeError)])
def test_constructor_rejects_invalid_broadcasting(value, error):
    with pytest.raises(error):
        Interface(broadcasting=value)


@pytest.mark.parametrize("args", [(Config(),), ("numpy",)])
def test_constructor_rejects_positional_configuration(args):
    with pytest.raises(TypeError):
        Interface(*args)


@pytest.mark.parametrize("kwargs", [{"config": Config()}])
def test_constructor_rejects_config_keyword(kwargs):
    with pytest.raises(TypeError):
        Interface(**kwargs)


def test_configuration_is_not_public_api():
    assert "Config" not in interface.__all__
    assert not hasattr(Interface(), "config")


def test_interfaces_have_stable_identity():
    first = Interface()
    second = Interface()

    assert first is not second
    assert first != second


def test_interface_repr_is_not_no_interface():
    assert repr(Interface(broadcasting="rank")) == "Interface(broadcasting='rank')"


def test_constant_returns_rv():
    result = Interface().constant(1)

    assert_type(result, ir.RV)
    assert type(result) is ir.RV


def test_constant_has_constant_op():
    result = Interface().constant(1)

    assert isinstance(result.op, ir.Constant)


def test_constant_preserves_shape():
    result = Interface().constant([[1, 2], [3, 4]])

    assert result.shape == (2, 2)


def test_constant_is_owned_by_calling_interface():
    pi = Interface()

    assert pi.constant(1).interface is pi


def test_constants_from_different_interfaces_have_different_owners():
    first = Interface(broadcasting="simple")
    second = Interface(broadcasting="rank")

    assert first.constant(1).interface is first
    assert second.constant(1).interface is second


def test_constant_propagates_invalid_value_error():
    with pytest.raises(ValueError):
        Interface().constant([])


def test_coerce_preserves_existing_rv():
    pi = Interface()
    value = ir.RV(ir.Constant(1))

    assert pi.coerce(value) is value


def test_coerce_creates_owned_constant():
    pi = Interface()
    value = pi.coerce(1)

    assert isinstance(value.op, ir.Constant)
    assert value.interface is pi


def test_add_builds_strict_scalar_addition():
    pi = Interface()
    left = pi.constant(1)
    right = pi.constant(2)

    result = pi.add(left, right)

    assert result.op == ir.Add()
    assert result.parents == (left, right)
    assert result.interface is pi


def test_add_has_generated_signature():
    pi = Interface()

    assert tuple(inspect.signature(pi.add).parameters) == ("a", "b")
    assert inspect.signature(pi.add, eval_str=True).return_annotation is ir.RV


def test_add_has_generated_documentation():
    doc = Interface().add.__doc__

    assert doc is not None
    assert "Create an RV by applying `pangolin.ir.Add` to its parents." in doc
    assert "Scalar `Add` operation." not in doc
    assert "the broadcasting policy selected when constructing the interface" in doc
    assert "a\n    parent 0." in doc
    assert "b\n    parent 1." in doc
    assert "one or more nested `pangolin.ir.VMap`" in doc
    assert "See Also\n--------\n:class:`pangolin.ir.Add`" in doc


def test_generated_add_documentation_example_runs():
    add = Interface().add
    test = doctest.DocTestParser().get_doctest(
        add.__doc__ or "",
        interface.__dict__,
        "Interface.add",
        None,
        0,
    )

    doctest.DebugRunner().run(test)


def test_generated_add_accepts_keywords_in_parameter_order():
    pi = Interface()

    result = pi.add(b=2, a=1)

    assert result.parents[0].op == ir.Constant(1)
    assert result.parents[1].op == ir.Constant(2)


def test_scalar_method_inherits_named_parent_descriptions():
    pi = Interface()
    normal = pi.normal

    assert tuple(inspect.signature(normal).parameters) == ("mu", "sigma")
    assert "mu\n    location / mean." in normal.__doc__
    assert "sigma\n    scale / standard deviation." in normal.__doc__


def test_scalar_method_inherits_distribution_wikipedia_link():
    normal = Interface().normal

    assert (
        "(`Wikipedia <https://en.wikipedia.org/wiki/Normal_distribution>`__)"
        in normal.__doc__.splitlines()[0]
    )


def test_scalar_method_builds_its_op():
    pi = Interface()
    normal = pi.normal

    result = normal(sigma=2, mu=1)

    assert result.op == ir.Normal()
    assert result.parents[0].op == ir.Constant(1)
    assert result.parents[1].op == ir.Constant(2)
    assert result.interface is pi


def test_infix_addition_delegates_to_interface():
    pi = Interface()
    left = pi.constant(1)
    right = pi.constant(2)

    result = left + right

    assert result.op == ir.Add()
    assert result.parents == (left, right)
    assert result.interface is pi


def test_add_coerces_python_values():
    pi = Interface()

    result = pi.add(1, 2)

    assert result.op == ir.Add()
    assert all(isinstance(parent.op, ir.Constant) for parent in result.parents)
    assert all(parent.interface is pi for parent in result.parents)


def test_infix_addition_coerces_python_value():
    pi = Interface()
    value = pi.constant(1)

    result = value + 2

    assert result.parents[0] is value
    assert result.parents[1].interface is pi


def test_reverse_infix_addition_coerces_python_value():
    pi = Interface()
    value = pi.constant(1)

    result = 2 + value

    assert result.parents[0].interface is pi
    assert result.parents[1] is value


def test_add_rejects_array_shapes_without_broadcasting():
    pi = Interface(broadcasting="off")

    with pytest.raises(ValueError):
        pi.add([1, 2], [3, 4])


def test_explicit_add_accepts_foreign_interface_rv():
    pi = Interface()
    foreign = Interface(broadcasting="rank")
    value = foreign.constant(1)

    result = pi.add(value, 2)

    assert result.parents[0] is value
    assert result.parents[0].interface is foreign
    assert result.parents[1].interface is pi
    assert result.interface is pi


def test_explicit_add_accepts_two_foreign_interface_rvs():
    pi = Interface()
    foreign = Interface(broadcasting="rank")
    left = foreign.constant(1)
    right = foreign.constant(2)

    result = pi.add(left, right)

    assert result.parents == (left, right)
    assert result.interface is pi


def test_explicit_add_preserves_repeated_input_identity():
    pi = Interface()
    value = pi.constant(1)

    result = pi.add(value, value)

    assert result.parents == (value, value)


def test_python_function_composes_infix_addition():
    pi = Interface()

    def add_one(x: ir.RV) -> ir.RV:
        return x + 1

    value = pi.constant(2)
    result = add_one(value)

    assert result.parents[0] is value
    assert result.interface is pi


def test_python_function_preserves_captured_rv():
    pi = Interface()
    foreign = Interface(broadcasting="rank")
    captured = foreign.constant(3)

    def add_captured(x: ir.RV) -> ir.RV:
        return x + captured

    value = pi.constant(2)
    result = add_captured(value)

    assert result.parents == (value, captured)
    assert result.interface is pi
    assert result.parents[1].interface is foreign


def test_vmap_composes_with_infix_addition():
    pi = Interface()
    values = pi.constant([1, 2, 3])

    mapped_add_one = functional.vmap(lambda x: x + 1)
    result = mapped_add_one(values)

    assert result.op == ir.VMap(ir.Add(), (0, None), 3)
    assert result.parents[0] is values
    assert result.interface is pi


def test_vmap_preserves_captured_rv():
    pi = Interface()
    foreign = Interface(broadcasting="rank")
    values = pi.constant([1, 2, 3])
    captured = foreign.constant(4)

    result = functional.vmap(lambda x: x + captured)(values)

    assert result.op == ir.VMap(ir.Add(), (0, None), 3)
    assert result.parents == (values, captured)
    assert result.interface is pi
    assert result.parents[1].interface is foreign
