import doctest
import inspect
import operator

import pytest

from pangolin import functional, ir
import pangolin.interface as interface
from pangolin.interface import Interface

scalar_op_classes = tuple(
    op_class for op_class in ir.ScalarOp.__subclasses__() if op_class.__module__ == ir.__name__
)


def scalar_arguments(op_class: type[ir.ScalarOp]) -> list[float]:
    descriptions = interface.scalar_parent_descriptions(op_class)
    return [0.1 * 2**position for position in range(len(descriptions))]


def test_scalar_decorator_preserves_the_original_function():
    def normal(self, mu: interface.RVLike, sigma: interface.RVLike) -> ir.RV:
        return self.normal(mu, sigma)

    annotations = normal.__annotations__.copy()
    assert interface.scalar(normal) is normal
    assert normal.__annotations__ == annotations


def test_scalar_methods_are_not_installed_on_instances():
    pi = Interface()
    for op_class in scalar_op_classes:
        name = str(op_class())
        assert name not in vars(pi)
        assert getattr(pi, name).__func__ is getattr(Interface, name)


def test_scalar_declaration_rejects_different_named_parents():
    def normal(self, location: interface.RVLike, scale: interface.RVLike): ...

    with pytest.raises(
        TypeError,
        match=r"Normal declares parent names \('mu', 'sigma'\), got \('location', 'scale'\)",
    ):
        interface.scalar(normal)


def test_scalar_declaration_rejects_wrong_arity():
    def add(self, a: interface.RVLike): ...

    with pytest.raises(TypeError, match="Add expects 2 parents, got 1"):
        interface.scalar(add)


def test_count_only_scalar_op_accepts_declared_parameter_names():
    descriptions = interface.scalar_parent_descriptions(ir.Add, ("left", "right"))

    assert descriptions == {"left": "parent 0", "right": "parent 1"}


def test_class_declaration_has_precise_signature_and_generated_docs():
    signature = inspect.signature(Interface.normal, eval_str=True)

    assert tuple(signature.parameters) == (
        "self",
        "mu",
        "sigma",
    )
    assert signature.return_annotation is ir.RV
    assert "mu\n    location / mean." in (Interface.normal.__doc__ or "")


@pytest.mark.parametrize("op_class", scalar_op_classes)
def test_every_scalar_op_has_an_interface_function(op_class):
    pi = Interface()
    function = getattr(pi, str(op_class()))

    result = function(*scalar_arguments(op_class))

    assert type(result) is ir.RV
    assert result.op == op_class()
    assert result.interface is pi
    assert all(parent.interface is pi for parent in result.parents)


@pytest.mark.parametrize("op_class", scalar_op_classes)
def test_scalar_methods_and_exports_preserve_exact_parents(op_class):
    pi = Interface()
    foreign = Interface(broadcasting="off")
    parents = tuple(foreign.constant(value) for value in scalar_arguments(op_class))
    for owner, function in (
        (pi, getattr(pi, str(op_class()))),
        (interface.default, getattr(interface, str(op_class()))),
    ):
        result = function(*parents)
        assert result.op == op_class()
        assert result.parents == parents
        assert result.interface is owner


@pytest.mark.parametrize("op_class", scalar_op_classes)
def test_every_scalar_function_has_ir_parameter_names(op_class):
    function = getattr(Interface(), str(op_class()))
    expected_names = tuple(interface.scalar_parent_descriptions(op_class))

    assert tuple(inspect.signature(function).parameters) == expected_names


@pytest.mark.parametrize("op_class", scalar_op_classes)
def test_every_scalar_function_inherits_ir_documentation(op_class):
    function = getattr(Interface(), str(op_class()))
    doc = function.__doc__ or ""

    assert (op_class.__doc__ or "").partition("\n")[0] not in doc
    for name, description in interface.scalar_parent_descriptions(op_class).items():
        assert f"{name}\n    {description}." in doc
    for note in op_class._notes:
        assert note in doc


@pytest.mark.parametrize("op_class", scalar_op_classes)
def test_default_broadcasting_documentation_is_distinct_from_methods(op_class):
    name = str(op_class())
    default_doc = getattr(interface, name).__doc__
    method_doc = getattr(Interface(), name).__doc__
    assert "This function belongs to the default interface." in default_doc
    assert (
        "Thus, all arguments must either be scalar or mutually broadcastable using "
        ":ref:`simple broadcasting <interface-broadcasting>`, " in default_doc
    )
    assert "unless changed within an :func:`~pangolin.interface.overload` block." in default_doc
    assert "the broadcasting policy selected when constructing the interface" not in default_doc
    assert "the broadcasting policy selected when constructing the interface" in method_doc
    assert "This function belongs to the default interface." not in method_doc


@pytest.mark.parametrize("op_class", scalar_op_classes)
def test_scalar_wikipedia_links_are_in_the_opening_sentence(op_class):
    name = str(op_class())
    for function in (getattr(interface, name), getattr(Interface(), name)):
        summary, see_also = function.__doc__.split("See Also")
        assert f":class:`pangolin.ir.{op_class.__name__}`" in see_also
        assert "wikipedia" not in see_also.lower()
        if op_class.random and op_class._wikipedia is not None:
            assert "(`Wikipedia <https://en.wikipedia.org/wiki/" in summary.splitlines()[0]
        else:
            assert "wikipedia" not in summary.splitlines()[0].lower()


@pytest.mark.parametrize("op_class", scalar_op_classes)
@pytest.mark.parametrize("module_example", [False, True])
def test_every_generated_scalar_example_runs(op_class, module_example):
    owner = interface if module_example else Interface()
    function = getattr(owner, str(op_class()))
    assert f">>> y = pi.{op_class()}([0.1, 0.1]" in function.__doc__
    assert ">>> x.shape\n()" in function.__doc__
    assert ">>> y.shape\n(2,)" in function.__doc__
    test = doctest.DocTestParser().get_doctest(
        function.__doc__ or "",
        interface.__dict__,
        f"Interface.{function.__name__}",
        None,
        0,
    )

    doctest.DebugRunner().run(test)


@pytest.mark.parametrize(
    ("operation", "op_class"),
    [
        (operator.add, ir.Add),
        (operator.sub, ir.Sub),
        (operator.mul, ir.Mul),
        (operator.truediv, ir.Div),
        (operator.pow, ir.Pow),
    ],
)
def test_infix_binary_operations(operation, op_class):
    pi = Interface()
    left = pi.constant(2)
    right = pi.constant(3)

    result = operation(left, right)

    assert result.op == op_class()
    assert result.parents == (left, right)
    assert result.interface is pi


@pytest.mark.parametrize(
    ("operation", "op_class"),
    [
        (operator.add, ir.Add),
        (operator.sub, ir.Sub),
        (operator.mul, ir.Mul),
        (operator.truediv, ir.Div),
        (operator.pow, ir.Pow),
    ],
)
def test_reverse_infix_binary_operations(operation, op_class):
    pi = Interface()
    right = pi.constant(3)

    result = operation(2, right)

    assert result.op == op_class()
    assert result.parents[0].op == ir.Constant(2)
    assert result.parents[1] is right
    assert result.interface is pi


def test_negation_uses_multiplication():
    pi = Interface()
    value = pi.constant(3)

    result = -value

    assert result.op == ir.Mul()
    assert result.parents[0] is value
    assert result.parents[1].op == ir.Constant(-1)
    assert result.interface is pi


def test_builtin_abs_uses_abs_op():
    pi = Interface()
    value = pi.constant(-3)

    result = abs(value)

    assert result.op == ir.Abs()
    assert result.parents == (value,)
    assert result.interface is pi


def test_scalar_distribution_accepts_named_arguments():
    pi = Interface()

    result = pi.student_t(sigma=3, nu=1, mu=2)

    assert result.op == ir.StudentT()
    assert tuple(parent.op for parent in result.parents) == (
        ir.Constant(1),
        ir.Constant(2),
        ir.Constant(3),
    )


def test_scalar_function_accepts_foreign_rv():
    pi = Interface()
    foreign = Interface(broadcasting="rank")
    value = foreign.constant(1)

    result = pi.exp(value)

    assert result.parents == (value,)
    assert result.interface is pi


def test_scalar_functions_compose_in_ordinary_function():
    pi = Interface()

    def calculation(x: ir.RV) -> ir.RV:
        return pi.exp(x) + pi.sin(x)

    value = pi.constant(1)
    result = calculation(value)

    assert result.op == ir.Add()
    assert result.parents[0].op == ir.Exp()
    assert result.parents[1].op == ir.Sin()
    assert result.interface is pi


def test_scalar_distribution_composes_with_vmap():
    pi = Interface()
    means = pi.constant([1, 2, 3])
    scale = pi.constant(1)

    result = functional.vmap(pi.normal, in_axes=(0, None))(means, scale)

    assert result.op == ir.VMap(ir.Normal(), (0, None), 3)
    assert result.parents == (means, scale)
    assert result.interface is pi
