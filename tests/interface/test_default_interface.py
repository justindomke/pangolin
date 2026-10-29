import doctest
import inspect
from typing import assert_type

from pangolin import ir
import pangolin.interface as interface


def test_default_is_an_interface_with_default_config():
    assert isinstance(interface.default, interface.Interface)
    assert interface.default._effective_config == interface.Config()


def test_default_function_wraps_the_default_method():
    assert inspect.isfunction(interface.normal)
    assert inspect.unwrap(interface.normal) == interface.default.normal
    assert interface.normal.__qualname__ == "normal"
    assert interface.normal.__module__ == "pangolin.interface"


def test_default_documentation_does_not_change_method_documentation():
    method_doc = interface.Interface.normal.__doc__ or ""
    assert ">>> pi = Interface()" in method_doc
    assert ">>> import pangolin.interface as pi" not in method_doc
    assert interface.default.normal.__doc__ == method_doc
    assert interface.Interface().normal.__doc__ == method_doc


def test_default_function_preserves_its_precise_signature():
    assert tuple(inspect.signature(interface.normal).parameters) == ("mu", "sigma")
    assert inspect.signature(interface.normal, eval_str=True).return_annotation is ir.RV


def test_default_function_documentation_uses_default_module():
    doc = interface.normal.__doc__ or ""

    assert ">>> import pangolin.interface as pi" in doc
    assert ">>> x = pi.normal(0.1, 0.2)" in doc
    test = doctest.DocTestParser().get_doctest(
        doc, interface.__dict__, "pangolin.interface.normal", None, 0
    )
    doctest.DebugRunner().run(test)


def test_default_function_returns_rv_owned_by_default_interface():
    result = interface.normal(0, 1)

    assert_type(result, ir.RV)
    assert result.interface is interface.default


def test_default_infix_syntax_returns_to_default_interface():
    x = interface.constant(1)
    result = x + 2

    assert result.op == ir.Add()
    assert result.interface is interface.default


def test_public_api_contains_every_default_function():
    scalar_names = {
        str(op_class())
        for op_class in ir.ScalarOp.__subclasses__()
        if op_class.__module__ == ir.__name__
    }
    expected = {"default", "constant", "neg", *scalar_names}

    assert expected <= set(interface.__all__)
