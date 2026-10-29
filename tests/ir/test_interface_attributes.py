from __future__ import annotations

import operator
from functools import partial
from typing import Any

import numpy as np
import pytest

from pangolin import functional, ir


class ExampleInterface(ir.RVInterface):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ir.RV]] = []

    def get_attribute(self, rv: ir.RV, name: str, /) -> Any:
        if name == "label":
            self.calls.append(("label", rv))
            return "example"
        if name == "scaled":
            return partial(self.scaled, rv)
        return super().get_attribute(rv, name)

    def scaled(self, rv: ir.RV, amount: int, *, offset: int = 0) -> ir.RV:
        factor = ir.RV(ir.Constant(amount))
        shift = ir.RV(ir.Constant(offset))
        product = ir.RV(ir.Mul(), rv, factor, interface=self)
        return ir.RV(ir.Add(), product, shift, interface=self)

    def add(self, x: Any, y: Any, /) -> ir.RV:
        self.calls.append(("add", x))
        return x

    @property
    def helper(self) -> str:
        raise AssertionError("unexposed property must not be evaluated")


def test_extension_property_and_method_use_original_rv():
    interface = ExampleInterface()
    x = ir.RV(ir.Constant(2), interface=interface)
    assert x.label == "example"
    assert interface.calls == [("label", x)]
    result = x.scaled(3, offset=1)
    assert result.parents[0].parents[0] is x
    assert result.interface is interface
    assert x + 1 is x


def test_unlisted_implemented_property_is_not_evaluated():
    x = ir.RV(ir.Constant(1), interface=ExampleInterface())
    with pytest.raises(AttributeError, match="does not expose 'helper'"):
        _ = x.helper


@pytest.mark.parametrize("name", ["config", "bind", "exposed", "missing", "vindex"])
def test_only_explicit_names_are_available(name):
    x = ir.RV(ir.Constant(1), interface=ExampleInterface())
    assert not hasattr(x, name)


def test_empty_interface_does_not_break_attribute_probes():
    x = ir.RV(ir.Constant(1))
    assert not hasattr(x, "missing")
    assert not hasattr(x, "__jax_array__")
    assert getattr(x, "missing", 17) == 17


@pytest.mark.parametrize(
    "operation", [operator.neg, operator.abs, lambda x: x.T, lambda x: x + 1, lambda x: x[0]]
)
def test_interface_base_provides_rejecting_defaults(operation):
    x = ir.RV(ir.Constant(1), interface=ir.RVInterface())
    with pytest.raises(NotImplementedError):
        operation(x)


def test_no_interface_is_an_instance_of_the_base():
    assert type(ir.NO_INTERFACE) is ir.RVInterface
    with pytest.raises(AttributeError, match="missing"):
        ir.NO_INTERFACE.get_attribute(ir.RV(ir.Constant(1)), "missing")


def test_standard_property_works_without_extension_hook():
    result = ir.RV(ir.Constant(7))

    class TransposeInterface(ir.RVInterface):
        def transpose(self, x: ir.RV, /) -> ir.RV:
            return result

    x = ir.RV(ir.Constant(1), interface=TransposeInterface())
    assert x.T is result


def test_standard_operator_works_without_extension_hook():
    result = ir.RV(ir.Constant(7))

    class AddInterface(ir.RVInterface):
        def add(self, x: Any, y: Any, /) -> ir.RV:
            return result

    x = ir.RV(ir.Constant(1), interface=AddInterface())
    assert x + 1 is result


@pytest.mark.parametrize(
    "name", ["_secret", "__jax_array__", "__array_interface__", "__setstate__"]
)
def test_unknown_private_and_dunder_names_are_not_forwarded(name):
    class ProbeInterface(ir.RVInterface):
        def get_attribute(self, rv, name):
            raise AssertionError("probe reached interface attribute lookup")

    x = ir.RV(ir.Constant(1), interface=ProbeInterface())
    assert not hasattr(x, name)


def test_core_attributes_never_use_the_extension_hook():
    class HostileInterface(ir.RVInterface):
        def get_attribute(self, rv, name):
            raise AssertionError("core attribute reached interface lookup")

    interface = HostileInterface()
    parent = ir.RV(ir.Constant(1))
    x = ir.RV(ir.Exp(), parent, interface=interface)
    y = ir.RV(ir.Exp(), parent, interface=interface)
    assert x.parents == (parent,)
    assert x.op == ir.Exp()
    assert x.shape == ()
    assert x.ndim == 0
    assert x.interface is interface
    assert x == x and x != y
    assert hash(x) == object.__hash__(x)
    with pytest.raises(TypeError, match="immutable"):
        x.parents = ()
    with pytest.raises(TypeError, match="immutable"):
        del x.op
    with pytest.raises(TypeError, match="truth value"):
        bool(x)
    with pytest.raises(TypeError, match="Cannot convert"):
        np.asarray(x)


def test_owned_property_attribute_error_does_not_fall_through(monkeypatch):
    def broken_label(self):
        raise AttributeError("broken core property")

    monkeypatch.setattr(ir.RV, "label", property(broken_label), raising=False)
    interface = ExampleInterface()
    x = ir.RV(ir.Constant(1), interface=interface)
    with pytest.raises(AttributeError):
        _ = x.label
    assert not interface.calls


def test_failed_construction_does_not_recurse_during_attribute_lookup():
    x = object.__new__(ir.RV)
    assert not hasattr(x, "missing")


@pytest.mark.parametrize("combinator", [functional.composite, functional.vmap])
def test_combinators_preserve_custom_conveniences(combinator):
    interface = ExampleInterface()
    x = ir.RV(
        ir.Constant([1, 2]) if combinator is functional.vmap else ir.Constant(1),
        interface=interface,
    )
    result = combinator(lambda value: value.scaled(2))(x)
    assert result.interface is interface
    assert result.label == "example"


def test_scan_preserves_custom_conveniences():
    interface = ExampleInterface()
    start = ir.RV(ir.Constant(1), interface=interface)
    result = functional.scan(lambda previous: previous.scaled(2), 3)(start)
    assert result.interface is interface
    assert result.label == "example"


def test_reflected_operator_fallback_preserves_not_implemented():
    class ForwardInterface(ir.RVInterface):
        def add(self, x, y):
            return NotImplemented

    class Right:
        def __radd__(self, other):
            return "reflected"

    x = ir.RV(ir.Constant(1), interface=ForwardInterface())
    assert operator.add(x, Right()) == "reflected"
