from __future__ import annotations

import inspect
from typing import Any

import pytest

from pangolin import functional, ir
from pangolin.functional.tracing import bind_interface


class AddInterface(ir.RVInterface):
    def add(self, x: Any, y: Any, /) -> ir.RV:
        if not isinstance(x, ir.RV):
            x = ir.RV(ir.Constant(x), interface=self)
        if not isinstance(y, ir.RV):
            y = ir.RV(ir.Constant(y), interface=self)
        return ir.RV(ir.Add(), x, y, interface=self)


def test_bind_interface_is_public():
    assert functional.bind_interface is bind_interface


def test_bind_interface_uses_the_bound_interface_for_syntax():
    interface = AddInterface()
    actual_interface = ir.RVInterface()
    value = ir.RV(ir.Constant(2), interface=actual_interface)

    result = bind_interface(interface, lambda x: x + 1)(value)

    assert result.op == ir.Add()
    assert result.parents[0] is value
    assert result.interface is interface
    assert result.parents[1].interface is interface


def test_bind_interface_preserves_the_wrapped_signature_and_name():
    interface = AddInterface()

    def add_if_enabled(value: ir.RV, amount: int, *, enabled: bool = True) -> ir.RV:
        return value + amount if enabled else value

    bound = bind_interface(interface, add_if_enabled)

    assert inspect.signature(bound) == inspect.signature(add_if_enabled)
    assert bound.__name__ == "add_if_enabled"
    assert bound.__qualname__ == add_if_enabled.__qualname__


def test_bind_interface_passes_non_rv_inputs_through_unchanged():
    interface = AddInterface()
    value = ir.RV(ir.Constant(2))
    configuration = object()
    seen: list[object] = []

    def fun(x: ir.RV, config: object) -> ir.RV:
        seen.append(config)
        return x + 1

    bind_interface(interface, fun)(value, configuration)

    assert seen == [configuration]
    assert seen[0] is configuration


def test_bind_interface_preserves_nested_argument_and_output_pytrees():
    interface = AddInterface()
    left = ir.RV(ir.Constant(2))
    right = ir.RV(ir.Constant(3))

    def fun(values: dict[str, ir.RV]) -> dict[str, list[ir.RV]]:
        total = values["left"] + values["right"]
        return {"results": [total, total]}

    result = bind_interface(interface, fun)({"left": left, "right": right})

    assert result["results"][0] is result["results"][1]
    assert result["results"][0].parents == (left, right)


def test_bind_interface_preserves_repeated_input_identity_during_tracing():
    interface = AddInterface()
    value = ir.RV(ir.Constant(2))

    def require_same(left: ir.RV, right: ir.RV) -> ir.RV:
        assert left is right
        return left + right

    result = bind_interface(interface, require_same)(value, value)

    assert result.parents == (value, value)


def test_bind_interface_returns_an_identity_output_as_the_exact_input():
    interface = AddInterface()
    value = ir.RV(ir.Constant(2))

    result = bind_interface(interface, lambda x: x)(value)

    assert result is value


def test_bind_interface_leaves_captured_rvs_unchanged():
    interface = AddInterface()
    value = ir.RV(ir.Constant(2))
    captured = ir.RV(ir.Constant(3), interface=ir.RVInterface())

    result = bind_interface(interface, lambda x: x + captured)(value)

    assert result.parents == (value, captured)
    assert result.parents[1] is captured


def test_bind_interface_preserves_interfaces_selected_explicitly_by_the_function():
    bound_interface = AddInterface()
    output_interface = ir.RVInterface()
    value = ir.RV(ir.Constant(2))

    def fun(x: ir.RV) -> ir.RV:
        return ir.RV(ir.Exp(), x, interface=output_interface)

    result = bind_interface(bound_interface, fun)(value)

    assert result.interface is output_interface
    assert result.parents == (value,)


def test_bind_interface_omits_generated_nodes_that_do_not_contribute():
    interface = AddInterface()
    value = ir.RV(ir.Constant(2))
    unused: list[ir.RV] = []

    def fun(x: ir.RV) -> ir.RV:
        unused.append(ir.RV(ir.Exp(), x, interface=interface))
        return x + 1

    result = bind_interface(interface, fun)(value)

    assert unused[0] not in result.parents
    assert all(parent is not unused[0] for parent in result.parents)


def test_bind_interface_rejects_a_non_rv_output_leaf():
    interface = AddInterface()
    value = ir.RV(ir.Constant(2))

    with pytest.raises(TypeError, match="trace_call output must be a PyTree"):
        bind_interface(interface, lambda x: [x, None])(value)  # type: ignore


def test_bind_interface_propagates_function_exceptions():
    interface = AddInterface()
    value = ir.RV(ir.Constant(2))

    def fail(x: ir.RV) -> ir.RV:
        raise RuntimeError("failed during tracing")

    with pytest.raises(RuntimeError, match="failed during tracing"):
        bind_interface(interface, fail)(value)
