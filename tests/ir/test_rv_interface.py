from __future__ import annotations

import operator

import numpy as np
import pytest

from pangolin import ir


class RecordingInterface(ir.RVInterface):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.result = ir.RV(ir.Constant(99))

    def record(self, name: str, *args: object) -> ir.RV:
        self.calls.append((name, args))
        return self.result

    def neg(self, x: ir.RV, /) -> ir.RV:
        return self.record("neg", x)

    def abs(self, x: ir.RV, /) -> ir.RV:
        return self.record("abs", x)

    def add(self, x: object, y: object, /) -> ir.RV:
        return self.record("add", x, y)

    def sub(self, x: object, y: object, /) -> ir.RV:
        return self.record("sub", x, y)

    def mul(self, x: object, y: object, /) -> ir.RV:
        return self.record("mul", x, y)

    def div(self, x: object, y: object, /) -> ir.RV:
        return self.record("div", x, y)

    def pow(self, x: object, y: object, /) -> ir.RV:
        return self.record("pow", x, y)

    def matmul(self, x: object, y: object, /) -> ir.RV:
        return self.record("matmul", x, y)

    def transpose(self, x: ir.RV, /) -> ir.RV:
        return self.record("transpose", x)

    def getitem(self, x: ir.RV, /, *indices: object) -> ir.RV:
        return self.record("getitem", x, *indices)

    def get_attribute(self, rv: ir.RV, name: str, /) -> object:
        if name == "vindex":
            return RecordingVectorIndexProxy(self, rv)
        return super().get_attribute(rv, name)


class RecordingVectorIndexProxy:
    def __init__(self, interface: RecordingInterface, rv: ir.RV) -> None:
        self.interface = interface
        self.rv = rv

    def __getitem__(self, index: object) -> ir.RV:
        indices = index if isinstance(index, tuple) else (index,)
        return self.interface.record("vindex", self.rv, *indices)


def test_rv_uses_no_interface_by_default():
    x = ir.RV(ir.Constant(1))

    assert x.interface is ir.NO_INTERFACE
    assert repr(ir.NO_INTERFACE) == "NO_INTERFACE"


def test_rv_stores_an_explicit_interface_by_identity():
    interface = RecordingInterface()
    x = ir.RV(ir.Constant(1), interface=interface)

    assert x.interface is interface


def test_rv_interface_is_immutable():
    x = ir.RV(ir.Constant(1), interface=RecordingInterface())

    with pytest.raises(TypeError, match="RVs are immutable"):
        setattr(x, "interface", RecordingInterface())


def test_parent_interfaces_need_not_match():
    left_interface = RecordingInterface()
    right_interface = RecordingInterface()
    result_interface = RecordingInterface()
    left = ir.RV(ir.Constant(1), interface=left_interface)
    right = ir.RV(ir.Constant(2), interface=right_interface)

    result = ir.RV(ir.Add(), left, right, interface=result_interface)

    assert result.parents == (left, right)
    assert result.interface is result_interface


@pytest.mark.parametrize(
    ("operation", "reverse_operation", "name"),
    [
        (operator.add, lambda left, right: right + left, "add"),
        (operator.sub, lambda left, right: right - left, "sub"),
        (operator.mul, lambda left, right: right * left, "mul"),
        (operator.truediv, lambda left, right: right / left, "div"),
        (operator.pow, lambda left, right: right**left, "pow"),
        (operator.matmul, lambda left, right: right @ left, "matmul"),
    ],
)
def test_binary_syntax_delegates_through_the_rv_interface(operation, reverse_operation, name):
    interface = RecordingInterface()
    x = ir.RV(ir.Constant(1), interface=interface)

    assert operation(x, 2) is interface.result
    assert interface.calls[-1] == (name, (x, 2))

    assert reverse_operation(x, 2) is interface.result
    assert interface.calls[-1] == (name, (2, x))


def test_unary_syntax_delegates_through_the_rv_interface():
    interface = RecordingInterface()
    x = ir.RV(ir.Constant(1), interface=interface)

    assert -x is interface.result
    assert interface.calls[-1] == ("neg", (x,))
    assert abs(x) is interface.result
    assert interface.calls[-1] == ("abs", (x,))
    assert x.T is interface.result
    assert interface.calls[-1] == ("transpose", (x,))


def test_indexing_delegates_through_the_rv_interface():
    interface = RecordingInterface()
    x = ir.RV(ir.Constant([[1, 2], [3, 4]]), interface=interface)

    assert x[1, :] is interface.result
    assert interface.calls[-1] == ("getitem", (x, 1, slice(None)))
    interface.calls.clear()
    proxy = x.vindex
    assert not interface.calls
    assert proxy[1, :] is interface.result
    assert interface.calls[-1] == ("vindex", (x, 1, slice(None)))
    assert proxy[...] is interface.result
    assert interface.calls[-1] == ("vindex", (x, Ellipsis))


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: -x,
        lambda x: abs(x),
        lambda x: x + 1,
        lambda x: 1 + x,
        lambda x: x - 1,
        lambda x: 1 - x,
        lambda x: x * 2,
        lambda x: 2 * x,
        lambda x: x / 2,
        lambda x: 2 / x,
        lambda x: x**2,
        lambda x: 2**x,
        lambda x: x @ 2,
        lambda x: 2 @ x,
        lambda x: x.T,
        lambda x: x[0],
    ],
)
def test_no_interface_rejects_syntax(operation):
    x = ir.RV(ir.Constant(1))

    with pytest.raises(NotImplementedError):
        operation(x)


def test_no_interface_rejects_extension_attributes():
    x = ir.RV(ir.Constant(1))
    with pytest.raises(AttributeError):
        _ = x.vindex


def test_truth_testing_and_numpy_conversion_are_always_rejected():
    x = ir.RV(ir.Constant(1), interface=RecordingInterface())

    with pytest.raises(TypeError, match="RVs do not have a truth value"):
        bool(x)
    with pytest.raises(TypeError, match="Cannot convert"):
        np.asarray(x)


def test_interface_is_omitted_from_structural_display_and_rv_equal():
    left = ir.RV(ir.Constant(1), interface=RecordingInterface())
    right = ir.RV(ir.Constant(1), interface=RecordingInterface())

    assert repr(left) == "RV(Constant(1))"
    assert str(left) == "1"
    assert ir.rv_equal(left, right)
