# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Check the interface base class and the return types of standard syntax."""

from typing import Any, TYPE_CHECKING, assert_type

from pangolin import functional, ir
from pangolin.interface import Interface


def test_interface_inherits_rv_interface():
    pi = Interface()
    interface: ir.RVInterface = pi
    value = ir.RV(ir.Constant(1), interface=pi)

    assert value.interface is interface
    assert isinstance(pi, ir.RVInterface)


def test_keyword_configuration_is_typed():
    pi = Interface(broadcasting="numpy")
    assert_type(pi, Interface)
    assert_type(pi.normal(0, 1), ir.RV)


def test_standard_operations_have_rv_return_types():
    pi = Interface()
    x = pi.constant([[1, 2], [3, 4]])
    assert_type(x.T, ir.RV)
    scalar = pi.constant(1)
    assert_type(scalar + 1, ir.RV)


def test_interface_is_accepted_by_functional_combinators():
    pi = Interface()
    bound = functional.bind_interface(pi, lambda value: value + 1)
    assert bound(ir.RV(ir.Constant(0))).interface is pi
    assert functional.vfor(lambda i: i + 1, interface=pi, i=2).interface is pi
    result = functional.sfor(
        lambda previous, t: previous + t, init=pi.constant(0), interface=pi, t=2
    )
    assert result.interface is pi


if TYPE_CHECKING:

    Interface("numpy")  # type: ignore
    Interface(broadcasting="invalid")  # type: ignore
    Interface(broadcasting=None)  # type: ignore
    Interface(config=None)  # type: ignore
    Interface().config  # type: ignore

    class WrongTranspose(ir.RVInterface):
        def transpose(self, x: ir.RV, /) -> str:  # type: ignore
            return "not an RV"

    class WrongAddition(ir.RVInterface):
        def add(self, x: Any, y: Any, /) -> int:  # type: ignore
            return 1
