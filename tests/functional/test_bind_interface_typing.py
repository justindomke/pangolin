# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Static typing tests for `functional.bind_interface`."""

from typing import TYPE_CHECKING, assert_type

from pangolin import functional, ir


def test_bind_interface_preserves_the_function_signature_and_output_tree():
    interface = ir.RVInterface()

    def fun(value: ir.RV, amount: int, *, enabled: bool) -> dict[str, ir.RV]:
        return {"result": ir.RV(ir.Exp(), value)}

    bound = functional.bind_interface(interface, fun)
    value = ir.RV(ir.Constant(1))

    assert_type(bound(value, 2, enabled=True), dict[str, ir.RV])


if TYPE_CHECKING:
    interface = ir.RVInterface()

    def valid(value: ir.RV, *, enabled: bool) -> ir.RV:
        return ir.RV(ir.Exp(), value)

    bound = functional.bind_interface(interface, valid)
    bound(ir.RV(ir.Constant(1)), enabled="yes")  # type: ignore

    def invalid(value: ir.RV) -> int:
        return 1

    functional.bind_interface(interface, invalid)  # type: ignore
