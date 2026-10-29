# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Static typing tests for `functional.composite`."""

from typing import TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin.functional import composite


def test_preserves_the_complete_function_signature():
    def fun(
        x: ir.RV,
        count: int,
        *values: float,
        enabled: bool,
        label: str = "default",
    ) -> ir.RV:
        return ir.RV(ir.Exp(), x)

    compressed = composite(fun)
    x = ir.RV(ir.Constant(1))

    assert_type(compressed(x, 2, 3.0, 4.0, enabled=True, label="example"), ir.RV)


if TYPE_CHECKING:

    def invalid_output(x: ir.RV) -> int:
        return 1

    composite(invalid_output)  # type: ignore

    def valid(x: ir.RV, *, enabled: bool) -> ir.RV:
        return ir.RV(ir.Exp(), x)

    compressed = composite(valid)
    x = ir.RV(ir.Constant(1))
    compressed(x, enabled="yes")  # type: ignore
