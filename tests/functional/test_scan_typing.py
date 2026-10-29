# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Static typing tests for `functional.scan`."""

from collections.abc import Callable
from typing import TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin.functional import scan


def test_returns_plain_rv():
    def step(carry: ir.RV) -> ir.RV:
        return ir.RV(ir.Exp(), carry)

    scanned = scan(step, 3)

    assert_type(scanned, Callable[[ir.RV], ir.RV])


def test_preserves_arbitrary_additional_argument_types():
    def step(
        carry: ir.RV,
        inputs: dict[str, ir.RV],
        configuration: str,
    ) -> ir.RV:
        return ir.RV(ir.Add(), carry, inputs[configuration])

    scanned = scan(step, 3, in_axes=({"x": 0}, None))

    assert_type(
        scanned,
        Callable[[ir.RV, dict[str, ir.RV], str], ir.RV],
    )


def test_preserves_more_than_six_additional_argument_types():
    def step(
        carry: ir.RV,
        a: ir.RV,
        b: ir.RV,
        c: ir.RV,
        d: ir.RV,
        e: ir.RV,
        f: ir.RV,
        seventh: int,
        eighth: str,
    ) -> ir.RV:
        return ir.RV(ir.Exp(), carry)

    scanned = scan(step, 3)

    assert_type(
        scanned,
        Callable[
            [ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, int, str],
            ir.RV,
        ],
    )


if TYPE_CHECKING:

    def invalid_output(carry: ir.RV) -> int:
        return 1

    scan(invalid_output, 3)  # type: ignore

    def valid_step(carry: ir.RV) -> ir.RV:
        return ir.RV(ir.Exp(), carry)

    valid = scan(valid_step, 3)
    valid(1)  # type: ignore
