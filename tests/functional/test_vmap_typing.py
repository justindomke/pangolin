# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Static typing tests for `functional.vmap`."""

from collections.abc import Callable
from typing import TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin.functional import vmap


def test_preserves_a_zero_argument_output_type():
    def fun() -> ir.RV:
        return ir.RV(ir.Constant(1))

    mapped = vmap(fun, axis_size=3, simplify=False)

    assert_type(mapped(), ir.RV)


def test_preserves_argument_and_output_pytree_types():
    def fun(
        inputs: dict[str, ir.RV],
        parameters: tuple[ir.RV, ir.RV],
        selected: str,
    ) -> dict[str, ir.RV]:
        total = ir.RV(ir.Add(), inputs[selected], parameters[0])
        return {"result": ir.RV(ir.Add(), total, parameters[1])}

    mapped = vmap(fun, in_axes=({"x": 0}, (None, None), None))

    assert_type(
        mapped,
        Callable[
            [dict[str, ir.RV], tuple[ir.RV, ir.RV], str],
            dict[str, ir.RV],
        ],
    )


def test_preserves_more_than_six_argument_types():
    def fun(
        a: ir.RV,
        b: ir.RV,
        c: ir.RV,
        d: ir.RV,
        e: ir.RV,
        f: ir.RV,
        seventh: int,
        eighth: str,
    ) -> ir.RV:
        return ir.RV(ir.Add(), a, b)

    mapped = vmap(fun)

    assert_type(
        mapped,
        Callable[
            [ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, int, str],
            ir.RV,
        ],
    )


if TYPE_CHECKING:

    def invalid_output(value: ir.RV) -> tuple[ir.RV, None]:
        return ir.RV(ir.Exp(), value), None

    vmap(invalid_output)  # type: ignore

    def valid_fun(value: ir.RV, configuration: str) -> ir.RV:
        return ir.RV(ir.Exp(), value)

    mapped = vmap(valid_fun, in_axes=(0, None))
    mapped(ir.RV(ir.Constant([1, 2, 3])), 1)  # type: ignore
