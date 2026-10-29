# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Check broadcasting callback types and preservation of positional arity."""

from typing import TYPE_CHECKING, assert_type

from pangolin import functional, ir


def zero() -> ir.RV:
    return ir.RV(ir.Constant(0))


def one(a: ir.RV) -> ir.RV:
    return a


def two(a: ir.RV, b: ir.RV) -> ir.RV:
    return a


def three(a: ir.RV, b: ir.RV, c: ir.RV) -> ir.RV:
    return a


def four(a: ir.RV, b: ir.RV, c: ir.RV, d: ir.RV) -> ir.RV:
    return a


def five(a: ir.RV, b: ir.RV, c: ir.RV, d: ir.RV, e: ir.RV) -> ir.RV:
    return a


def six(a: ir.RV, b: ir.RV, c: ir.RV, d: ir.RV, e: ir.RV, f: ir.RV) -> ir.RV:
    return a


def seven(a: ir.RV, b: ir.RV, c: ir.RV, d: ir.RV, e: ir.RV, f: ir.RV, g: ir.RV) -> ir.RV:
    return a


def test_positional_arity_and_return_types():
    x = ir.RV(ir.Constant(1))
    for combinator in (
        functional.broadcast_off,
        functional.broadcast_simple,
        functional.broadcast_rank,
        functional.broadcast_numpy,
    ):
        assert_type(combinator(zero)(), ir.RV)
        assert_type(combinator(one)(x), ir.RV)
        assert_type(combinator(two)(x, x), ir.RV)
        assert_type(combinator(three)(x, x, x), ir.RV)
        assert_type(combinator(four)(x, x, x, x), ir.RV)
        assert_type(combinator(five)(x, x, x, x, x), ir.RV)
        assert_type(combinator(six)(x, x, x, x, x, x), ir.RV)
        assert_type(combinator(seven)(x, x, x, x, x, x, x), ir.RV)


if TYPE_CHECKING:

    def wrong_input(x: int) -> ir.RV:
        return ir.RV(ir.Constant(x))

    def wrong_output(x: ir.RV) -> int:
        return 1

    x = ir.RV(ir.Constant(1))
    for combinator in (
        functional.broadcast_off,
        functional.broadcast_simple,
        functional.broadcast_rank,
        functional.broadcast_numpy,
    ):
        combinator(wrong_input)  # type: ignore
        combinator(wrong_output)  # type: ignore
        combinator(two)(x)  # type: ignore
        combinator(two)(x, x, x)  # type: ignore
        combinator(two)(x, 1)  # type: ignore
        combinator(six)(x, x, x, x, x)  # type: ignore
        combinator(six)(x, x, x, x, x, x, x)  # type: ignore
        combinator(seven)(x, x, x, x, x, x)  # type: ignore
        combinator(seven)(x, x, x, x, x, x, 1)  # type: ignore
