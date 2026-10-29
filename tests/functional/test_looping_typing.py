# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Loop returns one RV for scalar and array bodies, including nested loops."""

from functools import partial

from typing import TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin.functional import loop
from pangolin.functional.looping import Previous


def test_return_types_for_zero_one_and_many_coordinates():
    assert_type(loop(lambda: ir.RV(ir.Constant(0)), interface=ir.NO_INTERFACE), ir.RV)

    def one(i: ir.RV) -> ir.RV:
        return ir.RV(ir.Exp(), i)

    assert_type(loop(one, interface=ir.NO_INTERFACE, i=2), ir.RV)
    assert_type(loop(lambda i: ir.RV(ir.Exp(), i), interface=ir.NO_INTERFACE, i=2), ir.RV)
    assert_type(
        loop(
            lambda a, b, c, d, e, f, g: ir.RV(ir.Add(), a, g),
            interface=ir.NO_INTERFACE,
            a=1,
            b=1,
            c=1,
            d=1,
            e=1,
            f=1,
            g=2,
        ),
        ir.RV,
    )


def test_nested_array_output_is_an_rv():
    def outer(i: ir.RV) -> ir.RV:
        return loop(lambda j: ir.RV(ir.Add(), i, j), interface=ir.NO_INTERFACE, j=3)

    result = assert_type(loop(outer, interface=ir.NO_INTERFACE, i=2), ir.RV)
    assert_type(ir.RV(ir.Sum(0), result), ir.RV)
    assert result.shape == (2, 3)


def test_sequential_output_type():
    one = ir.RV(ir.Constant(1))

    def initial(i: ir.RV) -> ir.RV:
        return ir.RV(ir.Identity(), i)

    def model(prev: Previous, i: ir.RV, t: ir.RV | None) -> ir.RV:
        if t is None:
            return initial(i)
        old = prev[i, ir.RV(ir.Sub(), t, one)]
        return ir.RV(ir.Add(), old, one)

    result = assert_type(loop(model, interface=ir.NO_INTERFACE, i=2, t=[3]), ir.RV)
    assert result.shape == (2, 3)


if TYPE_CHECKING:
    loop(lambda prev, t: prev[t - 1] + 1, interface=ir.NO_INTERFACE, t=[3, lambda: ir.RV(ir.Constant(0))])  # type: ignore
    loop(lambda i: 1, interface=ir.NO_INTERFACE, i=2)  # type: ignore
    loop(lambda i: (ir.RV(ir.Exp(), i),), interface=ir.NO_INTERFACE, i=2)  # type: ignore
    loop(lambda i: ir.RV(ir.Exp(), i), i=2)  # type: ignore
    loop(lambda i: ir.RV(ir.Exp(), i), interface=ir.NO_INTERFACE, i="2")  # type: ignore


def test_multiple_sequential_output_type():
    one = ir.RV(ir.Constant(1))

    def model(prev: Previous, t: ir.RV | None, s: ir.RV | None) -> ir.RV:
        if t is None:
            return pi.constant(1)
        if s is None:
            return pi.constant(1)
        old_t = prev[ir.RV(ir.Sub(), t, one), s]
        old_s = prev[t, ir.RV(ir.Sub(), s, one)]
        return ir.RV(ir.Add(), old_t, old_s, interface=old_t.interface)

    from pangolin import interface as pi

    result = assert_type(
        loop(
            model,
            interface=ir.NO_INTERFACE,
            t=[2],
            s=[3],
        ),
        ir.RV,
    )
    assert result.shape == (2, 3)


def test_decorator_and_initializer_types():
    from pangolin import interface as pi

    @partial(loop, interface=pi.default)
    def vectorized(i=2) -> ir.RV:
        return pi.add(i, 1)

    assert_type(vectorized, ir.RV)

    @partial(loop, interface=pi.default)
    def explicit(prev: Previous, t=[3, pi.constant(0)]) -> ir.RV:
        return prev[t - 1] + 1

    assert_type(explicit, ir.RV)

    @partial(loop, interface=pi.default)
    def implicit(prev: Previous, t=[3]) -> ir.RV:
        if t is None:
            return pi.constant(0)
        return prev[t - 1] + 1

    assert_type(implicit, ir.RV)
    assert_type(
        loop(lambda prev, t: prev[t - 1] + 1, interface=pi.default, t=[3, pi.constant(0)]), ir.RV
    )
