# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Check callback argument inference and exact output typing for the named-loop combinators."""

from typing import TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin.interface import Interface, sfor, vfor

start = ir.RV(ir.Constant(0))


def test_vfor_arities_and_output_typing():
    assert_type(vfor(lambda: ir.RV(ir.Constant(0))), ir.RV)
    assert_type(vfor()(lambda: ir.RV(ir.Constant(0))), ir.RV)
    assert_type(vfor(lambda a: ir.RV(ir.Exp(), assert_type(a, ir.RV)), a=1), ir.RV)
    assert_type(vfor(a=1)(lambda a: ir.RV(ir.Exp(), assert_type(a, ir.RV))), ir.RV)
    assert_type(
        vfor(lambda a, b: ir.RV(ir.Exp(), assert_type(a, ir.RV)), a=1, b=1),
        ir.RV,
    )
    assert_type(
        vfor(a=1, b=1)(lambda a, b: ir.RV(ir.Exp(), assert_type(a, ir.RV))),
        ir.RV,
    )
    assert_type(
        vfor(
            lambda a, b, c: ir.RV(ir.Exp(), assert_type(a, ir.RV)),
            a=1,
            b=1,
            c=1,
        ),
        ir.RV,
    )
    assert_type(
        vfor(a=1, b=1, c=1)(lambda a, b, c: ir.RV(ir.Exp(), assert_type(a, ir.RV))),
        ir.RV,
    )
    assert_type(
        vfor(
            lambda a, b, c, d: ir.RV(ir.Exp(), assert_type(a, ir.RV)),
            a=1,
            b=1,
            c=1,
            d=1,
        ),
        ir.RV,
    )
    assert_type(
        vfor(a=1, b=1, c=1, d=1)(lambda a, b, c, d: ir.RV(ir.Exp(), assert_type(a, ir.RV))),
        ir.RV,
    )
    assert_type(
        vfor(
            lambda a, b, c, d, e: ir.RV(ir.Exp(), assert_type(a, ir.RV)),
            a=1,
            b=1,
            c=1,
            d=1,
            e=1,
        ),
        ir.RV,
    )
    assert_type(
        vfor(a=1, b=1, c=1, d=1, e=1)(lambda a, b, c, d, e: ir.RV(ir.Exp(), assert_type(a, ir.RV))),
        ir.RV,
    )
    assert_type(
        vfor(
            lambda a, b, c, d, e, f: ir.RV(ir.Exp(), assert_type(a, ir.RV)),
            a=1,
            b=1,
            c=1,
            d=1,
            e=1,
            f=1,
        ),
        ir.RV,
    )
    assert_type(
        vfor(a=1, b=1, c=1, d=1, e=1, f=1)(
            lambda a, b, c, d, e, f: ir.RV(ir.Exp(), assert_type(a, ir.RV))
        ),
        ir.RV,
    )
    assert_type(
        vfor(
            lambda a, b, c, d, e, f, g: ir.RV(ir.Exp(), assert_type(a, ir.RV)),
            a=1,
            b=1,
            c=1,
            d=1,
            e=1,
            f=1,
            g=1,
        ),
        ir.RV,
    )
    assert_type(
        vfor(a=1, b=1, c=1, d=1, e=1, f=1, g=1)(
            lambda a, b, c, d, e, f, g: ir.RV(ir.Exp(), assert_type(a, ir.RV))
        ),
        ir.RV,
    )

    result = vfor(lambda i: (ir.RV(ir.Exp(), i), {"x": ir.RV(ir.Log(), i)}), i=2)
    assert_type(result, tuple[ir.RV, dict[str, ir.RV]])


def test_sfor_arities_and_output_typing():
    initial = ir.RV(ir.Constant(0))
    assert_type(
        sfor(
            lambda previous, t: ir.RV(
                ir.Add(), assert_type(previous, ir.RV), assert_type(t, ir.RV)
            ),
            init=initial,
            t=1,
        ),
        ir.RV,
    )
    assert_type(
        sfor(init=initial, t=1)(
            lambda previous, t: ir.RV(ir.Add(), assert_type(previous, ir.RV), assert_type(t, ir.RV))
        ),
        ir.RV,
    )
    initial = ir.RV(ir.Constant([0]))
    assert_type(
        sfor(
            lambda previous, t, i: ir.RV(
                ir.Add(), ir.RV(ir.Index(), assert_type(previous, ir.RV), i), assert_type(t, ir.RV)
            ),
            init=initial,
            t=1,
            i=1,
        ),
        ir.RV,
    )
    assert_type(
        sfor(init=initial, t=1, i=1)(
            lambda previous, t, i: ir.RV(
                ir.Add(), ir.RV(ir.Index(), assert_type(previous, ir.RV), i), assert_type(t, ir.RV)
            )
        ),
        ir.RV,
    )
    initial = ir.RV(ir.Constant([[0]]))
    assert_type(
        sfor(
            lambda previous, t, i, j: ir.RV(
                ir.Add(),
                ir.RV(ir.Index(), assert_type(previous, ir.RV), i, j),
                assert_type(t, ir.RV),
            ),
            init=initial,
            t=1,
            i=1,
            j=1,
        ),
        ir.RV,
    )
    assert_type(
        sfor(init=initial, t=1, i=1, j=1)(
            lambda previous, t, i, j: ir.RV(
                ir.Add(),
                ir.RV(ir.Index(), assert_type(previous, ir.RV), i, j),
                assert_type(t, ir.RV),
            )
        ),
        ir.RV,
    )
    initial = ir.RV(ir.Constant([[[0]]]))
    assert_type(
        sfor(
            lambda previous, t, i, j, k: ir.RV(
                ir.Add(),
                ir.RV(ir.Index(), assert_type(previous, ir.RV), i, j, k),
                assert_type(t, ir.RV),
            ),
            init=initial,
            t=1,
            i=1,
            j=1,
            k=1,
        ),
        ir.RV,
    )
    assert_type(
        sfor(init=initial, t=1, i=1, j=1, k=1)(
            lambda previous, t, i, j, k: ir.RV(
                ir.Add(),
                ir.RV(ir.Index(), assert_type(previous, ir.RV), i, j, k),
                assert_type(t, ir.RV),
            )
        ),
        ir.RV,
    )
    initial = ir.RV(ir.Constant([[[[0]]]]))
    assert_type(
        sfor(
            lambda previous, t, i, j, k, l: ir.RV(
                ir.Add(),
                ir.RV(ir.Index(), assert_type(previous, ir.RV), i, j, k, l),
                assert_type(t, ir.RV),
            ),
            init=initial,
            t=1,
            i=1,
            j=1,
            k=1,
            l=1,
        ),
        ir.RV,
    )
    assert_type(
        sfor(init=initial, t=1, i=1, j=1, k=1, l=1)(
            lambda previous, t, i, j, k, l: ir.RV(
                ir.Add(),
                ir.RV(ir.Index(), assert_type(previous, ir.RV), i, j, k, l),
                assert_type(t, ir.RV),
            )
        ),
        ir.RV,
    )
    initial = ir.RV(ir.Constant([[[[[0]]]]]))
    assert_type(
        sfor(
            lambda previous, t, i, j, k, l, m: ir.RV(
                ir.Add(),
                ir.RV(ir.Index(), assert_type(previous, ir.RV), i, j, k, l, m),
                assert_type(t, ir.RV),
            ),
            init=initial,
            t=1,
            i=1,
            j=1,
            k=1,
            l=1,
            m=1,
        ),
        ir.RV,
    )
    assert_type(
        sfor(init=initial, t=1, i=1, j=1, k=1, l=1, m=1)(
            lambda previous, t, i, j, k, l, m: ir.RV(
                ir.Add(),
                ir.RV(ir.Index(), assert_type(previous, ir.RV), i, j, k, l, m),
                assert_type(t, ir.RV),
            )
        ),
        ir.RV,
    )


if TYPE_CHECKING:

    def wrong_coordinate(i: int) -> ir.RV:
        return start

    def wrong_carry(previous: int, t: ir.RV) -> ir.RV:
        return start

    vfor(wrong_coordinate, i=2)  # type: ignore
    vfor(i=2)(wrong_coordinate)  # type: ignore
    vfor(lambda i: 1, i=2)  # type: ignore
    vfor(lambda i: ir.RV(ir.Exp(), i), i=2)
    sfor(wrong_carry, init=start, t=2)  # type: ignore
    sfor(init=start, t=2)(wrong_carry)  # type: ignore
    sfor(lambda previous, t: (previous, t), init=start, t=2)  # type: ignore
    sfor(lambda previous, t: previous, init=0, t=2)


def test_configured_method_types():
    pi = Interface()
    assert_type(pi.vfor(lambda i: pi.exp(assert_type(i, ir.RV)), i=2), ir.RV)
    assert_type(pi.vfor(i=2)(lambda i: (pi.exp(i), pi.log(i))), tuple[ir.RV, ir.RV])
    assert_type(pi.sfor(lambda previous, t: previous + t, init=0, t=2), ir.RV)
    assert_type(pi.sfor(init=0, t=2)(lambda previous, t: previous + t), ir.RV)


if TYPE_CHECKING:
    pi = Interface()
    pi.vfor(wrong_coordinate, i=2)  # type: ignore
    pi.vfor(i=2)(wrong_coordinate)  # type: ignore
    pi.sfor(wrong_carry, init=0, t=2)  # type: ignore
    pi.sfor(init=0, t=2)(wrong_carry)  # type: ignore
