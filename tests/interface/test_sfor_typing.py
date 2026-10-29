"""Static typing tests for `sfor`."""

from pangolin.ir import RV

# pyright: reportUnnecessaryTypeIgnoreComment=error


from typing import TYPE_CHECKING, assert_type

import numpy as np

from pangolin.interface import normal
from pangolin.interface import sfor


def test_direct_form_types_state_and_coordinates_as_rvs():
    result = sfor(
        lambda previous, t, i: normal(
            assert_type(previous, RV)[assert_type(i, RV)],
            assert_type(t, RV) + 1,
        ),
        init=[0, 0],
        t=3,
        i=2,
    )

    assert_type(result, RV)


def test_decorator_form_returns_an_rv():
    decorate = sfor(init=[0, 0], t=3, i=2)

    result = decorate(
        lambda previous, t, i: normal(
            assert_type(previous, RV)[assert_type(i, RV)],
            assert_type(t, RV) + 1,
        )
    )

    assert_type(result, RV)


def test_variadic_overload_accepts_exactly_six_parameters():
    def body(
        previous: RV,
        t: RV,
        a: RV,
        b: RV,
        c: RV,
        d: RV,
    ) -> RV:
        return normal(previous[a, b, c, d], t + 1)

    direct = sfor(
        body,
        init=np.zeros((1, 1, 1, 1)),
        t=1,
        a=1,
        b=1,
        c=1,
        d=1,
    )
    assert_type(direct, RV)

    decorated = sfor(
        init=np.zeros((1, 1, 1, 1)),
        t=1,
        a=1,
        b=1,
        c=1,
        d=1,
    )(body)
    assert_type(decorated, RV)


def test_accepts_more_than_six_coordinate_parameters():
    def body(
        previous: RV,
        t: RV,
        a: RV,
        b: RV,
        c: RV,
        d: RV,
        e: RV,
        f: RV,
    ) -> RV:
        return normal(previous[a, b, c, d, e, f], t + 1)

    result = sfor(
        body,
        init=np.zeros((1, 1, 1, 1, 1, 1)),
        t=1,
        a=1,
        b=1,
        c=1,
        d=1,
        e=1,
        f=1,
    )

    assert_type(result, RV)


if TYPE_CHECKING:
    sfor(lambda previous, t: normal(previous, 1), init=0, t="3")  # type: ignore

    def wrong_state_type(previous: int, t: RV) -> RV:
        return normal(previous, 1)

    sfor(wrong_state_type, init=0, t=3)  # type: ignore

    def non_rv_output(previous: RV, t: RV) -> int:
        return 1

    sfor(non_rv_output, init=0, t=3)  # type: ignore
