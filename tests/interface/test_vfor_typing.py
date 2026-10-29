"""Static typing tests for `vfor`.

Run these tests with::

    pyright tests/interface/test_vfor_typing.py

`assert_type` checks positive inference but does nothing at runtime. Negative examples live under
`TYPE_CHECKING` so Pyright analyzes them without pytest executing them. Each expected error is
suppressed locally; the file-level ``reportUnnecessaryTypeIgnoreComment`` setting makes Pyright
fail if a regression causes an invalid example to be accepted and its suppression becomes unused.
"""

from pangolin.ir import RV

# pyright: reportUnnecessaryTypeIgnoreComment=error


from typing import TYPE_CHECKING, assert_type

from pangolin.interface import normal
from pangolin.interface import vfor


def test_accepts_every_supported_arity():
    result_0 = vfor(lambda: normal(0, 1))
    result_1 = vfor(lambda a: normal(assert_type(a, RV), 1), a=1)
    result_2 = vfor(
        lambda a, b: normal(assert_type(a, RV) + assert_type(b, RV), 1),
        a=1,
        b=1,
    )
    result_3 = vfor(
        lambda a, b, c: normal(
            assert_type(a, RV) + assert_type(b, RV) + assert_type(c, RV),
            1,
        ),
        a=1,
        b=1,
        c=1,
    )
    result_4 = vfor(
        lambda a, b, c, d: normal(
            assert_type(a, RV) + assert_type(b, RV) + assert_type(c, RV) + assert_type(d, RV),
            1,
        ),
        a=1,
        b=1,
        c=1,
        d=1,
    )
    result_5 = vfor(
        lambda a, b, c, d, e: normal(
            assert_type(a, RV)
            + assert_type(b, RV)
            + assert_type(c, RV)
            + assert_type(d, RV)
            + assert_type(e, RV),
            1,
        ),
        a=1,
        b=1,
        c=1,
        d=1,
        e=1,
    )
    result_6 = vfor(
        lambda a, b, c, d, e, f: normal(
            assert_type(a, RV)
            + assert_type(b, RV)
            + assert_type(c, RV)
            + assert_type(d, RV)
            + assert_type(e, RV)
            + assert_type(f, RV),
            1,
        ),
        a=1,
        b=1,
        c=1,
        d=1,
        e=1,
        f=1,
    )

    assert_type(result_0, RV)
    assert_type(result_1, RV)
    assert_type(result_2, RV)
    assert_type(result_3, RV)
    assert_type(result_4, RV)
    assert_type(result_5, RV)
    assert_type(result_6, RV)


def test_accepts_pytree_outputs():
    def fun(i: RV) -> tuple[RV, dict[str, RV]]:
        return normal(i, 1), {"deterministic": i + 1}

    result = vfor(fun, i=2)

    assert_type(result, tuple[RV, dict[str, RV]])


def test_accepts_more_than_six_arguments():
    def fun(
        a: RV,
        b: RV,
        c: RV,
        d: RV,
        e: RV,
        f: RV,
        seventh: int,
        eighth: str,
    ) -> RV:
        return normal(a + b + c + d + e + f, 1)

    result = vfor(fun, a=1, b=1, c=1, d=1, e=1, f=1, seventh=1, eighth=1)

    assert_type(result, RV)


def test_accepts_decorator_factory_syntax():
    decorate = vfor(i=2)
    result = decorate(lambda i: normal(assert_type(i, RV), 1))

    assert_type(result, RV)


def test_decorator_accepts_more_than_six_arguments():
    def fun(
        a: RV,
        b: RV,
        c: RV,
        d: RV,
        e: RV,
        f: RV,
        seventh: int,
    ) -> RV:
        return normal(a + b + c + d + e + f, 1)

    decorate = vfor(a=1, b=1, c=1, d=1, e=1, f=1, seventh=1)
    result = decorate(fun)

    assert_type(result, RV)


if TYPE_CHECKING:
    choices: list[RV] = []

    vfor(
        lambda i: normal(choices[i], 1),  # type: ignore
        i=3,
    )

    def integer_callback(i: int) -> RV:
        return normal(i, 1)

    vfor(integer_callback, i=3)  # type: ignore

    def wrong_second_parameter(i: RV, j: int) -> RV:
        return normal(i + j, 1)

    vfor(wrong_second_parameter, i=3, j=3)  # type: ignore

    def non_rv_output(i: RV) -> int:
        return 1

    vfor(non_rv_output, i=3)  # type: ignore
