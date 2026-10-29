# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Direct and decorator loop forms return RVs, including coerced outputs."""

from typing import TYPE_CHECKING, assert_type

from pangolin import ir, interface as pi


def test_direct_and_decorator_return_types():
    assert_type(pi.loop(lambda i: i + 1, i=2), ir.RV)
    assert_type(pi.loop(lambda i: 1.0, i=2), ir.RV)
    assert_type(pi.loop(lambda i: [1.0, 2.0], i=2), ir.RV)
    assert_type(pi.loop(lambda i=2: i + 1), ir.RV)
    assert_type(pi.loop(lambda prev, t: prev[t - 1] + 1.0, t=[2, 0.0]), ir.RV)

    assert_type(pi.loop(lambda prev, t: prev[t - 1] + 1.0, t=(2, 0.0)), ir.RV)

    @pi.loop
    def x(prev, t=[2]):
        if t is None:
            return 0.0
        return prev[t - 1] + 1.0

    assert_type(x, ir.RV)

    other = pi.Interface()
    assert_type(other.loop(lambda i: 1.0, i=2), ir.RV)
    assert_type(other.loop(lambda prev, t=[2, 0.0]: prev[t - 1] + 1.0), ir.RV)


if TYPE_CHECKING:
    pi.loop(lambda i: {"a": i}, i=2)  # type: ignore
    pi.loop(lambda prev, t: prev[t - 1] + 1, t=[3, lambda: 0.0])  # type: ignore
