# pyright: reportUnnecessaryTypeIgnoreComment=error

from pangolin import interface as pi, ir
from pangolin.ir import RV


from collections.abc import Callable
from typing import TYPE_CHECKING, assert_type

from pangolin.functional import vmap


def test_preserves_no_argument_signature():
    def fun() -> RV:
        return RV(ir.Constant(1), interface=pi.default)

    mapped = vmap(fun, axis_size=3)

    assert_type(mapped(), RV)


def test_preserves_input_and_output_pytree_structures():
    def fun(
        inputs: dict[str, RV],
        parameters: tuple[RV, RV],
    ) -> dict[str, RV]:
        return {
            "result": RV(ir.Add(), inputs["x"], parameters[0] + parameters[1], interface=pi.default)
        }

    mapped = vmap(fun)

    assert_type(
        mapped,
        Callable[
            [dict[str, RV], tuple[RV, RV]],
            dict[str, RV],
        ],
    )


if TYPE_CHECKING:

    def no_arguments() -> RV:
        return RV(ir.Constant(1))

    # A zero-argument mapped function must still reject supplied arguments.
    vmap(no_arguments, axis_size=3)(RV(ir.Constant(1)))  # type: ignore


def test_preserves_more_than_six_argument_types():
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
        return RV(ir.Add(), a, b, interface=pi.default)

    mapped = vmap(fun)

    assert_type(
        mapped,
        Callable[
            [RV, RV, RV, RV, RV, RV, int, str],
            RV,
        ],
    )
