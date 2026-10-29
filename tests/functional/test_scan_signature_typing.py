from pangolin import interface as pi, ir
from pangolin.ir import RV


from collections.abc import Callable
from typing import assert_type

from pangolin.functional import scan


def test_preserves_no_additional_arguments():
    def step(carry: RV) -> RV:
        return RV(ir.Exp(), carry, interface=pi.default)

    scanned = scan(step, 3)

    assert_type(
        scanned,
        Callable[[RV], RV],
    )


def test_preserves_additional_argument_structures():
    def step(
        carry: RV,
        inputs: dict[str, RV],
        parameters: tuple[RV, RV],
    ) -> RV:
        return RV(
            ir.Add(), carry, inputs["x"] + parameters[0] + parameters[1], interface=pi.default
        )

    scanned = scan(step, 3)

    assert_type(
        scanned,
        Callable[
            [RV, dict[str, RV], tuple[RV, RV]],
            RV,
        ],
    )


def test_preserves_more_than_six_additional_argument_types():
    def step(
        carry: RV,
        a: RV,
        b: RV,
        c: RV,
        d: RV,
        e: RV,
        f: RV,
        seventh: int,
        eighth: str,
    ) -> RV:
        return RV(ir.Exp(), carry, interface=pi.default)

    scanned = scan(step, 3)

    assert_type(
        scanned,
        Callable[
            [RV, RV, RV, RV, RV, RV, RV, int, str],
            RV,
        ],
    )
