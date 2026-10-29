"""Static typing tests for `compositing`.

Run with::

    pyright tests/functional/test_composite_from_trace_typing.py
"""

from pangolin import interface as pi, ir
from pangolin.ir import RV

# pyright: reportUnnecessaryTypeIgnoreComment=error


from typing import TYPE_CHECKING, assert_type

from pangolin.functional.compositing import composite
from pangolin.functional.compositing import composite_from_trace


def make_normal(mean: RV, *, scale: RV) -> RV:
    return RV(ir.Normal(), mean, scale, interface=pi.default)


def test_composite_preserves_the_signature():
    mean = RV(ir.Constant(0), interface=pi.default)
    scale = RV(ir.Constant(1), interface=pi.default)

    result = composite(make_normal)(mean, scale=scale)

    assert_type(result, RV)


def test_composite_from_trace_returns_an_op():
    mean = RV(ir.Constant(0), interface=pi.default)
    scale = RV(ir.Constant(1), interface=pi.default)
    output = make_normal(mean, scale=scale)

    op = composite_from_trace(output, [output], [mean, scale])

    assert_type(op, ir.Op)


def test_existing_output_also_returns_an_op():
    output = RV(ir.Constant(0), interface=pi.default)

    op = composite_from_trace(output, [], [output], allow_existing_output=True)

    assert_type(op, ir.Op)


if TYPE_CHECKING:
    mean = RV(ir.Constant(0), interface=pi.default)
    scale = RV(ir.Constant(1), interface=pi.default)
    composite(make_normal)(1, scale=scale)  # type: ignore
    composite(make_normal)(mean)  # type: ignore
