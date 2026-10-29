"""Static typing tests for `tracing.trace_call`.

Run with::

    pyright tests/functional/test_trace_call_typing.py

The negative case uses a necessary suppression. The file-level setting makes the test fail if the
invalid call becomes accepted and that suppression becomes unnecessary.
"""

from pangolin import interface as pi, ir
from pangolin.ir import RV

# pyright: reportUnnecessaryTypeIgnoreComment=error


from typing import TYPE_CHECKING, assert_type

from pangolin.interface import constant
from pangolin.interface import exp
from pangolin.functional.tracing import trace_call
from pangolin.util import PyTree


def make_normal(mean: RV, *, scale: RV) -> RV:
    return RV(ir.Normal(), mean, scale, interface=pi.default)


def make_tree(
    x: RV,
) -> tuple[RV, dict[str, RV]]:
    y = RV(ir.Exp(), x, interface=pi.default)
    z = RV(ir.Log(), y, interface=pi.default)
    return y, {"result": z}


def test_trace_call_returns_an_rv_pytree_and_flat_rv_lists():
    mean = RV(ir.Constant(0), interface=pi.default)
    scale = RV(ir.Constant(1), interface=pi.default)

    output, generated, boundary = trace_call(make_normal, mean, scale=scale)

    assert_type(generated, list[RV])
    assert_type(boundary, list[RV])
    assert_type(output, PyTree[RV])
    assert isinstance(output, RV)


def test_trace_call_preserves_the_pytree_structure_at_runtime():
    x = RV(ir.Constant(0), interface=pi.default)

    output, generated, boundary = trace_call(make_tree, x)

    assert_type(generated, list[RV])
    assert_type(boundary, list[RV])
    assert_type(
        output,
        PyTree[RV],
    )
    assert isinstance(output, tuple)
    assert isinstance(output[0], RV)
    assert isinstance(output[1], dict)
    assert isinstance(output[1]["result"], RV)


def test_trace_call_preserves_a_preexisting_output():
    x = RV(ir.Constant(0), interface=pi.default)

    def identity(value: RV) -> RV:
        return value

    output, generated, boundary = trace_call(identity, x)

    assert_type(output, PyTree[RV])
    assert output is x
    assert_type(generated, list[RV])
    assert_type(boundary, list[RV])


def test_trace_call_accepts_typed_non_rv_arguments():
    def fun(value: int, enabled: bool, option: None) -> RV:
        return RV(ir.Constant(value if enabled else 0), interface=pi.default)

    output, generated, boundary = trace_call(fun, 2, True, None)

    assert_type(output, PyTree[RV])
    assert isinstance(output, RV)
    assert_type(generated, list[RV])
    assert_type(boundary, list[RV])


def test_trace_call_accepts_interface_coercion():
    constant_output, _, _ = trace_call(constant, 2)
    exp_output, _, _ = trace_call(exp, 2)

    assert_type(constant_output, PyTree[RV])
    assert_type(exp_output, PyTree[RV])
    assert isinstance(constant_output, RV)
    assert isinstance(exp_output, RV)


if TYPE_CHECKING:

    def invalid_output() -> int:
        return 1

    trace_call(invalid_output)  # type: ignore

    def invalid_tree() -> tuple[RV, None]:
        return RV(ir.Constant(1), interface=pi.default), None

    trace_call(invalid_tree)  # type: ignore
