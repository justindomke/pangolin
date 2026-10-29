from typing import TYPE_CHECKING, Any, assert_type

import pytest

from pangolin import dag, functional, interface as pi, ir
from pangolin.functional.tracing import AbstractOp


def test_exports():
    assert pi.grad == pi.default.grad
    assert functional.grad is not pi.grad


def test_square_builds_two_vjps_and_a_sum():
    def square(x: ir.RV) -> ir.RV:
        return x * x

    x = pi.constant(3.0)
    result = pi.grad(square)(x)
    assert_type(result, Any)
    assert result.op == ir.Add()
    assert [p.op for p in result.parents] == [ir.VJP(ir.Mul(), 0), ir.VJP(ir.Mul(), 1)]
    assert result.parents[0].parents[1:] == (x, x)
    assert result.interface is pi.default
    assert not any(
        isinstance(node.op, (AbstractOp, ir.Composite)) for node in dag.upstream_nodes(result)
    )


def test_identity_returns_one_even_for_random_input():
    result = pi.grad(lambda x: x)(pi.normal(0, 1))
    assert result.op == ir.Identity()
    assert result.parents[0].op == ir.Constant(1.0)
    assert result.interface is pi.default


def test_unused_array_argument_returns_matching_zeros():
    result = pi.grad(lambda x: pi.constant(7))(pi.constant([[1, 2], [3, 4]]))
    assert result.op == ir.Constant([[0.0, 0.0], [0.0, 0.0]])
    assert result.shape == (2, 2)


def test_same_rv_in_two_arguments_remains_two_independent_arguments():
    x = pi.constant(3.0)
    result = pi.grad(lambda a, b: a * b)(x, x)
    assert result.op == ir.VJP(ir.Mul(), 0)
    assert result.parents[1:] == (x, x)


def test_capture_alias_does_not_get_differentiated():
    x = pi.constant(3.0)
    result = pi.grad(lambda a: a * x)(x)
    assert result.op == ir.VJP(ir.Mul(), 0)


def test_selected_input_is_abstracted_once_and_keeps_its_interface():
    x = pi.constant(3.0)
    calls = []

    def fun(a):
        calls.append(a)
        return a * a

    pi.grad(fun)(x)
    assert len(calls) == 1
    assert calls[0] is not x
    assert calls[0].interface is x.interface


def test_output_interface_is_preserved():
    interface = ir.RVInterface()
    result = pi.grad(lambda x: ir.RV(ir.Exp(), x, interface=interface))(pi.constant(0))
    assert result.interface is interface


def test_unused_random_node_is_ignored():
    def fun(x):
        pi.normal(0, 1)
        return x * x

    assert pi.grad(fun)(pi.constant(2)).op == ir.Add()


@pytest.mark.parametrize("argnum", [True, 0.5, "0"])
def test_rejects_invalid_argnum_type(argnum):
    with pytest.raises(TypeError, match="integer"):
        pi.grad(lambda x: x, argnum)


def test_rejects_negative_argnum():
    with pytest.raises(ValueError, match="nonnegative"):
        pi.grad(lambda x: x, -1)


def test_requires_selected_positional_rv():
    with pytest.raises(ValueError, match="positional"):
        pi.grad(lambda x: x)(x=pi.constant(0))
    with pytest.raises(TypeError, match="leaves are RV"):
        functional.grad(lambda x: x)(0)


def test_rejects_random_output():
    with pytest.raises(ValueError, match="random RVs"):
        pi.grad(lambda x: pi.normal(x, 1))(pi.constant(0))


def test_rejects_random_intermediate_even_off_derivative_path():
    with pytest.raises(ValueError, match="random RVs"):
        pi.grad(lambda x: x + pi.normal(0, 1))(pi.constant(0))


def test_rejects_array_output():
    with pytest.raises(ValueError, match="scalar"):
        pi.grad(lambda x: x)(pi.constant([1, 2]))


@pytest.mark.parametrize("output", [None, 1, [], [pi.constant(0)]])
def test_rejects_non_single_rv_output(output):
    with pytest.raises(TypeError):
        pi.grad(lambda x: output)(pi.constant(0))


def test_signature_preserves_static_arguments():
    def fun(x: ir.RV, *, enabled: bool) -> ir.RV:
        return x * x if enabled else x

    result = pi.grad(fun)(pi.constant(0), enabled=True)
    assert_type(result, Any)
    if TYPE_CHECKING:
        pi.grad(fun)(pi.constant(0), enabled="bad")  # type: ignore


def test_lambda_results_can_feed_typed_consumers():
    def scalar_consumer(x: ir.RV) -> ir.RV:
        return x

    def tree_consumer(xs: list[ir.RV]) -> list[ir.RV]:
        return xs

    x = pi.constant(2.0)
    scalar = pi.grad(lambda x: x * x)(x)
    tree = pi.grad(lambda xs: xs[0] * xs[1])([x, x])
    assert_type(scalar, Any)
    assert_type(tree, Any)
    assert scalar_consumer(scalar).op == ir.Add()
    assert [r.op for r in tree_consumer(tree)] == [ir.VJP(ir.Mul(), 0), ir.VJP(ir.Mul(), 1)]


def test_broad_callback_keeps_runtime_rv_requirement():
    def fun(x: pi.RVLike) -> ir.RV:
        return pi.exp(x)

    result = pi.grad(fun)(pi.constant(2.0))
    assert_type(result, Any)
    assert result.op == ir.VJP(ir.Exp(), 0)
    with pytest.raises(TypeError, match="leaves are RV"):
        functional.grad(fun)(2.0)


def test_configured_interface_with_dynamic_seventh_argument():
    configured = pi.Interface(broadcasting="off")

    def fun(a: ir.RV, b: ir.RV, c: ir.RV, d: ir.RV, e: ir.RV, f: ir.RV,
            p: dict[str, ir.RV], *, enabled: bool = True) -> ir.RV:
        return p["x"] * a if enabled else p["x"]

    x = configured.constant(2.0)
    args = (x, x, x, x, x, x, {"x": x})
    result = configured.grad(fun, len(args) - 1)(*args)
    assert_type(result, Any)
    assert result["x"].op == ir.VJP(ir.Mul(), 0)
    assert result["x"].parents[1:] == (x, x)
