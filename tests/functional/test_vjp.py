from typing import TYPE_CHECKING, assert_type

import pytest

from pangolin import dag, functional, interface as pi, ir
from pangolin.functional.tracing import AbstractOp


def test_exports():
    assert pi.vjp is functional.vjp
    assert pi.default.vjp is functional.vjp


def test_seeds_with_supplied_cotangent_and_calls_once():
    calls = []

    def fun(x: ir.RV, *, enabled: bool) -> ir.RV:
        calls.append(enabled)
        return x * x

    x, cotangent = pi.constant(3.0), pi.constant(2.0)
    derivative = pi.vjp(fun)
    result = derivative(cotangent, x, enabled=True)
    assert_type(result, ir.RV)
    assert calls == [True]
    assert result.op == ir.Add()
    assert [parent.op for parent in result.parents] == [ir.VJP(ir.Mul(), 0), ir.VJP(ir.Mul(), 1)]
    assert all(parent.parents == (cotangent, x, x) for parent in result.parents)
    assert not any(
        isinstance(node.op, (AbstractOp, ir.Composite)) for node in dag.upstream_nodes(result)
    )
    assert getattr(derivative, "__name__") == "vjp_fun"
    if TYPE_CHECKING:
        derivative(cotangent, x, enabled="bad")  # type: ignore


def test_array_identity_returns_cotangent():
    x, cotangent = pi.constant([1.0, 2.0]), pi.constant([3.0, 4.0])
    assert pi.vjp(lambda x: x)(cotangent, x) is cotangent


def test_identity_preserves_output_interface_without_copying_random_cotangent():
    x = pi.constant(2.0)
    cotangent = ir.RV(ir.Normal(), pi.constant(0.0), pi.constant(1.0))
    result = pi.vjp(lambda x: x)(cotangent, x)
    assert result.op == ir.Identity()
    assert result.parents == (cotangent,)
    assert result.interface is x.interface


def test_unused_array_argument_returns_zeros():
    result = pi.vjp(lambda x: pi.constant([1.0, 2.0]))(
        pi.constant([3.0, 4.0]), pi.constant([0.0, 0.0, 0.0])
    )
    assert result.op == ir.Constant([0.0, 0.0, 0.0])


def test_argnum_does_not_count_cotangent():
    x = pi.constant(2.0)
    result = pi.vjp(lambda a, b: a * b, 1)(pi.constant(3.0), x, x)
    assert result.op == ir.VJP(ir.Mul(), 1)


def test_cotangent_does_not_steal_function_keyword():
    x = pi.constant(2.0)

    def fun(a, *, cotangent):
        return a * cotangent

    result = pi.vjp(fun)(x, x, cotangent=x)
    assert result.op == ir.VJP(ir.Mul(), 0)
    assert result.parents == (x, x, x)


def test_rejects_non_rv_cotangent():
    with pytest.raises(TypeError, match="cotangent must be an RV"):
        pi.vjp(lambda x: x)(1.0, pi.constant(2.0))  # type: ignore


@pytest.mark.parametrize("fun", [lambda x: x, lambda x: pi.constant([1.0, 2.0])])
def test_rejects_wrong_cotangent_shape_even_for_identity_or_unused_argument(fun):
    with pytest.raises(ValueError, match="cotangent shape"):
        pi.vjp(fun)(pi.constant(1.0), pi.constant([0.0, 1.0]))


def test_rejects_internal_randomness():
    with pytest.raises(ValueError, match="random RVs"):
        pi.vjp(lambda x: x + pi.normal(0, 1))(pi.constant(1.0), pi.constant(2.0))
