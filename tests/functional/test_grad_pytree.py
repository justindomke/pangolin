"""Check tree gradients, differentiation boundaries, sharing, and runtime contracts."""

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, assert_type

import jax.tree_util
import pytest

from pangolin import dag, interface as pi, ir
from pangolin.functional.tracing import AbstractOp


class Pair(NamedTuple):
    left: ir.RV
    right: ir.RV


@partial(jax.tree_util.register_dataclass, data_fields=["value", "unused"], meta_fields=["label"])
@dataclass(frozen=True)
class Tagged:
    value: ir.RV
    unused: ir.RV
    label: str


def test_nested_tree_preserves_structure_shapes_and_identity_derivative():
    x, y = pi.constant(2.0), pi.constant([3.0, 4.0])
    tree = {"used": (x,), "unused": [y, {"empty": ()}]}
    result = pi.grad(lambda p: p["used"][0])(tree)
    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(tree)
    assert result["used"][0].op == ir.Identity()
    assert result["used"][0].parents[0].op == ir.Constant(1.0)
    assert result["unused"][0].op == ir.Constant([0.0, 0.0])
    assert [r.shape for r in jax.tree_util.tree_leaves(result)] == [(2,), ()]


def test_namedtuple_return_is_preserved_at_runtime():
    def product(p: Pair, *, enabled: bool) -> ir.RV:
        return p.left * p.right if enabled else p.left

    x, y = pi.constant(2.0), pi.constant(3.0)
    result = pi.grad(product)(Pair(x, y), enabled=True)
    assert_type(result, Any)
    assert type(result) is Pair
    assert result.left.op == ir.VJP(ir.Mul(), 0)
    assert result.right.op == ir.VJP(ir.Mul(), 1)
    assert result.left.parents[1:] == result.right.parents[1:] == (x, y)


def test_registered_custom_tree_return_is_preserved_at_runtime():
    def square(p: Tagged) -> ir.RV:
        assert p.label == "fixed metadata"
        return p.value * p.value

    result = pi.grad(square)(Tagged(pi.constant(2.0), pi.constant([1.0, 2.0]), "fixed metadata"))
    assert_type(result, Any)
    assert type(result) is Tagged
    assert result.value.op == ir.Add()
    assert result.unused.op == ir.Constant([0.0, 0.0])
    assert result.label == "fixed metadata"


def test_one_call_and_fresh_abstract_for_each_occurrence():
    x = pi.constant(2.0)
    calls = []

    def product(p):
        calls.append(p)
        return p[0] * p[1]

    result = pi.grad(product)([x, x])
    assert len(calls) == 1
    a, b = calls[0]
    assert a is not b and a is not x and b is not x
    assert isinstance(a.op, AbstractOp) and isinstance(b.op, AbstractOp)
    assert a.shape == b.shape == ()
    assert a.interface is b.interface is x.interface
    assert [r.op for r in result] == [ir.VJP(ir.Mul(), 0), ir.VJP(ir.Mul(), 1)]
    assert all(r.parents[1:] == (x, x) for r in result)
    assert not any(isinstance(n.op, AbstractOp) for n in dag.upstream_nodes(result))


def test_reconnected_forward_intermediates_are_shared_across_gradient_leaves():
    x, y = pi.constant(2.0), pi.constant(3.0)

    def square_product(p):
        shared = p[0] * p[1]
        return shared * shared

    dx, dy = pi.grad(square_product)((x, y))
    assert dx.op == ir.VJP(ir.Mul(), 0)
    assert dy.op == ir.VJP(ir.Mul(), 1)
    assert dx.parents[0] is dy.parents[0]
    accumulated = dx.parents[0]
    assert accumulated.op == ir.Add()
    shared = accumulated.parents[0].parents[1]
    assert shared.op == ir.Mul() and shared.parents == (x, y)
    assert all(n.parents[1] is shared and n.parents[2] is shared for n in accumulated.parents)
    assert not any(isinstance(n.op, (AbstractOp, ir.Composite)) for n in dag.upstream_nodes([dx, dy]))


def test_selected_leaf_interfaces_and_output_interface_are_distinct_concerns():
    first, second, output_interface = ir.RVInterface(), ir.RVInterface(), ir.RVInterface()
    x = ir.RV(ir.Constant(2.0), interface=first)
    y = ir.RV(ir.Constant(3.0), interface=second)
    calls = []

    def fun(p):
        calls.append((p[0].interface, p[1].interface))
        return ir.RV(ir.Exp(), p[0], interface=output_interface)

    result = pi.grad(fun)([x, y])
    assert calls == [(first, second)]
    assert all(r.interface is output_interface for r in result)
    assert result[0].parents[1] is x
    assert result[1].op == ir.Constant(0.0)


@pytest.mark.parametrize("tree", [[], (), {}, {"a": [], "b": ((), {})}])
def test_empty_trees_still_call_function_once(tree):
    calls = []

    def fun(p):
        calls.append(p)
        return pi.constant(1.0)

    assert pi.grad(fun)(tree) == tree
    assert calls == [tree]


@pytest.mark.parametrize("tree", [None, [None], {"x": 2.0}, [pi.constant(0), ["bad"]]])
def test_invalid_leaf_rejected_before_function_call(tree):
    calls = []

    def fun(p):
        calls.append(p)
        return pi.constant(1.0)

    with pytest.raises(TypeError, match="leaves are RV"):
        pi.grad(fun)(tree)
    assert not calls


def test_empty_and_unused_trees_do_not_bypass_randomness_or_output_validation():
    for tree in [[], [pi.constant(1.0)]]:
        with pytest.raises(ValueError, match="random RVs"):
            pi.grad(lambda p: pi.normal(0, 1) + 1)(tree)
        with pytest.raises(ValueError, match="scalar"):
            pi.grad(lambda p: pi.constant([1.0, 2.0]))(tree)
        with pytest.raises(TypeError):
            # Deliberately violate the scalar-output contract to exercise runtime validation.
            pi.grad(lambda p: {"output": pi.constant(1.0)})(tree)  # type: ignore


def test_existing_random_output_and_unused_new_random_nodes_are_allowed():
    captured = pi.normal(0, 1)

    def fun(p):
        pi.normal(0, 1)
        return captured

    result = pi.grad(fun)([pi.constant([1.0, 2.0]), pi.constant(3.0)])
    assert [r.op for r in result] == [ir.Constant([0.0, 0.0]), ir.Constant(0.0)]


def test_nonzero_argnum_preserves_tree_and_static_kwargs():
    def fun(scale: float, p: dict[str, ir.RV], *, enabled: bool) -> ir.RV:
        return scale * p["x"] if enabled else p["x"]

    result = pi.grad(fun, 1)(2.0, {"x": pi.constant(3.0)}, enabled=True)
    assert_type(result, Any)
    assert result["x"].op == ir.VJP(ir.Mul(), 1)


def test_vjp_is_not_silently_generalized():
    with pytest.raises(TypeError, match="selected argument must be an RV"):
        pi.vjp(lambda p: p[0])(pi.constant(1.0), [pi.constant(2.0)])


def test_grad_output_is_still_a_single_scalar_rv():
    p = [pi.constant(2.0), pi.constant(3.0)]
    with pytest.raises(TypeError):
        pi.grad(lambda p: p)(p)
    with pytest.raises(TypeError):
        pi.grad(pi.grad(lambda p: p[0] * p[1]))(p)
