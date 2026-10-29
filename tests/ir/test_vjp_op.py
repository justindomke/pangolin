from typing import Any, assert_type

import numpy as np
import pytest

from pangolin import ir


def test_vjp_is_public_and_preserves_base_op_type():
    base = ir.Mul()
    op = ir.VJP(base, 1)

    assert "VJP" in ir.__all__
    assert op.base_op is base
    assert_type(op, ir.VJP[ir.Mul])
    assert_type(op.base_op, ir.Mul)
    assert op.argnum == 1
    assert op.random is False
    assert op.discrete is None


@pytest.mark.parametrize("base", [None, "mul", 3, ir.Mul])
def test_non_op_is_rejected(base: Any):
    with pytest.raises(TypeError, match="base_op must be an Op"):
        ir.VJP(base, 0)


@pytest.mark.parametrize(
    "base",
    [
        ir.Normal(),
        ir.Bernoulli(),
        ir.VMap(ir.Normal(), (0, None)),
        ir.Scan(ir.Normal(), 3, (None,)),
        ir.Composite(2, (ir.Normal(),), ((0, 1),)),
        ir.Bijected(
            ir.Normal(),
            ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),))),
        ),
    ],
)
def test_random_base_is_rejected(base: ir.Op):
    with pytest.raises(ValueError, match="base_op must be deterministic"):
        ir.VJP(base, 0)


@pytest.mark.parametrize("argnum", [True, False, np.bool_(True), 0.0, "0", None])
def test_noninteger_argnum_is_rejected(argnum: Any):
    with pytest.raises(TypeError, match="argnum must be an integer"):
        ir.VJP(ir.Mul(), argnum)


def test_negative_argnum_is_rejected():
    with pytest.raises(ValueError, match="argnum must be nonnegative"):
        ir.VJP(ir.Mul(), -1)


def test_numpy_integer_is_normalized():
    argnum: Any = np.int64(1)
    op = ir.VJP(ir.Mul(), argnum)
    assert type(op.argnum) is int
    assert op == ir.VJP(ir.Mul(), 1)


@pytest.mark.parametrize(
    ("base", "argnum", "shapes", "expected"),
    [
        (ir.Mul(), 0, ((), (), ()), ()),
        (ir.Mul(), 1, ((), (), ()), ()),
        (ir.Matmul(), 0, ((2, 4), (2, 3), (3, 4)), (2, 3)),
        (ir.Matmul(), 1, ((2, 4), (2, 3), (3, 4)), (3, 4)),
        (ir.Sum(1), 0, ((2,), (2, 3)), (2, 3)),
        (ir.VMap(ir.Mul(), (0, None)), 0, ((3,), (3,), ()), (3,)),
        (ir.VMap(ir.Mul(), (0, None)), 1, ((3,), (3,), ()), ()),
        (ir.Scan(ir.Add(), 3, (0,)), 0, ((3,), (), (3,)), ()),
        (ir.Scan(ir.Add(), 3, (0,)), 1, ((3,), (), (3,)), (3,)),
        (ir.Composite(2, (ir.Mul(), ir.Exp()), ((0, 1), (2,))), 1, ((), (), ()), ()),
        (ir.LogProb(ir.Normal()), 0, ((), (), (), ()), ()),
        (ir.LogProb(ir.Normal()), 2, ((), (), (), ()), ()),
        (ir.VJP(ir.Mul(), 0), 1, ((), (), (), ()), ()),
        (ir.Index(), 0, ((), (3,), ()), (3,)),
    ],
)
def test_output_shape_matches_selected_input(
    base: ir.Op, argnum: int, shapes: tuple[ir.Shape, ...], expected: ir.Shape
):
    op = ir.VJP(base, argnum)
    assert op.get_shape(*shapes) == expected
    assert not op.random


@pytest.mark.parametrize("argnum", [2, 3, 100])
def test_argnum_counts_only_base_inputs(argnum: int):
    with pytest.raises(ValueError, match="argnum must index an input"):
        ir.VJP(ir.Mul(), argnum).get_shape((), (), ())


def test_parameter_free_base_has_no_differentiation_input():
    with pytest.raises(ValueError, match="argnum must index an input"):
        ir.VJP(ir.Constant(1.0), 0).get_shape(())


def test_missing_cotangent():
    op: Any = ir.VJP(ir.Mul(), 0)
    with pytest.raises(TypeError, match="cotangent_shape"):
        op.get_shape()


@pytest.mark.parametrize("shapes", [((),), ((), ()), ((), (), (), ())])
def test_base_arity_is_validated(shapes: tuple[ir.Shape, ...]):
    with pytest.raises(TypeError):
        ir.VJP(ir.Mul(), 0).get_shape(*shapes)


def test_base_input_shapes_are_validated():
    with pytest.raises(ValueError):
        ir.VJP(ir.Matmul(), 0).get_shape((2, 5), (2, 3), (4, 5))


@pytest.mark.parametrize("cotangent_shape", [(), (4,), (1, 4), (4, 2)])
def test_cotangent_requires_exact_output_shape(cotangent_shape: ir.Shape):
    with pytest.raises(ValueError, match="cotangent shape .* does not match"):
        ir.VJP(ir.Matmul(), 0).get_shape(cotangent_shape, (2, 3), (3, 4))


def test_scalar_output_does_not_accept_length_one_cotangent():
    with pytest.raises(ValueError, match="cotangent shape .* does not match"):
        ir.VJP(ir.Mul(), 0).get_shape((1,), (), ())


def test_rv_parents_keep_their_identity_and_can_be_random():
    one = ir.RV(ir.Constant(1.0))
    random = ir.RV(ir.Normal(), one, one)
    result = ir.RV(ir.VJP(ir.Mul(), 0), one, random, random)

    assert result.parents == (one, random, random)
    assert result.parents[1] is result.parents[2]
    assert result.shape == ()
    assert not result.op.random


def test_vjp_can_be_nested_inside_containers():
    assert ir.VMap(ir.VJP(ir.Mul(), 0), (0, 0, None)).get_shape((3,), (3,), ()) == (3,)
    assert ir.Scan(ir.VJP(ir.Mul(), 0), 3, (None, None)).get_shape((), (), ()) == (3,)
    assert ir.Composite(3, (ir.VJP(ir.Mul(), 0),), ((0, 1, 2),)).get_shape((), (), ()) == ()


def test_equality_hash_and_printing():
    op = ir.VJP(ir.Mul(), 0)
    same = ir.VJP(ir.Mul(), 0)
    assert op == same
    assert hash(op) == hash(same)
    assert {op: "found"}[same] == "found"
    assert op != ir.VJP(ir.Mul(), 1)
    assert op != ir.VJP(ir.Add(), 0)
    assert op != ir.Mul()
    assert op != "vjp(mul, 0)"
    assert repr(op) == "VJP(Mul(), 0)"
    assert str(op) == "vjp(mul, 0)"


def test_subclass_is_not_equal():
    class OtherVJP(ir.VJP):
        pass

    assert OtherVJP(ir.Mul(), 0) != ir.VJP(ir.Mul(), 0)


@pytest.mark.parametrize("attribute", ["base_op", "argnum"])
def test_configuration_is_immutable(attribute: str):
    op = ir.VJP(ir.Mul(), 0)
    with pytest.raises(TypeError):
        setattr(op, attribute, None)
    with pytest.raises(TypeError):
        delattr(op, attribute)
