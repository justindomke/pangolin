from typing import Any, assert_type

import pytest

from pangolin import ir


def test_public_and_preserves_base_op_type():
    base = ir.Normal()
    op = ir.CDF(base)
    assert "CDF" in ir.__all__
    assert op.base_op is base
    assert_type(op, ir.CDF[ir.Normal])
    assert_type(op.base_op, ir.Normal)


@pytest.mark.parametrize("base", [ir.Normal(), ir.Poisson(), ir.Categorical()])
def test_deterministic_even_for_discrete_base(base: ir.Op):
    op = ir.CDF(base)
    assert not op.random
    assert op.discrete is None


@pytest.mark.parametrize("base", [None, "normal", 3, ir.Normal])
def test_rejects_non_ops(base: Any):
    with pytest.raises(TypeError, match="base_op must be an Op"):
        ir.CDF(base)


@pytest.mark.parametrize(
    "base",
    [
        ir.Constant(0),
        ir.Exp(),
        ir.VMap(ir.Exp(), (0,)),
        ir.Composite(1, (ir.Exp(),), ((0,),)),
        ir.Scan(ir.Exp(), 3, ()),
        ir.LogProb(ir.Normal()),
        ir.CDF(ir.Normal()),
    ],
)
def test_rejects_deterministic_ops(base: ir.Op):
    with pytest.raises(ValueError, match="base_op must be random"):
        ir.CDF(base)


@pytest.mark.parametrize(
    ("base", "value_shape", "parent_shapes"),
    [
        (ir.Normal(), (), ((), ())),
        (ir.Poisson(), (), ((),)),
        (ir.Categorical(), (), ((4,),)),
        (ir.MultiNormal(), (3,), ((3,), (3, 3))),
        (ir.Dirichlet(), (3,), ((3,),)),
        (ir.VMap(ir.Normal(), (0, None)), (3,), ((3,), ())),
        (ir.VMap(ir.VMap(ir.Normal(), (0, None)), (0, None)), (2, 3), ((2, 3), ())),
        (ir.Composite(2, (ir.Exp(), ir.Normal()), ((0,), (2, 1))), (), ((), ())),
        (ir.Scan(ir.Normal(), 3, (None,)), (3,), ((), ())),
        (
            ir.Bijected(
                ir.Normal(),
                ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),))),
            ),
            (),
            ((), ()),
        ),
    ],
)
def test_joint_scalar_shape_even_when_backend_support_is_unavailable(
    base: ir.Op, value_shape: ir.Shape, parent_shapes: tuple[ir.Shape, ...]
):
    assert ir.CDF(base).get_shape(value_shape, *parent_shapes) == ()


@pytest.mark.parametrize(
    ("base", "value_shape", "parent_shapes"),
    [
        (ir.Normal(), (1,), ((), ())),
        (ir.Normal(), (), ((3,), ())),
        (ir.Poisson(), (), ((3,),)),
        (ir.MultiNormal(), (), ((3,), (3, 3))),
        (ir.VMap(ir.Normal(), (0, None)), (1,), ((3,), ())),
    ],
)
def test_no_implicit_broadcasting(
    base: ir.Op, value_shape: ir.Shape, parent_shapes: tuple[ir.Shape, ...]
):
    with pytest.raises(ValueError):
        ir.CDF(base).get_shape(value_shape, *parent_shapes)


@pytest.mark.parametrize("shapes", [(), ((),), ((), ()), ((), (), (), ())])
def test_validates_number_of_parents(shapes: tuple[ir.Shape, ...]):
    with pytest.raises(TypeError):
        ir.CDF(ir.Normal()).get_shape(*shapes)


def test_mapping_cdf_is_not_joint_cdf():
    assert ir.VMap(ir.CDF(ir.Normal()), (0, 0, None)).get_shape((3,), (3,), ()) == (3,)
    assert ir.CDF(ir.VMap(ir.Normal(), (0, None))).get_shape((3,), (3,), ()) == ()


def test_equality_hash_and_printing():
    op = ir.CDF(ir.Normal())
    assert op == ir.CDF(ir.Normal())
    assert hash(op) == hash(ir.CDF(ir.Normal()))
    assert op != ir.CDF(ir.NormalPrec())
    assert op != ir.LogProb(ir.Normal())
    assert op != ir.Normal()
    assert repr(op) == "CDF(Normal())"
    assert str(op) == "cdf(normal)"


def test_immutable():
    op = ir.CDF(ir.Normal())
    with pytest.raises(TypeError):
        setattr(op, "base_op", ir.Poisson())
    with pytest.raises(TypeError):
        delattr(op, "base_op")


def test_rv_preserves_parent_identity_and_accepts_fractional_discrete_threshold():
    rate = ir.RV(ir.Exponential(), ir.RV(ir.Constant(1.0)))
    value = ir.RV(ir.Constant(1.5))
    probability = ir.RV(ir.CDF(ir.Poisson()), value, rate)
    assert probability.parents[0] is value
    assert probability.parents[1] is rate
    assert probability.shape == ()
    assert not probability.op.random
