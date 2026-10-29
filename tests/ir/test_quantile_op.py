from typing import Any, assert_type

import pytest

from pangolin import ir


def test_public_and_preserves_base_op_type():
    base = ir.Normal()
    op = ir.Quantile(base)
    assert "Quantile" in ir.__all__
    assert op.base_op is base
    assert_type(op, ir.Quantile[ir.Normal])
    assert_type(op.base_op, ir.Normal)


@pytest.mark.parametrize("base", [ir.Normal(), ir.Poisson(), ir.Categorical()])
def test_deterministic_even_for_discrete_base(base: ir.Op):
    op = ir.Quantile(base)
    assert not op.random
    assert op.discrete is None


@pytest.mark.parametrize("base", [None, "normal", 3, ir.Normal])
def test_rejects_non_ops(base: Any):
    with pytest.raises(TypeError, match="base_op must be an Op"):
        ir.Quantile(base)


@pytest.mark.parametrize(
    "base",
    [
        ir.Constant(0), ir.Exp(), ir.VMap(ir.Exp(), (0,)),
        ir.Composite(1, (ir.Exp(),), ((0,),)), ir.Scan(ir.Exp(), 3, ()),
        ir.LogProb(ir.Normal()), ir.CDF(ir.Normal()), ir.Quantile(ir.Normal()),
    ],
)
def test_rejects_deterministic_ops(base: ir.Op):
    with pytest.raises(ValueError, match="base_op must be random"):
        ir.Quantile(base)


@pytest.mark.parametrize(
    ("base", "parent_shapes"),
    [
        (ir.Normal(), ((), ())),
        (ir.Poisson(), ((),)),
        (ir.Categorical(), ((4,),)),
        (ir.Composite(2, (ir.Exp(), ir.Normal()), ((0,), (2, 1))), ((), ())),
        (ir.Truncated(ir.Normal()), ((), (), (), ())),
        (
            ir.Bijected(
                ir.Normal(),
                ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),))),
            ),
            ((), ()),
        ),
    ],
)
def test_accepts_scalar_distributions(base: ir.Op, parent_shapes: tuple[ir.Shape, ...]):
    assert ir.Quantile(base).get_shape((), *parent_shapes) == ()


def test_scalar_output_can_have_array_parameters_and_internal_vmap():
    base = ir.Composite(
        2, (ir.VMap(ir.Exp(), (0,)), ir.Sum(0), ir.Normal()), ((0,), (2,), (3, 1))
    )
    assert ir.Quantile(base).get_shape((), (3,), ()) == ()


def test_accepts_custom_distribution_without_backend_capabilities():
    class ParameterFreeDistribution(ir.Op):
        random: bool = True
        """True"""
        discrete: bool = False
        """False"""

        def get_shape(self) -> ir.Shape:
            return ()

    assert ir.Quantile(ParameterFreeDistribution()).get_shape(()) == ()


@pytest.mark.parametrize(
    ("base", "parent_shapes"),
    [
        (ir.MultiNormal(), ((2,), (2, 2))),
        (ir.Dirichlet(), ((3,),)),
        (ir.VMap(ir.Normal(), (0, None)), ((1,), ())),
        (ir.Scan(ir.Normal(), 3, (None,)), ((), ())),
        (ir.Composite(2, (ir.VMap(ir.Normal(), (0, None)),), ((0, 1),)), ((3,), ())),
    ],
)
def test_rejects_array_outputs_at_shape_validation(base: ir.Op, parent_shapes: tuple[ir.Shape, ...]):
    op = ir.Quantile(base)
    with pytest.raises(ValueError, match="base_op must have scalar output"):
        op.get_shape((), *parent_shapes)


@pytest.mark.parametrize("shape", [(1,), (3,), (2, 3)])
def test_probability_must_be_scalar_without_broadcasting(shape: ir.Shape):
    with pytest.raises(ValueError, match="probability must be scalar"):
        ir.Quantile(ir.Normal()).get_shape(shape, (), ())


@pytest.mark.parametrize("shapes", [(), ((),), ((), ()), ((),) * 4])
def test_validates_number_of_parents(shapes: tuple[ir.Shape, ...]):
    with pytest.raises(TypeError):
        ir.Quantile(ir.Normal()).get_shape(*shapes)


def test_validates_base_parameter_shapes():
    with pytest.raises(ValueError):
        ir.Quantile(ir.Normal()).get_shape((), (3,), ())


def test_rv_preserves_parent_order_and_identity_with_random_probability():
    zero = ir.RV(ir.Constant(0))
    one = ir.RV(ir.Constant(1))
    probability = ir.RV(ir.Uniform(), zero, one)
    location = ir.RV(ir.Normal(), zero, one)
    rv = ir.RV(ir.Quantile(ir.Normal()), probability, location, one)
    assert rv.shape == ()
    assert all(actual is expected for actual, expected in zip(rv.parents, (probability, location, one)))
    assert not rv.op.random


@pytest.mark.parametrize("probability", [0, 1, 0.5, -1, 2, float("nan"), float("inf")])
def test_probability_values_are_not_checked_by_ir(probability: float):
    parents = tuple(ir.RV(ir.Constant(value)) for value in (probability, 0, 1))
    assert ir.RV(ir.Quantile(ir.Normal()), *parents).shape == ()


def test_rv_construction_rejects_vector_probability():
    parents = tuple(ir.RV(ir.Constant(value)) for value in ([0.5], 0, 1))
    with pytest.raises(ValueError, match="probability must be scalar"):
        ir.RV(ir.Quantile(ir.Normal()), *parents)


def test_composition_with_other_ops():
    op = ir.Quantile(ir.Normal())
    assert ir.VMap(op, (0, 0, None)).get_shape((3,), (3,), ()) == (3,)
    assert ir.Composite(3, (op,), ((0, 1, 2),)).get_shape((), (), ()) == ()
    assert ir.Scan(op, 3, (None, None)).get_shape((), (), ()) == (3,)
    assert ir.VJP(op, 0).get_shape((), (), (), ()) == ()


def test_equality_hash_and_representations():
    op = ir.Quantile(ir.Normal())
    equal = ir.Quantile(ir.Normal())
    assert op == equal
    assert hash(op) == hash(equal)
    assert {op: "value"}[equal] == "value"
    assert op != ir.Quantile(ir.NormalPrec())
    assert op != ir.CDF(ir.Normal())
    assert op != ir.Normal()
    assert op != "quantile"
    assert repr(op) == "Quantile(Normal())"
    assert str(op) == "quantile(normal)"


def test_equality_distinguishes_subclasses():
    class OtherQuantile(ir.Quantile[ir.Normal]):
        pass

    assert ir.Quantile(ir.Normal()) != OtherQuantile(ir.Normal())


def test_immutable():
    op = ir.Quantile(ir.Normal())
    with pytest.raises(TypeError):
        setattr(op, "base_op", ir.Poisson())
    with pytest.raises(TypeError):
        delattr(op, "base_op")
