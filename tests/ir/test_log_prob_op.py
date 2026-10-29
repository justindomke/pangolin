from typing import Any, assert_type

import pytest

from pangolin import ir


def test_log_prob_is_public_and_preserves_base_op_type():
    base = ir.Normal()
    op = ir.LogProb(base)

    assert "LogProb" in ir.__all__
    assert op.base_op is base
    assert_type(op, ir.LogProb[ir.Normal])
    assert_type(op.base_op, ir.Normal)


@pytest.mark.parametrize("base", [ir.Normal(), ir.Bernoulli(), ir.Categorical()])
def test_log_prob_is_deterministic_for_continuous_and_discrete_distributions(base: ir.Op):
    op = ir.LogProb(base)

    assert op.random is False
    assert op.discrete is None


@pytest.mark.parametrize("base", [None, "normal", 3, ir.Normal])
def test_log_prob_rejects_non_ops(base: Any):
    with pytest.raises(TypeError, match="base_op must be an Op"):
        ir.LogProb(base)


@pytest.mark.parametrize(
    "base",
    [
        ir.Constant(0),
        ir.Exp(),
        ir.VMap(ir.Exp(), (0,)),
        ir.Composite(1, (ir.Exp(),), ((0,),)),
        ir.Scan(ir.Exp(), 3, ()),
        ir.LogProb(ir.Normal()),
    ],
)
def test_log_prob_rejects_deterministic_ops_including_containers(base: ir.Op):
    with pytest.raises(ValueError, match="base_op must be random"):
        ir.LogProb(base)


@pytest.mark.parametrize(
    ("base", "value_shape", "parent_shapes"),
    [
        (ir.Normal(), (), ((), ())),
        (ir.Bernoulli(), (), ((),)),
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
def test_log_prob_shape_is_scalar(
    base: ir.Op, value_shape: ir.Shape, parent_shapes: tuple[ir.Shape, ...]
):
    assert ir.LogProb(base).get_shape(value_shape, *parent_shapes) == ()


@pytest.mark.parametrize("value_shape", [(1,), (3,), (2, 3)])
def test_log_prob_does_not_broadcast_scalar_distributions(value_shape: ir.Shape):
    with pytest.raises(ValueError, match="value shape .* does not match base_op output shape"):
        ir.LogProb(ir.Normal()).get_shape(value_shape, (), ())


@pytest.mark.parametrize("value_shape", [(), (1,), (1, 3), (2, 3)])
def test_log_prob_requires_exact_array_value_shape(value_shape: ir.Shape):
    with pytest.raises(ValueError, match="value shape .* does not match base_op output shape"):
        ir.LogProb(ir.MultiNormal()).get_shape(value_shape, (3,), (3, 3))


@pytest.mark.parametrize("parent_shapes", [(), ((),), ((), (), ())])
def test_log_prob_validates_distribution_arity(parent_shapes: tuple[ir.Shape, ...]):
    with pytest.raises((TypeError, ValueError)):
        ir.LogProb(ir.Normal()).get_shape((), *parent_shapes)


def test_log_prob_requires_an_evaluation_value():
    op: Any = ir.LogProb(ir.Normal())
    with pytest.raises(TypeError):
        op.get_shape()


def test_log_prob_validates_distribution_parameter_shapes():
    with pytest.raises(ValueError):
        ir.LogProb(ir.MultiNormal()).get_shape((3,), (3,), (2, 2))


def test_log_prob_allows_a_distribution_without_parameters():
    class ParameterFreeDistribution(ir.Op):
        random: bool = True
        """True"""
        discrete: bool = False
        """False"""

        def get_shape(self) -> ir.Shape:
            return ()

    assert ir.LogProb(ParameterFreeDistribution()).get_shape(()) == ()


def test_log_prob_rv_preserves_value_and_parameter_identity():
    zero = ir.RV(ir.Constant(0))
    one = ir.RV(ir.Constant(1))
    mu = ir.RV(ir.Normal(), zero, one)
    value = ir.RV(ir.Normal(), mu, one)

    logp = ir.RV(ir.LogProb(value.op), value, *value.parents)

    assert logp.shape == ()
    assert logp.parents == (value, mu, one)
    assert logp.op.random is False


def test_log_prob_accepts_an_independent_evaluation_value():
    value = ir.RV(ir.Constant(2))
    mu = ir.RV(ir.Constant(0))
    sigma = ir.RV(ir.Constant(1))

    logp = ir.RV(ir.LogProb(ir.Normal()), value, mu, sigma)

    assert logp.parents == (value, mu, sigma)
    assert logp.shape == ()


def test_log_prob_validates_shape_during_rv_construction():
    value = ir.RV(ir.Constant([0, 1]))
    mu = ir.RV(ir.Constant(0))
    sigma = ir.RV(ir.Constant(1))

    with pytest.raises(ValueError, match="value shape"):
        ir.RV(ir.LogProb(ir.Normal()), value, mu, sigma)


def test_log_prob_can_be_composed_as_a_deterministic_op():
    op = ir.Composite(3, (ir.LogProb(ir.Normal()), ir.Exp()), ((0, 1, 2), (3,)))

    assert op.random is False
    assert op.get_shape((), (), ()) == ()


def test_mapping_log_prob_is_distinct_from_log_prob_of_mapped_distribution():
    individual = ir.VMap(ir.LogProb(ir.Normal()), (0, 0, None))
    joint = ir.LogProb(ir.VMap(ir.Normal(), (0, None)))

    assert individual.get_shape((3,), (3,), ()) == (3,)
    assert joint.get_shape((3,), (3,), ()) == ()


def test_log_prob_equality_hashing_and_representations():
    op = ir.LogProb(ir.VMap(ir.Normal(), (0, None)))
    equal = ir.LogProb(ir.VMap(ir.Normal(), [0, None]))
    different = ir.LogProb(ir.VMap(ir.Normal(), (None, 0)))

    assert op == equal
    assert hash(op) == hash(equal)
    assert op != different
    assert op != ir.LogProb(ir.Normal())
    assert op != ir.LogProb(ir.Bernoulli())
    assert op != op.base_op
    assert op != "log_prob"
    assert {op: "value"}[equal] == "value"
    assert repr(op) == "LogProb(VMap(Normal(), [0, None]))"
    assert str(op) == "log_prob(vmap(normal, [0, None]))"


def test_log_prob_equality_distinguishes_subclasses():
    class OtherLogProb(ir.LogProb[ir.Normal]):
        pass

    assert ir.LogProb(ir.Normal()) != OtherLogProb(ir.Normal())


def test_log_prob_is_immutable():
    op = ir.LogProb(ir.Normal())

    with pytest.raises(TypeError, match="Ops are immutable"):
        setattr(op, "base_op", ir.Bernoulli())
    with pytest.raises(TypeError, match="Ops are immutable"):
        delattr(op, "base_op")
