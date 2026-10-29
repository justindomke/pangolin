from typing import Any, assert_type

import pytest

from pangolin import ir


def test_public_and_preserves_base_op_type():
    base = ir.Normal()
    op = ir.Truncated(base)
    assert "Truncated" in ir.__all__
    assert op.base_op is base
    assert_type(op, ir.Truncated[ir.Normal])
    assert_type(op.base_op, ir.Normal)
    assert op.random is True
    assert op.discrete is False


@pytest.mark.parametrize("base", [None, "normal", 3, ir.Normal])
def test_rejects_non_ops(base: Any):
    with pytest.raises(TypeError, match="base_op must be an Op"):
        ir.Truncated(base)


@pytest.mark.parametrize(
    "base",
    [
        ir.Constant(0),
        ir.Exp(),
        ir.VMap(ir.Exp(), (0,)),
        ir.Composite(1, (ir.Exp(),), ((0,),)),
        ir.LogProb(ir.Normal()),
    ],
)
def test_rejects_deterministic_ops(base: ir.Op):
    with pytest.raises(ValueError, match="base_op must be random"):
        ir.Truncated(base)


@pytest.mark.parametrize(
    "base",
    [
        ir.Bernoulli(),
        ir.Categorical(),
        ir.Poisson(),
        ir.VMap(ir.Bernoulli(), (0,)),
        ir.Composite(1, (ir.Exp(), ir.Poisson()), ((0,), (1,))),
    ],
)
def test_rejects_discrete_ops(base: ir.Op):
    with pytest.raises(ValueError, match="base_op must be continuous"):
        ir.Truncated(base)


@pytest.mark.parametrize(
    ("base", "parent_shapes"),
    [
        (ir.Normal(), ((), ())),
        (ir.Exponential(), ((),)),
        (ir.StudentT(), ((), (), ())),
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
def test_accepts_scalar_continuous_distributions(base: ir.Op, parent_shapes: tuple[ir.Shape, ...]):
    assert ir.Truncated(base).get_shape((), (), *parent_shapes) == ()


def test_scalar_output_can_have_array_parameters_and_internal_vmap():
    base = ir.Composite(
        2,
        (ir.VMap(ir.Exp(), (0,)), ir.Sum(0), ir.Normal()),
        ((0,), (2,), (3, 1)),
    )
    assert ir.Truncated(base).get_shape((), (), (3,), ()) == ()


def test_accepts_custom_distribution_without_backend_capabilities():
    class ParameterFreeDistribution(ir.Op):
        random: bool = True
        """True"""
        discrete: bool = False
        """False"""

        def get_shape(self) -> ir.Shape:
            return ()

    for lower, upper in ((True, True), (True, False), (False, True), (False, False)):
        op = ir.Truncated(ParameterFreeDistribution(), lower=lower, upper=upper)
        shapes = ((),) * (int(lower) + int(upper))
        assert op.get_shape(*shapes) == ()
        if shapes:
            with pytest.raises(TypeError, match="bound shapes"):
                op.get_shape(*shapes[:-1])


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
def test_rejects_array_outputs_at_shape_validation(
    base: ir.Op, parent_shapes: tuple[ir.Shape, ...]
):
    op = ir.Truncated(base)
    with pytest.raises(ValueError, match="base_op must have scalar output"):
        op.get_shape((), (), *parent_shapes)


@pytest.mark.parametrize("lower, upper", [((1,), ()), ((), (1,)), ((3,), (3,)), ((2, 3), ())])
def test_bounds_must_be_scalar_without_broadcasting(lower: ir.Shape, upper: ir.Shape):
    with pytest.raises(ValueError, match="truncation bounds must be scalar"):
        ir.Truncated(ir.Normal()).get_shape(lower, upper, (), ())


@pytest.mark.parametrize("shapes", [(), ((),), ((), ()), ((), (), ()), ((),) * 5])
def test_validates_number_of_parents(shapes: tuple[ir.Shape, ...]):
    with pytest.raises(TypeError):
        ir.Truncated(ir.Normal()).get_shape(*shapes)


def test_validates_base_parameter_shapes():
    with pytest.raises(ValueError):
        ir.Truncated(ir.Normal()).get_shape((), (), (3,), ())


def test_rv_preserves_parent_order_and_identity_with_random_bounds():
    zero = ir.RV(ir.Constant(0))
    one = ir.RV(ir.Constant(1))
    lower = ir.RV(ir.Normal(), zero, one)
    upper = ir.RV(ir.Add(), lower, one)
    rv = ir.RV(ir.Truncated(ir.Normal()), lower, upper, zero, one)
    assert rv.shape == ()
    assert all(
        actual is expected for actual, expected in zip(rv.parents, (lower, upper, zero, one))
    )
    assert rv.op.random
    assert not rv.op.discrete


@pytest.mark.parametrize(
    "lower, upper", [(-float("inf"), float("inf")), (1, 0), (0, 0), (float("nan"), 1), (2, 3)]
)
def test_bound_values_and_interval_mass_are_not_checked_by_ir(lower: float, upper: float):
    parents = tuple(ir.RV(ir.Constant(value)) for value in (lower, upper, 0, 1))
    rv = ir.RV(ir.Truncated(ir.Uniform()), *parents)
    assert rv.shape == ()


def test_rv_construction_rejects_vector_bounds():
    parents = tuple(ir.RV(ir.Constant(value)) for value in ([0], 1, 0, 1))
    with pytest.raises(ValueError, match="truncation bounds must be scalar"):
        ir.RV(ir.Truncated(ir.Normal()), *parents)


def test_can_be_mapped_and_used_by_other_wrappers():
    op = ir.Truncated(ir.Normal())
    assert ir.VMap(op, (None, None, 0, None)).get_shape((), (), (3,), ()) == (3,)
    assert ir.Composite(4, (op,), ((0, 1, 2, 3),)).get_shape((), (), (), ()) == ()
    assert ir.Scan(op, 3, (None, None, None)).get_shape((), (), (), ()) == (3,)
    assert ir.LogProb(op).get_shape((), (), (), (), ()) == ()
    assert ir.CDF(op).get_shape((), (), (), (), ()) == ()


def test_equality_hash_and_representations():
    op = ir.Truncated(ir.Normal())
    equal = ir.Truncated(ir.Normal())
    assert op == equal
    assert hash(op) == hash(equal)
    assert {op: "value"}[equal] == "value"
    assert op != ir.Truncated(ir.NormalPrec())
    assert op != ir.Normal()
    assert op != "truncated"
    assert repr(op) == "Truncated(Normal())"
    assert str(op) == "truncated(normal)"
    assert repr(ir.Truncated(op)) == "Truncated(Truncated(Normal()))"


def test_equality_distinguishes_subclasses():
    class OtherTruncated(ir.Truncated[ir.Normal]):
        pass

    assert ir.Truncated(ir.Normal()) != OtherTruncated(ir.Normal())


def test_immutable():
    op = ir.Truncated(ir.Normal())
    with pytest.raises(TypeError, match="Ops are immutable"):
        setattr(op, "base_op", ir.Exponential())
    with pytest.raises(TypeError, match="Ops are immutable"):
        delattr(op, "base_op")


@pytest.mark.parametrize(
    "lower, upper", [(True, True), (True, False), (False, True), (False, False)]
)
def test_optional_bounds_parent_layout_and_wrappers(lower: bool, upper: bool):
    op = ir.Truncated(ir.Normal(), lower=lower, upper=upper)
    assert op.lower is lower
    assert op.upper is upper
    lo, hi, mu, sigma = (ir.RV(ir.Constant(v)) for v in (-1, 2, 0, 1))
    bounds = ((lo,) if lower else ()) + ((hi,) if upper else ())
    parents = (*bounds, mu, sigma)
    shapes = ((),) * len(parents)
    rv = ir.RV(op, *parents)
    assert rv.shape == ()
    assert all(a is b for a, b in zip(rv.parents, parents, strict=True))
    assert ir.LogProb(op).get_shape((), *shapes) == ()
    assert ir.CDF(op).get_shape((), *shapes) == ()
    assert ir.Quantile(op).get_shape((), *shapes) == ()
    axes = (None,) * len(bounds) + (0, None)
    assert ir.VMap(op, axes).get_shape(*(((),) * len(bounds)), (3,), ()) == (3,)
    assert ir.Composite(len(parents), (op,), (tuple(range(len(parents))),)).get_shape(*shapes) == ()

    for count in range(len(parents) + 2):
        if count != len(parents):
            with pytest.raises(TypeError):
                op.get_shape(*(((),) * count))
    for i in range(len(bounds)):
        invalid = list(shapes)
        invalid[i] = (1,)
        with pytest.raises(ValueError, match="truncation bounds must be scalar"):
            op.get_shape(*invalid)

    equivalent = ir.Truncated(ir.Normal(), lower=lower, upper=upper)
    assert op == equivalent
    assert hash(op) == hash(equivalent)
    assert op != ir.Truncated(ir.Normal(), lower=not lower, upper=upper)
    assert op != ir.Truncated(ir.Normal(), lower=lower, upper=not upper)


@pytest.mark.parametrize("flag", ["lower", "upper"])
@pytest.mark.parametrize("value", [None, 0, 1, -1, 0.0, "False", ir.Constant(1)])
def test_bound_flags_require_booleans(flag: str, value: Any):
    with pytest.raises(TypeError, match="must be booleans"):
        ir.Truncated(ir.Normal(), **{flag: value})


@pytest.mark.parametrize("flag", ["lower", "upper"])
def test_bound_flags_are_immutable(flag: str):
    op = ir.Truncated(ir.Normal())
    with pytest.raises(TypeError, match="Ops are immutable"):
        setattr(op, flag, False)
    with pytest.raises(TypeError, match="Ops are immutable"):
        delattr(op, flag)


def test_optional_bound_representations():
    for flags, suffix in [
        ({"lower": False}, ", lower=False"),
        ({"upper": False}, ", upper=False"),
        ({"lower": False, "upper": False}, ", lower=False, upper=False"),
    ]:
        op = ir.Truncated(ir.Normal(), **flags)
        assert repr(op) == f"Truncated(Normal(){suffix})"
        assert str(op) == f"truncated(normal{suffix})"


def test_nested_one_sided_truncations():
    inner = ir.Truncated(ir.Normal(), upper=False)
    outer = ir.Truncated(inner, lower=False)
    assert outer.get_shape((), (), (), ()) == ()
    assert repr(outer) == "Truncated(Truncated(Normal(), upper=False), lower=False)"


def test_absent_bounds_do_not_consume_array_parameters():
    base = ir.Composite(2, (ir.Sum(0), ir.Normal()), ((0,), (2, 1)))
    assert ir.Truncated(base, lower=False, upper=False).get_shape((3,), ()) == ()
    assert ir.Truncated(base, lower=False).get_shape((), (3,), ()) == ()
    assert ir.Truncated(base, upper=False).get_shape((), (3,), ()) == ()


def test_no_bounds_still_requires_scalar_continuous_random_base():
    for base in (ir.Add(), ir.Poisson()):
        with pytest.raises(ValueError):
            ir.Truncated(base, lower=False, upper=False)
    with pytest.raises(ValueError, match="base_op must have scalar output"):
        ir.Truncated(ir.MultiNormal(), lower=False, upper=False).get_shape((2,), (2, 2))
