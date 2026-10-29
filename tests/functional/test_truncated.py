from typing import TYPE_CHECKING, Any, assert_type

import pytest

from pangolin import dag, functional, interface as pi, ir
from pangolin.functional.truncating import truncate_op


def constant(value):
    return ir.RV(ir.Constant(value))


def normal(mean: ir.RV, scale: ir.RV) -> ir.RV:
    return ir.RV(ir.Normal(), mean, scale)


def test_public_export_and_primitive():
    lower, upper, mean, scale = map(constant, (-1, 2, 0, 1))
    fun = functional.truncated(normal, lo=lower, hi=upper)
    result = fun(mean, scale)
    assert "truncated" in functional.__all__
    assert_type(result, ir.RV)
    assert result.op == ir.Truncated(ir.Normal())
    assert result.parents == (lower, upper, mean, scale)
    assert result.shape == ()
    assert result.op.random and not result.op.discrete
    assert result.interface is ir.NO_INTERFACE
    assert getattr(fun, "__name__") == "truncated_normal"


def test_preserves_positional_keyword_and_default_arguments():
    lower, upper, mean, scale = map(constant, (-1, 2, 0, 1))

    def distribution(mean: ir.RV, /, *, scale: ir.RV = scale) -> ir.RV:
        return normal(mean, scale)

    fun = functional.truncated(distribution, lower, upper)
    assert fun(mean).parents == (lower, upper, mean, scale)
    assert fun(mean, scale=upper).parents == (lower, upper, mean, upper)


def test_function_is_called_once_per_invocation_with_arguments_unchanged():
    lower, upper = constant(0), constant(1)
    parameters = {"mean": constant(0), "scale": [constant(1)]}
    calls = []

    def distribution(params, *, label):
        calls.append((params, label))
        return normal(params["mean"], params["scale"][0])

    fun = functional.truncated(distribution, lower, upper)
    assert calls == []
    first = fun(parameters, label="normal")
    second = fun(parameters, label="normal")
    assert len(calls) == 2
    assert all(params is parameters and label == "normal" for params, label in calls)
    assert first is not second
    assert first.parents == second.parents
    assert first.op == second.op


def test_does_not_steal_keywords_named_lo_hi_or_fun():
    low, high = constant(0), constant(1)

    def distribution(*, lo, hi, fun):
        assert fun == "normal"
        return normal(lo, hi)

    result = functional.truncated(distribution, low, high)(lo=low, hi=high, fun="normal")
    assert result.parents == (low, high, low, high)


def test_does_not_coerce_arguments_but_preserves_functions_own_coercion():
    low, high = constant(0), constant(1)
    seen = []

    def distribution(mean, scale):
        seen.append((mean, scale))
        return pi.normal(mean, scale)

    result = functional.truncated(distribution, low, high)(0.5, 1.0)
    assert seen == [(0.5, 1.0)]
    assert result.parents[2].op == ir.Constant(0.5)
    assert result.parents[3].op == ir.Constant(1.0)
    assert result.interface is pi.default


@pytest.mark.parametrize("bad", [0, float("inf"), [0], (0,)])
@pytest.mark.parametrize("position", [0, 1])
def test_rejects_non_rv_bounds_at_combinator_construction(bad: Any, position: int):
    bounds = [constant(0), constant(1)]
    bounds[position] = bad
    calls = []

    def distribution():
        calls.append(True)
        return pi.normal(0, 1)

    with pytest.raises(TypeError, match="bounds must be RVs"):
        functional.truncated(distribution, *bounds)
    assert calls == []


@pytest.mark.parametrize("value", [[0], [0, 1], [[0, 1]]])
@pytest.mark.parametrize("position", [0, 1])
def test_rejects_array_bounds_even_for_matching_array_output(value, position):
    bounds = [constant(0), constant(1)]
    bounds[position] = constant(value)
    with pytest.raises(ValueError, match="bounds must be scalar"):
        functional.truncated(lambda: pi.normal(value, 1), *bounds)


@pytest.mark.parametrize(
    "low, high", [(-float("inf"), float("inf")), (2, 1), (0, 0), (float("nan"), 1)]
)
def test_does_not_validate_realized_bound_values(low, high):
    lower, upper = constant(low), constant(high)
    result = functional.truncated(pi.normal, lower, upper)(0, 1)
    assert result.parents[:2] == (lower, upper)


def test_preserves_deterministic_intermediates_and_drops_original_draw():
    lower, upper, mean, log_scale = map(constant, (-1, 2, 0, 1))
    created = []

    def distribution():
        scale = ir.RV(ir.Exp(), log_scale)
        draw = normal(mean, scale)
        created.extend((scale, draw))
        return draw

    result = functional.truncated(distribution, lower, upper)()
    assert result.parents == (lower, upper, mean, created[0])
    assert created[1] not in dag.upstream_nodes(result)
    assert result.op == ir.Truncated(ir.Normal())


def test_output_interface_is_preserved_not_taken_from_bounds_or_inputs():
    interface = ir.RVInterface()
    lower, upper = pi.constant(0), pi.constant(1)
    mean, scale = constant(0), constant(1)

    def distribution(mean, scale):
        return ir.RV(ir.Normal(), mean, scale, interface=interface)

    result = functional.truncated(distribution, lower, upper)(mean, scale)
    assert result.interface is interface
    assert lower.interface is pi.default
    assert mean.interface is ir.NO_INTERFACE


def test_random_bounds_and_parameters_are_shared_without_rewriting():
    lower = pi.normal(0, 1)
    upper = lower + pi.exponential(1)
    mean, scale = pi.normal(0, 1), pi.exponential(1)
    result = functional.truncated(lambda: pi.normal(mean, scale), lower, upper)()
    assert result.parents == (lower, upper, mean, scale)


def test_repeated_bounds_and_parameters_keep_aliasing():
    shared = constant(1)
    result = functional.truncated(normal, shared, shared)(shared, shared)
    assert all(parent is shared for parent in result.parents)


@pytest.mark.parametrize("output", [None, 3, [], (), {}, [pi.normal(0, 1)]])
def test_rejects_non_single_rv_output(output):
    with pytest.raises(TypeError):
        functional.truncated(lambda: output, constant(0), constant(1))()


def test_rejects_existing_output_even_if_other_rvs_were_created():
    existing = pi.normal(0, 1)

    def distribution():
        pi.normal(0, 1)
        return existing

    with pytest.raises(ValueError, match="created during"):
        functional.truncated(distribution, constant(0), constant(1))()


def test_rejects_existing_argument_returned_unchanged():
    with pytest.raises(ValueError, match="created during"):
        functional.truncated(lambda x: x, constant(0), constant(1))(pi.normal(0, 1))


def test_rejects_deterministic_output():
    with pytest.raises(ValueError, match="must be random"):
        functional.truncated(pi.exp, constant(0), constant(1))(0)


@pytest.mark.parametrize(
    "distribution",
    [
        lambda: pi.bernoulli(0.5),
        lambda: pi.bernoulli([0.2, 0.3]),
        functional.composite(lambda: pi.poisson(1), simplify=False),
    ],
)
def test_rejects_discrete_outputs_through_wrappers(distribution):
    with pytest.raises(ValueError, match="must be continuous"):
        functional.truncated(distribution, constant(0), constant(1))()


def test_rejects_contributing_new_random_intermediate():
    with pytest.raises(ValueError, match="random intermediate"):
        functional.truncated(lambda: pi.normal(pi.normal(0, 1), 1), constant(0), constant(1))()


def test_ignores_unused_random_draws_before_and_after_output_creation():
    def distribution():
        pi.normal(0, 1)
        output = pi.normal(0, 1)
        pi.normal(0, 1)
        return output

    result = functional.truncated(distribution, constant(0), constant(1))()
    assert result.op == ir.Truncated(ir.Normal())


def test_custom_scalar_distribution_needs_no_backend_support():
    class CustomDistribution(ir.Op):
        random: bool = True
        """True"""
        discrete: bool = False
        """False"""

        def get_shape(self, parameter_shape: ir.Shape) -> ir.Shape:
            assert parameter_shape == (2,)
            return ()

    parameter = constant([0, 1])
    result = functional.truncated(
        lambda: ir.RV(CustomDistribution(), parameter), constant(0), constant(1)
    )()
    assert result.op == ir.Truncated(CustomDistribution())
    assert result.parents[2] is parameter


def test_nested_truncation_keeps_each_pair_of_bounds_in_order():
    a, b, c, d, mean, scale = map(constant, (-3, 4, -1, 2, 0, 1))
    result = functional.truncated(functional.truncated(normal, a, b), c, d)(mean, scale)
    assert result.op == ir.Truncated(ir.Truncated(ir.Normal()))
    assert result.parents == (c, d, a, b, mean, scale)


def test_vmap_keeps_nonleading_axis_and_explicit_size():
    base = ir.VMap(ir.VMap(ir.Normal(), (0, None), 2), (1, None), 3)
    means, scale = constant([[0, 1, 2], [3, 4, 5]]), constant(1)
    lower, upper = constant(0), constant(10)
    result = functional.truncated(lambda: ir.RV(base, means, scale), lower, upper)()
    expected = ir.VMap(
        ir.VMap(ir.Truncated(ir.Normal()), (None, None, 0, None), 2),
        (None, None, 1, None),
        3,
    )
    assert result.op == expected
    assert result.parents == (lower, upper, means, scale)
    assert result.shape == (3, 2)


def test_vmap_keeps_inferred_axis_size():
    base = ir.VMap(ir.Normal(), (0, None))
    result = functional.truncated(
        lambda: ir.RV(base, constant([0, 1]), constant(1)), constant(0), constant(10)
    )()
    assert result.op == ir.VMap(ir.Truncated(ir.Normal()), (None, None, 0, None))
    assert result.shape == (2,)


def test_vmap_with_all_unmapped_inputs():
    base = ir.VMap(ir.Normal(), (None, None), 4)
    result = functional.truncated(
        lambda: ir.RV(base, constant(0), constant(1)), constant(0), constant(10)
    )()
    assert result.op == ir.VMap(ir.Truncated(ir.Normal()), (None,) * 4, 4)
    assert result.shape == (4,)


def test_composite_reindexes_inputs_intermediates_unused_and_repeated_parents():
    base = ir.Composite(
        3,
        (ir.Constant(2), ir.Add(), ir.Mul(), ir.Normal()),
        ((), (2, 3), (4, 4), (5, 2)),
    )
    inputs = tuple(map(constant, (99, 88, 1)))
    lower, upper = constant(0), constant(10)
    result = functional.truncated(lambda: ir.RV(base, *inputs), lower, upper)()
    expected = ir.Composite(
        5,
        (base.ops[0], base.ops[1], base.ops[2], ir.Truncated(ir.Normal())),
        ((), (4, 5), (6, 6), (0, 1, 7, 4)),
    )
    assert result.op == expected
    assert result.parents == (lower, upper, *inputs)
    assert isinstance(result.op, ir.Composite)
    assert all(a is b for a, b in zip(result.op.ops[:-1], base.ops[:-1]))
    assert base.par_nums == ((), (2, 3), (4, 4), (5, 2))


def test_zero_input_composite_and_degenerate_composite_are_not_simplified():
    base = ir.Composite(0, (ir.Constant(0), ir.Constant(1), ir.Normal()), ((), (), (0, 1)))
    lower, upper = constant(0), constant(10)
    result = functional.truncated(lambda: ir.RV(base), lower, upper)()
    assert result.op == ir.Composite(
        2,
        (ir.Constant(0), ir.Constant(1), ir.Truncated(ir.Normal())),
        ((), (), (0, 1, 2, 3)),
    )
    simple = ir.Composite(2, (ir.Normal(),), ((1, 0),))
    result = functional.truncated(lambda: ir.RV(simple, constant(1), constant(0)), lower, upper)()
    assert result.op == ir.Composite(4, (ir.Truncated(ir.Normal()),), ((0, 1, 3, 2),))


def test_deep_alternating_composite_vmap_and_bijected_structure():
    bijector = ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),)))
    scalar = ir.Bijected(ir.Normal(), bijector)
    inner = ir.Composite(2, (ir.Exp(), scalar), ((1,), (0, 2)))
    mapped = ir.VMap(inner, (0, None), 2)
    middle = ir.Composite(2, (mapped,), ((1, 0),))
    outer = ir.Composite(2, (ir.VMap(middle, (None, 0), 3),), ((1, 0),))
    means, log_scale = constant([[0, 1], [2, 3], [4, 5]]), constant(0)
    lower, upper = constant(1), constant(10)
    result = functional.truncated(lambda: ir.RV(outer, means, log_scale), lower, upper)()
    truncated_inner = ir.Composite(4, (ir.Exp(), ir.Truncated(scalar)), ((3,), (0, 1, 2, 4)))
    truncated_middle = ir.Composite(
        4, (ir.VMap(truncated_inner, (None, None, 0, None), 2),), ((0, 1, 3, 2),)
    )
    expected = ir.Composite(
        4, (ir.VMap(truncated_middle, (None, None, None, 0), 3),), ((0, 1, 3, 2),)
    )
    assert result.op == expected
    assert result.shape == (3, 2)
    assert result.parents == (lower, upper, means, log_scale)


def test_scalar_bijected_is_wrapped_whole_even_with_composite_inside():
    base = ir.Composite(2, (ir.Normal(),), ((0, 1),))
    bijector = ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),)))
    bijected = ir.Bijected(base, bijector)
    result = functional.truncated(
        lambda: ir.RV(bijected, constant(0), constant(1)), constant(1), constant(10)
    )()
    assert isinstance(result.op, ir.Truncated)
    assert result.op.base_op is bijected


def test_parameterized_bijector_keeps_its_parameters_after_bounds():
    offset, mean, scale, lower, upper = map(pi.constant, (10, 0, 1, 10, 20))
    shifted_normal = functional.tforms.shift_by(offset)(normal)
    result = functional.truncated(shifted_normal, lower, upper)(mean, scale)
    assert isinstance(result.op, ir.Truncated)
    assert isinstance(result.op.base_op, ir.Bijected)
    assert result.op.base_op.bijector.n_biject_params == 1
    assert result.parents == (lower, upper, offset, mean, scale)


def test_rejects_array_bijected_even_when_base_is_vmap():
    forward = ir.VMap(ir.Exp(), (0,))
    inverse = ir.VMap(ir.Log(), (0,))
    ldj = ir.Composite(2, (ir.Sum(0),), ((0,),))
    bijector = ir.Bijector(forward, inverse, ldj)
    base = ir.Bijected(ir.VMap(ir.Normal(), (0, None)), bijector)
    with pytest.raises(ValueError, match="scalar output"):
        functional.truncated(
            lambda: ir.RV(base, constant([0, 1]), constant(1)), constant(0), constant(10)
        )()


@pytest.mark.parametrize("wrap", [lambda op: op, lambda op: ir.Composite(2, (op,), ((0, 1),))])
@pytest.mark.parametrize(
    "base, inputs",
    [
        (ir.MultiNormal(), ([0, 0], [[1, 0], [0, 1]])),
        (ir.Scan(ir.Normal(), 3, (None,)), (0, 1)),
        (ir.VMap(ir.MultiNormal(), (0, None)), ([[0, 0], [1, 1]], [[1, 0], [0, 1]])),
    ],
)
def test_rejects_non_scalar_leaf_distributions_even_under_wrappers(wrap, base, inputs):
    parents = tuple(map(constant, inputs))
    with pytest.raises(ValueError, match="scalar output"):
        functional.truncated(lambda: ir.RV(wrap(base), *parents), constant(0), constant(1))()


def test_interface_broadcasting_is_retained_without_broadcasting_bounds():
    lower, upper = pi.constant(0), pi.constant(1)
    result = functional.truncated(pi.normal, lower, upper)([0, 1, 2], 1)
    assert result.op == ir.VMap(ir.Truncated(ir.Normal()), (None, None, 0, None), 3)
    assert result.shape == (3,)
    assert result.interface is pi.default


def test_varying_bounds_require_explicit_outer_vectorization():
    lower, upper = pi.constant([0, 1, 2]), pi.constant([1, 2, 3])
    mean, scale = pi.constant(0), pi.constant(1)

    def bounded(lo, hi):
        return functional.truncated(normal, lo, hi)(mean, scale)

    result = functional.vmap(bounded)(lower, upper)
    assert result.op == ir.VMap(ir.Truncated(ir.Normal()), (0, 0, None, None), 3)
    assert result.parents == (lower, upper, mean, scale)
    assert result.shape == (3,)


def test_scalar_composite_can_use_internal_vectorization_without_truncating_it():
    mapped_exp = ir.VMap(ir.Exp(), (0,))
    base = ir.Composite(2, (mapped_exp, ir.Sum(0), ir.Normal()), ((0,), (2,), (3, 1)))
    values, scale = constant([0, 1, 2]), constant(1)
    lower, upper = constant(0), constant(10)
    result = functional.truncated(lambda: ir.RV(base, values, scale), lower, upper)()
    assert result.op == ir.Composite(
        4,
        (mapped_exp, ir.Sum(0), ir.Truncated(ir.Normal())),
        ((2,), (4,), (0, 1, 5, 3)),
    )
    assert result.shape == ()


def test_vectorizing_before_or_after_truncation_gives_equivalent_ops_and_parents():
    lower, upper = pi.constant(0), pi.constant(1)
    means, scale = pi.constant([0, 1, 2]), pi.constant(1)
    before = functional.vmap(functional.truncated(normal, lower, upper), (0, None))(means, scale)
    after = functional.truncated(functional.vmap(normal, (0, None)), lower, upper)(means, scale)
    assert isinstance(before.op, ir.VMap)
    assert isinstance(after.op, ir.VMap)
    assert before.op.base_op == after.op.base_op
    assert before.parents == after.parents
    assert before.op == after.op


def test_composite_and_nested_truncation_through_public_combinators():
    lower, upper, mean, scale = map(constant, (0, 10, 0, 1))
    truncated_normal = functional.truncated(normal, lower, upper)
    inner = functional.composite(truncated_normal, simplify=False)
    result = functional.truncated(inner, lower, upper)(mean, scale)
    assert isinstance(result.op, ir.Composite)
    assert result.op.ops[-1] == ir.Truncated(ir.Truncated(ir.Normal()))
    assert result.shape == ()


def test_scalar_cdf_and_log_prob_can_wrap_truncation_structurally():
    lower, upper, mean, scale = map(constant, (0, 10, 0, 1))
    fun = functional.truncated(normal, lower, upper)
    density = functional.log_prob(fun)(mean, mean, scale)
    probability = functional.cdf(fun)(mean, mean, scale)
    assert density.op == ir.LogProb(ir.Truncated(ir.Normal()))
    assert probability.op == ir.CDF(ir.Truncated(ir.Normal()))
    assert density.parents == probability.parents == (mean, lower, upper, mean, scale)


@pytest.mark.parametrize(
    "lower, upper", [(True, True), (True, False), (False, True), (False, False)]
)
def test_optional_bounds_preserve_calls_and_parent_layout(lower: bool, upper: bool):
    lo, hi = constant(-1) if lower else None, constant(2) if upper else None
    mu, sigma = pi.constant(0), pi.constant(1)
    draws = []

    def distribution(mean, *, scale=sigma):
        result = pi.normal(mean, scale)
        draws.append(result)
        return result

    fun = functional.truncated(distribution, lo, hi)
    assert not draws
    result = fun(mu)
    assert len(draws) == 1
    bounds = tuple(bound for bound in (lo, hi) if bound is not None)
    assert result.parents == (*bounds, mu, sigma)
    assert result.interface is pi.default
    if bounds:
        assert result is not draws[0]
        assert result.op == ir.Truncated(ir.Normal(), lower=lower, upper=upper)
    else:
        assert result is draws[0]


def test_optional_bound_defaults_and_nested_one_sided_truncations():
    lo, hi, mean, scale = map(constant, (-1, 2, 0, 1))
    result = functional.truncated(functional.truncated(normal, lo), hi=hi)(mean, scale)
    assert result.op == ir.Truncated(ir.Truncated(ir.Normal(), upper=False), lower=False)
    assert result.parents == (hi, lo, mean, scale)
    assert functional.truncated(normal)(mean, scale).op == ir.Normal()


@pytest.mark.parametrize(
    "lower, upper", [(True, True), (True, False), (False, True), (False, False)]
)
def test_optional_bounds_through_nested_composites_and_nonleading_vmaps(lower: bool, upper: bool):
    scalar = ir.Composite(2, (ir.Exp(), ir.Normal()), ((1,), (0, 2)))
    mapped = ir.VMap(ir.VMap(scalar, (0, None), 2), (1, None), 3)
    base = ir.Composite(2, (mapped,), ((1, 0),))
    n = int(lower) + int(upper)
    bound_indices = tuple(range(n))
    inner = ir.Composite(
        2 + n,
        (ir.Exp(), ir.Truncated(ir.Normal(), lower=lower, upper=upper)),
        ((1 + n,), (*bound_indices, n, 2 + n)),
    )
    expected = ir.Composite(
        2 + n,
        (ir.VMap(ir.VMap(inner, (None,) * n + (0, None), 2), (None,) * n + (1, None), 3),),
        ((*bound_indices, 1 + n, n),),
    )
    assert truncate_op(base, lower=lower, upper=upper) == expected
    scale, means = constant(0), constant([[0, 1, 2], [3, 4, 5]])
    lo, hi = constant(-1) if lower else None, constant(2) if upper else None
    result = functional.truncated(lambda: ir.RV(base, scale, means), lo, hi)()
    assert result.op == (expected if n else base)
    assert result.shape == (3, 2)
    assert result.parents == (*tuple(b for b in (lo, hi) if b is not None), scale, means)


@pytest.mark.parametrize(
    "lower, upper", [(True, True), (True, False), (False, True), (False, False)]
)
def test_optional_bounds_on_zero_input_composite(lower: bool, upper: bool):
    base = ir.Composite(0, (ir.Constant(0), ir.Constant(1), ir.Normal()), ((), (), (0, 1)))
    n = int(lower) + int(upper)
    expected = ir.Composite(
        n,
        (*base.ops[:-1], ir.Truncated(ir.Normal(), lower=lower, upper=upper)),
        ((), (), (*range(n), n, n + 1)),
    )
    assert truncate_op(base, lower=lower, upper=upper) == expected
    assert expected.get_shape(*(((),) * n)) == ()


@pytest.mark.parametrize("value", [None, 0, 1, "True"])
def test_truncate_op_requires_boolean_flags(value: Any):
    for base in (
        ir.Normal(),
        ir.VMap(ir.Normal(), (0, None)),
        ir.Composite(2, (ir.Normal(),), ((0, 1),)),
    ):
        with pytest.raises(TypeError, match="must be booleans"):
            truncate_op(base, lower=value)
        with pytest.raises(TypeError, match="must be booleans"):
            truncate_op(base, upper=value)


@pytest.mark.parametrize(
    "fun, error, message",
    [
        (lambda: [], TypeError, "exactly one"),
        (lambda: pi.exp(0), ValueError, "must be random"),
        (lambda: pi.poisson(1), ValueError, "must be continuous"),
        (lambda: pi.normal(pi.normal(0, 1), 1), ValueError, "random intermediate"),
        (lambda: pi.multi_normal([0, 0], [[1, 0], [0, 1]]), ValueError, "scalar output"),
        (
            lambda: ir.RV(ir.Scan(ir.Normal(), 3, (None,)), constant(0), constant(1)),
            ValueError,
            "scalar output",
        ),
    ],
)
def test_no_bounds_still_validates_output_contract(fun, error, message):
    with pytest.raises(error, match=message):
        functional.truncated(fun)()


def test_no_bounds_rejects_existing_output():
    existing = pi.normal(0, 1)
    with pytest.raises(ValueError, match="created during"):
        functional.truncated(lambda: existing)()


def test_one_sided_bounds_can_be_vectorized_explicitly():
    lo = pi.constant([-1, 0, 1])
    mean, scale = pi.constant(0), pi.constant(1)
    result = functional.vmap(lambda bound: functional.truncated(normal, bound)(mean, scale))(lo)
    assert result.op == ir.VMap(ir.Truncated(ir.Normal(), upper=False), (0, None, None), 3)
    assert result.parents == (lo, mean, scale)


if TYPE_CHECKING:
    bound = constant(0)
    fun = functional.truncated(normal, bound, bound)
    fun(bound)  # type: ignore
    fun(bound, "bad")  # type: ignore
    fun(bound, bound, bound)  # type: ignore
    functional.truncated(normal, 0.0, bound)  # type: ignore
    assert_type(functional.truncated(normal)(bound, bound), ir.RV)
    assert_type(functional.truncated(normal, hi=bound)(bound, bound), ir.RV)
