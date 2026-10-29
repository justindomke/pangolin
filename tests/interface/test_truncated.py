from typing import TYPE_CHECKING, assert_type

import numpy as np
import pytest

from pangolin import functional, interface as pi, ir


def model(mean: pi.RVLike, *, scale: pi.RVLike = 1.0) -> ir.RV:
    return pi.normal(mean, scale)


def test_default_export_and_configured_method():
    assert "truncated" in pi.__all__
    assert pi.truncated == pi.default.truncated
    assert pi.truncated != functional.truncated
    configured = pi.Interface()
    result = configured.truncated(configured.normal, 0)(0, 1)
    assert result.interface is configured
    assert all(parent.interface is configured for parent in result.parents)


@pytest.mark.parametrize(
    "bounds, expected_op, expected",
    [
        ({}, ir.Normal(), ()),
        ({"lo": None, "hi": None}, ir.Normal(), ()),
        ({"lo": 0}, ir.Truncated(ir.Normal(), upper=False), (0,)),
        ({"lo": 0, "hi": None}, ir.Truncated(ir.Normal(), upper=False), (0,)),
        ({"hi": 2}, ir.Truncated(ir.Normal(), lower=False), (2,)),
        ({"lo": None, "hi": 2}, ir.Truncated(ir.Normal(), lower=False), (2,)),
        ({"lo": -1, "hi": 2}, ir.Truncated(ir.Normal()), (-1, 2)),
    ],
)
def test_optional_bounds(bounds, expected_op, expected):
    result = pi.truncated(pi.normal, **bounds)(1.1, 2.2)
    assert result.op == expected_op
    assert tuple(parent.op for parent in result.parents) == tuple(
        ir.Constant(value) for value in (*expected, 1.1, 2.2)
    )
    assert result.shape == ()


def test_positional_bounds_and_preserved_keyword_signature():
    fun = pi.truncated(model, -1, 2)
    result = fun(0, scale=2)
    assert_type(result, ir.RV)
    assert result.parents[0].op == ir.Constant(-1)
    assert result.parents[1].op == ir.Constant(2)
    assert result.parents[3].op == ir.Constant(2)
    assert fun(0).parents[3].op == ir.Constant(1.0)
    assert_type(pi.Interface().truncated(model, hi=1)(0, scale=2), ir.RV)


@pytest.mark.parametrize("bound", [0, 0.0, np.float32(0), np.int64(0), np.array(0.0)])
def test_coerces_scalar_bounds(bound):
    result = pi.truncated(pi.normal, bound, 1)(0, 1)
    assert result.parents[0].shape == ()
    assert result.parents[0].op == ir.Constant(bound)


@pytest.mark.parametrize("bound", [[0], (0,), np.array([0]), np.zeros((2, 3))])
@pytest.mark.parametrize("name", ["lo", "hi"])
def test_rejects_array_bounds_when_constructing_combinator(bound, name):
    calls = []

    def distribution():
        calls.append(True)
        return pi.normal(0, 1)

    with pytest.raises(ValueError, match="bounds must be scalar"):
        pi.truncated(distribution, **{name: bound})
    assert calls == []


def test_preserves_existing_bound_rvs_with_foreign_interfaces():
    other = pi.Interface(broadcasting="off")
    lower = other.normal(0, 1)
    upper = lower + 1
    result = pi.truncated(pi.normal, lower, upper)(0, 1)
    assert result.parents[:2] == (lower, upper)
    assert result.interface is pi.default


def test_output_interface_is_selected_by_function_not_wrapper():
    owner, other = pi.Interface(), pi.Interface()
    result = owner.truncated(other.normal, 0, 1)(0, 1)
    assert result.interface is other
    assert all(parent.interface is owner for parent in result.parents[:2])
    assert all(parent.interface is other for parent in result.parents[2:])


def test_raw_output_keeps_no_interface():
    zero = ir.RV(ir.Constant(0))
    one = ir.RV(ir.Constant(1))
    result = pi.truncated(lambda: ir.RV(ir.Normal(), zero, one), 0, 1)()
    assert result.interface is ir.NO_INTERFACE
    assert result.parents[2:] == (zero, one)
    assert all(parent.interface is pi.default for parent in result.parents[:2])


def test_bounds_are_coerced_once_and_shared_between_invocations():
    fun = pi.truncated(pi.normal, 0, 1)
    first, second = fun(0, 1), fun(2, 1)
    assert first is not second
    assert first.parents[:2] == second.parents[:2]


def test_does_not_coerce_function_arguments_or_call_function_early():
    parameters = {"means": [0, 1], "scale": 1}
    calls = []

    def distribution(params, *, label):
        calls.append((params, label))
        return pi.normal(params["means"], params["scale"])

    fun = pi.truncated(distribution, 0)
    assert calls == []
    result = fun(parameters, label="normal")
    assert len(calls) == 1
    assert calls[0][0] is parameters
    assert calls[0][1] == "normal"
    assert result.shape == (2,)


def test_normal_broadcasting_uses_shared_bounds():
    result = pi.truncated(pi.normal, 0)([0, 1, 2], 1)
    assert result.op == ir.VMap(ir.Truncated(ir.Normal(), upper=False), (None, 0, None), 3)
    assert result.shape == (3,)


def test_broadcasting_comes_from_base_function_not_wrapper():
    strict = pi.Interface(broadcasting="off")
    result = strict.truncated(pi.normal, 0)([0, 1], 1)
    assert result.shape == (2,)
    assert result.interface is pi.default
    with pytest.raises(ValueError):
        pi.truncated(strict.normal, 0)([0, 1], 1)


def test_configured_broadcasting_is_looked_up_when_function_is_called():
    configured = pi.Interface(broadcasting="off")
    fun = configured.truncated(configured.normal, 0)
    with pytest.raises(ValueError):
        fun([0, 1], 1)
    with configured.overload(broadcasting="simple"):
        result = fun([0, 1], 1)
        assert result.shape == (2,)
        assert result.interface is configured
    with pytest.raises(ValueError):
        fun([0, 1], 1)


def test_numpy_broadcasting_produces_nested_vmaps():
    configured = pi.Interface(broadcasting="numpy")
    result = configured.truncated(configured.normal, 0)([[0], [1]], [1, 2, 3])
    assert result.shape == (2, 3)
    assert result.interface is configured
    assert isinstance(result.op, ir.VMap)
    assert isinstance(result.op.base_op, ir.VMap)
    assert result.op.base_op.base_op == ir.Truncated(ir.Normal(), upper=False)
    assert result.op.in_axes[0] is None
    assert result.op.base_op.in_axes[0] is None


def test_custom_scalar_distribution_can_accept_vector_parameters():
    def distribution(values):
        return pi.normal(pi.sum(values, axis=0), 1)

    values = pi.constant([0, 1, 2])
    result = pi.truncated(distribution, 0)(values)
    assert result.op == ir.Truncated(ir.Normal(), upper=False)
    assert result.shape == ()
    assert result.parents[1].op == ir.Sum(axis=0)
    assert result.parents[1].parents == (values,)


def test_explicit_vectorization_can_supply_different_scalar_bounds():
    lower, upper = pi.constant([0, 1]), pi.constant([1, 2])
    result = pi.vmap(lambda lo, hi: pi.truncated(pi.normal, lo, hi)(0, 1))(lower, upper)
    assert result.shape == (2,)
    assert result.parents[:2] == (lower, upper)


def test_composes_with_composite_and_scalar_bijection():
    lognormal = pi.tforms.exp(pi.normal)
    fun = pi.composite(lognormal, simplify=False)
    result = pi.truncated(fun, 1, 10)(0, 1)
    assert isinstance(result.op, ir.Composite)
    assert isinstance(result.op.ops[-1], ir.Truncated)
    assert isinstance(result.op.ops[-1].base_op, ir.Bijected)
    assert result.interface is pi.default


@pytest.mark.parametrize(
    "fun, match",
    [
        (lambda: pi.bernoulli(0.5), "continuous"),
        (lambda: pi.multi_normal([0, 0], [[1, 0], [0, 1]]), "scalar output"),
        (lambda: pi.normal(pi.normal(0, 1), 1), "random intermediate"),
    ],
)
def test_functional_restrictions_are_preserved(fun, match):
    with pytest.raises(ValueError, match=match):
        pi.truncated(fun, 0, 1)()


@pytest.mark.parametrize("lo, hi", [(None, None), (0, None), (None, 1), (0, 1)])
def test_only_supplied_bounds_are_coerced_once(lo, hi):
    calls = []

    class RecordingInterface(pi.Interface):
        def coerce(self, value: pi.RVLike, /) -> ir.RV:
            calls.append(value)
            return super().coerce(value)

    configured = RecordingInterface()
    fun = configured.truncated(pi.normal, lo, hi)
    expected = tuple(bound for bound in (lo, hi) if bound is not None)
    assert calls == list(expected)
    first, second = fun(1.1, 2.2), fun(3.3, 4.4)
    assert calls == list(expected)
    n = len(expected)
    assert first.parents[:n] == second.parents[:n]
    assert all(parent.interface is configured for parent in first.parents[:n])
    assert all(parent.interface is pi.default for parent in first.parents[n:])
    assert first.interface is second.interface is pi.default


@pytest.mark.parametrize("name", ["lo", "hi"])
def test_one_sided_existing_bound_keeps_identity_and_interface(name):
    owner, other = pi.Interface(), pi.Interface()
    bound = other.normal(1.1, 2.2)
    result = owner.truncated(owner.normal, **{name: bound})(3.3, 4.4)
    assert result.parents[0] is bound
    assert result.parents[0].interface is other
    assert result.interface is owner
    assert all(parent.interface is owner for parent in result.parents[1:])


def test_no_bounds_returns_function_output_unchanged():
    owner, other = pi.Interface(), pi.Interface()
    draws = []

    def distribution():
        result = other.normal(1.1, 2.2)
        draws.append(result)
        return result

    result = owner.truncated(distribution)()
    assert len(draws) == 1
    assert result is draws[0]
    assert result.interface is other


@pytest.mark.parametrize("lo, hi", [(-np.inf, None), (None, np.inf), (-np.inf, np.inf)])
def test_explicit_infinite_bounds_are_not_treated_as_omitted(lo, hi):
    # Their values violate the contract, but checking that remains the backend's job.
    result = pi.truncated(pi.normal, lo, hi)(1.1, 2.2)
    assert result.op == ir.Truncated(ir.Normal(), lower=lo is not None, upper=hi is not None)
    expected = tuple(bound for bound in (lo, hi) if bound is not None)
    assert tuple(parent.op for parent in result.parents[: len(expected)]) == tuple(
        ir.Constant(value) for value in expected
    )


if TYPE_CHECKING:
    bounded = pi.truncated(model, 0)
    bounded()  # type: ignore
    bounded(0, 1)  # type: ignore
    bounded(0, unexpected=True)  # type: ignore
    bounded(0, scale=object())  # type: ignore
    assert_type(pi.truncated(model)(1.1), ir.RV)
    assert_type(pi.truncated(model, None, 1)(1.1), ir.RV)
    assert_type(pi.truncated(model, 0, None)(1.1), ir.RV)
