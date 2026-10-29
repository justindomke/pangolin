from typing import TYPE_CHECKING, assert_type

import pytest

from pangolin import functional, interface as pi, ir


def constant(value):
    return ir.RV(ir.Constant(value))


def normal(mean: ir.RV, *, scale: ir.RV) -> ir.RV:
    return ir.RV(ir.Normal(), mean, scale)


def test_public_exports():
    assert pi.log_prob == pi.default.log_prob


def test_primitive_and_keyword_arguments():
    value, mean, scale = constant(2), constant(0), constant(1)
    fun = functional.log_prob(normal)
    result = fun(value, mean, scale=scale)
    assert_type(result, ir.RV)
    assert result.op == ir.LogProb(ir.Normal())
    assert result.parents == (value, mean, scale)
    assert result.shape == ()
    assert not result.op.random
    assert getattr(fun, "__name__") == "log_prob_normal"


def test_preserves_intermediate_nodes_and_output_interface():
    created = []
    interface = ir.RVInterface()

    def fun(x):
        scale = ir.RV(ir.Exp(), x)
        output = ir.RV(ir.Normal(), x, scale, interface=interface)
        created.extend([scale, output])
        return output

    value, x = constant(1), constant(0)
    result = functional.log_prob(fun)(value, x)
    assert result.parents == (value, x, created[0])
    assert result.interface is interface
    assert created[1] not in result.parents
    assert result.op == ir.LogProb(ir.Normal())


def test_captured_random_parameters_are_allowed():
    mean = pi.normal(0, 1)
    scale = pi.exponential(1)
    result = functional.log_prob(lambda: pi.normal(mean, scale))(constant(0))
    assert result.parents[1:] == (mean, scale)


def test_static_arguments_and_pytrees_are_unchanged():
    options = {"scale": constant(1), "label": "normal"}
    calls = []

    def fun(params, *, enabled):
        calls.append((params, enabled))
        return normal(constant(0), scale=params["scale"])

    functional.log_prob(fun)(constant(0), options, enabled=True)
    assert len(calls) == 1
    assert calls[0][0] is options
    assert calls[0][1] is True


def test_discrete_output():
    result = functional.log_prob(pi.bernoulli)(constant(1), 0.3)
    assert result.op == ir.LogProb(ir.Bernoulli())


def test_unused_random_nodes_are_ignored():
    def fun():
        pi.normal(0, 1)
        return pi.normal(0, 1)

    assert functional.log_prob(fun)(constant(0)).op == ir.LogProb(ir.Normal())


def test_rejects_internal_randomness():
    with pytest.raises(ValueError, match="random intermediate"):
        functional.log_prob(lambda: pi.normal(pi.normal(0, 1), 1))(constant(0))


def test_rejects_existing_output():
    output = pi.normal(0, 1)
    with pytest.raises(ValueError, match="created during"):
        functional.log_prob(lambda: output)(constant(0))


def test_rejects_deterministic_output():
    with pytest.raises(ValueError, match="must be random"):
        functional.log_prob(lambda: constant(0))(constant(0))


@pytest.mark.parametrize("output", [1, None, [], [pi.normal(0, 1)], {"x": pi.normal(0, 1)}])
def test_rejects_non_single_rv_output(output):
    with pytest.raises(TypeError):
        functional.log_prob(lambda: output)(constant(0))


def test_rejects_non_rv_value_before_calling_function():
    calls = []

    def fun():
        calls.append(True)
        return pi.normal(0, 1)

    with pytest.raises(TypeError, match="evaluation value must be an RV"):
        functional.log_prob(fun)(0)  # type: ignore
    assert calls == []


def test_rejects_wrong_value_shape():
    with pytest.raises(ValueError):
        functional.log_prob(pi.normal)(constant([0, 1]), 0, 1)


def test_array_output_still_has_scalar_log_probability():
    result = functional.log_prob(functional.vmap(pi.normal, (0, None)))(
        constant([0, 1]), constant([0, 0]), constant(1)
    )
    assert isinstance(result.op, ir.LogProb)
    assert isinstance(result.op.base_op, ir.VMap)
    assert result.shape == ()


def test_explicit_composite_is_preserved():
    fun = functional.composite(normal, simplify=False)
    result = functional.log_prob(fun)(constant(0), constant(0), scale=constant(1))
    assert result.op == ir.LogProb(ir.Composite(2, [ir.Normal()], [[0, 1]]))


def test_evaluation_argument_does_not_steal_function_keywords():
    def fun(value: ir.RV) -> ir.RV:
        return pi.normal(value, 1)

    mean = constant(0)
    result = functional.log_prob(fun)(constant(1), value=mean)
    assert result.parents[1] is mean


if TYPE_CHECKING:
    value = constant(0)
    density = functional.log_prob(normal)
    density(value, value)  # type: ignore
    density(value, value, scale="bad")  # type: ignore
