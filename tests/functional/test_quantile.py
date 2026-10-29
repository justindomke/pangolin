"""Quantile tracing contracts, graph sharing, and scalar restrictions."""

from typing import TYPE_CHECKING, assert_type

import pytest

from pangolin import functional, interface as pi, ir


def normal(mean: ir.RV, *, scale: ir.RV) -> ir.RV:
    return ir.RV(ir.Normal(), mean, scale)


def test_exports_and_signature():
    assert "quantile" in functional.__all__
    probability, mean, scale = (pi.constant(x) for x in (0.5, 0, 1))
    fun = functional.quantile(normal)
    result = fun(probability, mean, scale=scale)
    assert_type(result, ir.RV)
    assert result.op == ir.Quantile(ir.Normal())
    assert result.parents == (probability, mean, scale)
    assert result.shape == ()
    assert not result.op.random
    assert getattr(fun, "__name__") == "quantile_normal"
    assert getattr(functional.quantile(lambda: pi.normal(0, 1)), "__name__") == "quantile_function"


def test_calls_once_and_preserves_parameter_graph_and_output_interface():
    created = []
    owner = ir.RVInterface()

    def model(x):
        scale = ir.RV(ir.Exp(), x)
        output = ir.RV(ir.Normal(), x, scale, interface=owner)
        created.extend((scale, output))
        return output

    probability, x = pi.constant(0.5), pi.constant(0.0)
    result = functional.quantile(model)(probability, x)
    assert len(created) == 2
    assert result.parents == (probability, x, created[0])
    assert created[1] not in result.parents
    assert result.interface is owner


def test_random_inputs_captures_and_probability_are_preserved():
    probability = pi.uniform(0, 1)
    location = pi.normal(0, 1)
    scale = pi.lognormal(0, 1)
    result = functional.quantile(lambda x: pi.normal(x, scale))(probability, location)
    assert result.parents == (probability, location, scale)


def test_python_arguments_pass_through_unchanged():
    params = {"location": pi.constant(0), "metadata": ["normal", 2]}

    def model(parameters, *, probability):
        assert parameters is params
        assert probability == "static"
        return pi.normal(parameters["location"], 1)

    result = functional.quantile(model)(pi.constant(0.5), params, probability="static")
    assert result.parents[1] is params["location"]


def test_unused_random_nodes_are_ignored():
    def model():
        pi.normal(0, 1)
        return pi.normal(0, 1)

    assert functional.quantile(model)(pi.constant(0.5)).op == ir.Quantile(ir.Normal())


def test_discrete_and_truncated_distributions_are_structurally_valid():
    p = pi.constant(0.5)
    assert functional.quantile(pi.poisson)(p, 2).op == ir.Quantile(ir.Poisson())
    assert functional.quantile(pi.truncated(pi.normal, 0))(p, 0, 1).op == ir.Quantile(
        ir.Truncated(ir.Normal(), upper=False)
    )


def test_composite_structure_is_retained():
    model = functional.composite(normal, simplify=False)
    p, zero, one = (pi.constant(x) for x in (0.5, 0, 1))
    result = functional.quantile(model)(p, zero, scale=one)
    assert isinstance(result.op, ir.Quantile)
    assert isinstance(result.op.base_op, ir.Composite)
    assert result.parents == (p, zero, one)


@pytest.mark.parametrize("output", [None, 1, [], [pi.normal(0, 1)]])
def test_rejects_non_single_output(output):
    with pytest.raises(TypeError):
        functional.quantile(lambda: output)(pi.constant(0.5))


def test_rejects_existing_deterministic_and_internally_random_outputs():
    existing = pi.normal(0, 1)
    for model, message in [
        (lambda: existing, "created during"),
        (lambda: pi.constant(0), "must be random"),
        (lambda: pi.normal(pi.normal(0, 1), 1), "random intermediate"),
    ]:
        with pytest.raises(ValueError, match=message):
            functional.quantile(model)(pi.constant(0.5))


def test_probability_requires_rv_without_coercion():
    with pytest.raises(TypeError, match="probability must be an RV"):
        functional.quantile(pi.normal)(0.5, 0, 1)  # type: ignore


def test_scalar_shapes_are_required_even_for_length_one_arrays():
    with pytest.raises(ValueError, match="scalar"):
        functional.quantile(pi.normal)(pi.constant([0.5]), 0, 1)
    with pytest.raises(ValueError, match="scalar"):
        functional.quantile(pi.normal)(pi.constant(0.5), [0], 1)
    with pytest.raises(ValueError, match="scalar"):
        functional.quantile(pi.multi_normal)(pi.constant(0.5), [0, 1], [[1, 0], [0, 1]])


if TYPE_CHECKING:
    p = pi.constant(0.5)
    quantile = functional.quantile(normal)
    quantile(p, p)  # type: ignore
    quantile(p, p, scale="bad")  # type: ignore
