from typing import TYPE_CHECKING, assert_type

import pytest

from pangolin import functional, interface as pi, ir


def normal(mean: ir.RV, *, scale: ir.RV) -> ir.RV:
    return ir.RV(ir.Normal(), mean, scale)


def test_exports_and_signature():
    assert pi.cdf == pi.default.cdf
    value, mean, scale = (pi.constant(x) for x in (2, 0, 1))
    fun = functional.cdf(normal)
    result = fun(value, mean, scale=scale)
    assert_type(result, ir.RV)
    assert result.op == ir.CDF(ir.Normal())
    assert result.parents == (value, mean, scale)
    assert result.shape == ()
    assert not result.op.random
    assert getattr(fun, "__name__") == "cdf_normal"


def test_preserves_graph_and_output_interface():
    created = []
    interface = ir.RVInterface()

    def fun(x):
        scale = ir.RV(ir.Exp(), x)
        output = ir.RV(ir.Normal(), x, scale, interface=interface)
        created.extend([scale, output])
        return output

    value, x = pi.constant(1), pi.constant(0)
    result = functional.cdf(fun)(value, x)
    assert len(created) == 2
    assert result.parents == (value, x, created[0])
    assert result.interface is interface
    assert result.op == ir.CDF(ir.Normal())


def test_allows_captured_random_parameters():
    mean = pi.normal(0, 1)
    result = functional.cdf(lambda: pi.normal(mean, 1))(pi.constant(0))
    assert result.parents[1] is mean


def test_static_arguments_are_passed_unchanged():
    parameters = {"mean": pi.constant(0), "label": "normal"}

    def fun(params, *, value):
        assert params is parameters
        assert value == "normal"
        return pi.normal(params["mean"], 1)

    result = functional.cdf(fun)(pi.constant(0), parameters, value="normal")
    assert result.parents[1] is parameters["mean"]


def test_discrete_output():
    result = functional.cdf(pi.poisson)(pi.constant(2), 1)
    assert result.op == ir.CDF(ir.Poisson())


def test_unused_random_nodes_are_ignored():
    def fun():
        pi.normal(0, 1)
        return pi.normal(0, 1)

    assert functional.cdf(fun)(pi.constant(0)).op == ir.CDF(ir.Normal())


def test_rejects_internal_randomness():
    with pytest.raises(ValueError, match="random intermediate"):
        functional.cdf(lambda: pi.normal(pi.normal(0, 1), 1))(pi.constant(0))


def test_rejects_existing_output():
    output = pi.normal(0, 1)
    with pytest.raises(ValueError, match="created during"):
        functional.cdf(lambda: output)(pi.constant(0))


def test_rejects_deterministic_output():
    with pytest.raises(ValueError, match="must be random"):
        functional.cdf(lambda: pi.constant(0))(pi.constant(0))


@pytest.mark.parametrize("output", [None, 1, [], [pi.normal(0, 1)]])
def test_rejects_non_single_rv_output(output):
    with pytest.raises(TypeError):
        functional.cdf(lambda: output)(pi.constant(0))


def test_rejects_non_rv_value():
    with pytest.raises(TypeError, match="evaluation value must be an RV"):
        functional.cdf(pi.normal)(0, 0, 1)  # type: ignore


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        functional.cdf(pi.normal)(pi.constant([0, 1]), 0, 1)


def test_array_output_has_scalar_cdf():
    result = functional.cdf(pi.multi_normal)(pi.constant([0, 1]), [0, 0], [[1, 0], [0, 1]])
    assert result.op == ir.CDF(ir.MultiNormal())
    assert result.shape == ()


if TYPE_CHECKING:
    value = pi.constant(0)
    fun = functional.cdf(normal)
    fun(value, value)  # type: ignore
    fun(value, value, scale="bad")  # type: ignore
