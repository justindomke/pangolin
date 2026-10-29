"""Probability coercion and configured interfaces for the quantile combinator."""

from typing import assert_type

import numpy as np
import pytest

from pangolin import functional, interface as pi, ir


def test_default_and_configured_exports():
    assert "quantile" in pi.__all__
    assert pi.quantile == pi.default.quantile
    assert pi.quantile != functional.quantile
    configured = pi.Interface()
    result = configured.quantile(configured.normal)(0.5, 0, 1)
    assert_type(result, ir.RV)
    assert result.interface is configured
    assert all(parent.interface is configured for parent in result.parents)


@pytest.mark.parametrize("p", [0, 0.5, np.float32(0.5), np.array(0.5)])
def test_probability_coercion(p):
    result = pi.quantile(pi.normal)(p, 0, 1)
    assert result.parents[0].op == ir.Constant(p)
    assert result.parents[0].shape == ()


def test_output_interface_comes_from_function_and_existing_probability_is_preserved():
    owner, other = pi.Interface(), pi.Interface()
    p = other.constant(0.5)
    result = owner.quantile(other.normal)(p, 0, 1)
    assert result.parents[0] is p
    assert result.interface is other
    result = owner.quantile(other.normal)(0.5, 0, 1)
    assert result.parents[0].interface is owner
    assert result.interface is other


def test_model_arguments_are_not_coerced():
    data = {"means": [0, 1], "label": "normal"}
    calls = []

    def model(parameters, *, scale=1):
        assert parameters is data
        assert scale == 1
        calls.append(True)
        return pi.normal(parameters["means"][0], scale)

    quantile = pi.quantile(model)
    assert not calls
    assert_type(quantile(0.5, data, scale=1), ir.RV)
    assert calls == [True]


def test_no_implicit_broadcasting():
    configured = pi.Interface(broadcasting="numpy")
    with pytest.raises(ValueError, match="scalar"):
        configured.quantile(configured.normal)([0.1, 0.9], 0, 1)
    with pytest.raises(ValueError, match="scalar"):
        configured.quantile(configured.normal)(0.5, [0, 1], 1)


def test_explicit_broadcasting_wraps_scalar_quantiles():
    result = pi.broadcast(pi.quantile(pi.normal))([0.1, 0.9], [0, 1], 1)
    assert result.shape == (2,)
    assert result.op == ir.VMap(ir.Quantile(ir.Normal()), (0, 0, None), 2)
