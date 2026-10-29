"""Probability combinators coerce evaluation values, not model arguments."""

import numpy as np
import pytest
from typing import assert_type

from pangolin import functional, ir
from pangolin import interface as pi


@pytest.mark.parametrize("name", ["cdf", "log_prob"])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("value", [0.0, [0.0, 1.0], np.array([0.0, 1.0])])
def test_coerces_evaluation_value(name, configured, value):
    interface = pi.Interface(broadcasting="numpy") if configured else pi.default
    combinator = getattr(interface if configured else pi, name)
    mean = interface.constant(np.zeros(np.shape(value)))
    result = combinator(interface.normal)(value, mean, 1)
    assert result.parents[0].interface is interface
    np.testing.assert_array_equal(result.parents[0].op.value, value)
    expected = getattr(functional, name)(interface.normal)(interface.constant(value), mean, 1)
    assert ir.rv_equal(result, expected)
    assert result.shape == ()


@pytest.mark.parametrize("name", ["cdf", "log_prob"])
def test_preserves_existing_values_and_model_arguments(name):
    caller, producer = pi.Interface(), pi.Interface()
    value = producer.constant(0)
    marker = object()
    calls = []

    def model(argument, *, option):
        assert argument is marker
        assert option is marker
        calls.append(argument)
        return producer.normal(0, 1)

    result = getattr(caller, name)(model)(value, marker, option=marker)
    assert calls == [marker]
    assert result.parents[0] is value
    assert result.interface is producer


@pytest.mark.parametrize("name", ["cdf", "log_prob"])
def test_does_not_broadcast_evaluation_value(name):
    with pytest.raises(ValueError):
        getattr(pi, name)(pi.normal)(0, [0, 1], 1)


def test_typed_probability_wrappers():
    def model(mean: float, *, scale: float) -> ir.RV:
        return pi.normal(mean, scale)

    assert_type(pi.cdf(model)(0.0, 0.0, scale=1.0), ir.RV)
    assert_type(pi.log_prob(model)(0.0, 0.0, scale=1.0), ir.RV)
    interface = pi.Interface()
    assert_type(interface.cdf(model)(0.0, 0.0, scale=1.0), ir.RV)
    assert_type(interface.log_prob(model)(0.0, 0.0, scale=1.0), ir.RV)
