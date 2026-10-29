"""Interface scalar coercion leaves functional differentiation semantics intact."""

from typing import TYPE_CHECKING, Any, assert_type

import jax
import numpy as np
import pytest

from pangolin import functional, interface as pi, ir, jax_backend


def test_real_scalars_become_scalar_rvs():
    for value in (0, 2.5, np.int32(2), np.int64(3), np.float32(0.5), np.float64(1.5)):
        result = pi.grad(pi.exp)(value)
        assert_type(result, Any)
        assert result.op == ir.VJP(ir.Exp(), 0)
        argument = result.parents[1]
        assert argument.op == ir.Constant(value)
        assert argument.shape == ()
        assert argument.interface is pi.default


def test_rejected_values_are_not_coerced_or_passed_to_fun():
    calls = []

    def fun(x):
        calls.append(x)
        return pi.constant(1)

    for value in (True, np.bool_(False), 1j, "1", None, np.array(1.0),
                  np.array([1.0]), jax.numpy.array(1.0), [1.0], {"x": 1.0}):
        with pytest.raises(TypeError, match="leaves are RV"):
            pi.grad(fun)(value)
    assert calls == []


def test_configured_interface_and_single_call():
    configured = pi.Interface(broadcasting="off")
    calls = []

    def fun(x):
        calls.append(x)
        assert isinstance(x, ir.RV)
        assert x.interface is configured
        return x * x

    result = configured.grad(fun)(2.0)
    assert len(calls) == 1
    assert result.interface is configured
    assert result.parents[0].parents[1].interface is configured


def test_only_selected_argument_is_coerced():
    offset = [3.0]
    calls = []

    def fun(data, x, *, enabled):
        calls.append((data, x, enabled))
        assert data is offset
        assert enabled is True
        return x * data[0]

    result = pi.grad(fun, argnum=1)(offset, 2.0, enabled=True)
    assert len(calls) == 1
    assert result.op == ir.VJP(ir.Mul(), 0)
    assert result.parents[1].op == ir.Constant(2.0)
    assert result.parents[2].op == ir.Constant(3.0)


def test_unselected_floats_remain_real_floats_in_every_position():
    value = 2.5
    calls = []

    for argnum in range(3):
        def fun(a, b, c, *, offset):
            arguments = (a, b, c)
            calls.append(arguments)
            for index, argument in enumerate(arguments):
                if index == argnum:
                    assert isinstance(argument, ir.RV)
                else:
                    assert argument is value
                    assert type(argument) is float
            assert offset is value
            assert type(offset) is float
            return arguments[argnum] * offset

        result = pi.grad(fun, argnum)(value, value, value, offset=value)
        assert result.op == ir.VJP(ir.Mul(), 0)
    assert len(calls) == 3


def test_existing_rvs_and_output_interfaces_are_preserved():
    configured = pi.Interface(broadcasting="off")
    x = configured.constant(2.0)
    result = pi.grad(lambda a: a * a)(x)
    assert result.interface is configured
    assert result.parents[0].parents[1:] == (x, x)

    result = configured.grad(lambda a: pi.exp(a))(2.0)
    assert result.interface is pi.default
    assert result.parents[1].interface is configured


def test_tree_inputs_are_not_coerced():
    x = pi.constant(2.0)
    result = pi.grad(lambda tree: tree["x"] * tree["ys"][0])({"x": x, "ys": [x]})
    assert result["x"].op == ir.VJP(ir.Mul(), 0)
    assert result["ys"][0].op == ir.VJP(ir.Mul(), 1)
    assert result["x"].parents[1:] == (x, x)
    assert pi.grad(lambda tree: pi.constant(0))({}) == {}


def test_argument_validation_remains_functionals_responsibility():
    for argnum in (True, "0", 0.5):
        with pytest.raises(TypeError, match="integer"):
            pi.grad(lambda x: x, argnum)  # type: ignore
    with pytest.raises(ValueError, match="nonnegative"):
        pi.grad(lambda x: x, -1)
    with pytest.raises(ValueError, match="positional"):
        pi.grad(lambda x: x)(x=2.0)
    with pytest.raises(ValueError, match="positional"):
        pi.grad(lambda x: x, 1)(2.0)
    with pytest.raises(TypeError, match="leaves are RV"):
        functional.grad(pi.exp)(2.0)


def test_scalar_coercion_does_not_relax_output_contract():
    with pytest.raises(ValueError, match="random RVs"):
        pi.grad(lambda x: pi.normal(x, 1))(0.0)
    with pytest.raises(ValueError, match="scalar"):
        pi.grad(lambda x: x + pi.constant([1, 2]))(0.0)


def test_static_signature_is_not_widened():
    def square(x: ir.RV, *, enabled: bool = True) -> ir.RV:
        return x * x if enabled else x

    assert_type(pi.grad(square)(pi.constant(2.0), enabled=True), Any)
    if TYPE_CHECKING:
        pi.grad(square)(2.0)  # type: ignore
        pi.grad(square)(pi.constant(2.0), enabled="yes")  # type: ignore


def test_scalar_grad_compositions_numerically():
    results = (
        pi.grad(pi.grad(lambda x: x * x * x))(2.0),
        pi.grad(pi.log_prob(pi.normal))(2.0, 0.0, 2.0),
        pi.grad(lambda x: pi.cdf(pi.normal)(x, 0.0, 1.0))(0.0),
    )
    for result, expected in zip(results, (12.0, -0.5, 1 / np.sqrt(2 * np.pi)), strict=True):
        actual = jax_backend.ancestor_sample_graph(result, jax.random.key(0))
        assert isinstance(actual, jax.Array)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
