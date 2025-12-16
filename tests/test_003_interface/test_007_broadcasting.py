# test that broadcasting behavior varies based on environmental variable
# this needs some typical python packaging madness

# TODO: Add tests with broadcasting completely off

import pytest
import sys


@pytest.fixture(autouse=True)
def reset_module_imports():
    """Ensure fresh module imports for each test in this file only."""

    PROJECT_ROOT = "pangolin"

    modules_to_clear = [
        key
        for key in list(sys.modules.keys())
        if key == PROJECT_ROOT or key.startswith(f"{PROJECT_ROOT}.")
    ]

    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]

    yield  # Test runs here

    # Optional: Clean up after test too
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]


def test_unary_op_off():
    from pangolin.interface import exp, constant, bernoulli
    from pangolin import ir, interface

    with interface.broadcasting_off():

        for fun, op in [(exp, ir.Exp()), (bernoulli, ir.Bernoulli())]:

            a = constant(0.5)
            b = constant([0.1, 0.2, 0.3])
            c = constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

            assert fun(a).op == op

            for arg in [b, c]:
                try:
                    fun(arg).op
                    assert False
                except ValueError as e:
                    assert str(e).startswith(f"{op.name} op got parent shapes")


def test_unary_op_simple():
    from pangolin.interface import exp, constant, bernoulli
    from pangolin import ir, interface

    with interface.broadcasting_simple():

        for fun, op in [(exp, ir.Exp()), (bernoulli, ir.Bernoulli())]:

            a = constant(0.5)
            b = constant([0.1, 0.2, 0.3])
            c = constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

            assert fun(a).op == op
            assert fun(b).op == ir.VMap(op, (0,), 3)
            assert fun(c).op == ir.VMap(ir.VMap(op, (0,), 3), (0,), 2)


def test_unary_op_numpy():
    from pangolin.interface import exp, constant, bernoulli
    from pangolin import ir, interface

    with interface.broadcasting_numpy():
        for fun, op in [(exp, ir.Exp()), (bernoulli, ir.Bernoulli())]:

            a = constant(0.5)
            b = constant([0.1, 0.2, 0.3])
            c = constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

            assert fun(a).op == op
            assert fun(b).op == ir.VMap(op, (0,), 3)
            assert fun(c).op == ir.VMap(ir.VMap(op, (0,), 3), (0,), 2)


def test_binary_op_off():
    from pangolin.interface import add, constant, normal
    from pangolin import ir, interface

    with interface.broadcasting_off():

        for fun, op in [(add, ir.Add()), (normal, ir.Normal())]:
            a = constant(1)
            b = constant([1, 2, 3])
            c = constant([[1, 2, 3], [4, 5, 6]])

            assert fun(a, a).op == op

            for args in [
                (a, b),
                (a, c),
                (b, a),
                (b, b),
                (b, c),
                (c, a),
                (c, b),
                (c, c),
            ]:
                try:
                    fun(*args)
                except ValueError as e:
                    assert str(e).startswith(f"{op.name} op got parent shapes")


def test_binary_op_simple():
    from pangolin.interface import add, constant, normal
    from pangolin import ir, interface

    with interface.broadcasting_simple():

        for fun, op in [(add, ir.Add()), (normal, ir.Normal())]:
            a = constant(1)
            b = constant([1, 2, 3])
            c = constant([[1, 2, 3], [4, 5, 6]])

            assert fun(a, a).op == op
            assert fun(a, b).op == ir.VMap(op, (None, 0), 3)
            assert fun(a, c).op == ir.VMap(ir.VMap(op, (None, 0), 3), (None, 0), 2)
            assert fun(b, a).op == ir.VMap(op, (0, None), 3)
            assert fun(b, b).op == ir.VMap(op, (0, 0), 3)
            assert fun(c, a).op == ir.VMap(ir.VMap(op, (0, None), 3), (0, None), 2)
            assert fun(c, c).op == ir.VMap(ir.VMap(op, (0, 0), 3), (0, 0), 2)

            for args in [(b, c), (c, b)]:
                try:
                    fun(*args)
                    assert False
                except ValueError as e:
                    assert str(e).startswith("Can't broadcast")


def test_binary_op_numpy():
    from pangolin.interface import add, constant, normal
    from pangolin import ir, interface

    with interface.broadcasting_numpy():

        for fun, op in [(add, ir.Add()), (normal, ir.Normal())]:
            a = constant(1)
            b = constant([1, 2, 3])
            c = constant([[1, 2, 3], [4, 5, 6]])
            d = constant([7, 8, 9, 10])

            assert fun(a, a).op == op
            assert fun(a, b).op == ir.VMap(op, (None, 0), 3)
            assert fun(a, c).op == ir.VMap(ir.VMap(op, (None, 0), 3), (None, 0), 2)
            assert fun(b, a).op == ir.VMap(op, (0, None), 3)
            assert fun(b, b).op == ir.VMap(op, (0, 0), 3)
            assert fun(b, c).op == ir.VMap(ir.VMap(op, (0, 0), 3), (None, 0), 2)
            assert fun(c, a).op == ir.VMap(ir.VMap(op, (0, None), 3), (0, None), 2)
            assert fun(c, b).op == ir.VMap(ir.VMap(op, (0, 0), 3), (0, None), 2)
            assert fun(c, c).op == ir.VMap(ir.VMap(op, (0, 0), 3), (0, 0), 2)

            for args in [(b, d), (c, d), (d, c)]:
                try:
                    fun(*args)
                    assert False
                except ValueError as e:
                    assert str(e).startswith("shape mismatch")


def test_tertiary_op_off():
    from pangolin.interface import student_t, beta_binomial, constant
    from pangolin import ir, interface

    with interface.broadcasting_off():

        for fun, op in [(student_t, ir.StudentT()), (beta_binomial, ir.BetaBinomial())]:
            a = constant(1.5)
            b = constant([1, 2, 3])
            c = constant([[1, 2, 3], [4, 5, 6]])

            assert fun(a, a, a).op == op

            for args in [
                (a, b, a),
                (b, a, b),
                (c, a, c),
                (c, c, c),
                (a, b, c),
                (b, a, c),
            ]:
                try:
                    fun(*args)
                except ValueError as e:
                    assert str(e).startswith(f"{op.name} op got parent shapes")


def test_tertiary_op_simple():
    from pangolin.interface import student_t, beta_binomial, constant
    from pangolin import ir, interface

    with interface.broadcasting_simple():

        for fun, op in [(student_t, ir.StudentT()), (beta_binomial, ir.BetaBinomial())]:
            a = constant(1.5)
            b = constant([1, 2, 3])
            c = constant([[1, 2, 3], [4, 5, 6]])

            assert fun(a, a, a).op == op
            assert fun(a, b, a).op == ir.VMap(op, (None, 0, None), 3)
            assert fun(b, a, b).op == ir.VMap(op, (0, None, 0), 3)
            assert fun(c, a, c).op == ir.VMap(
                ir.VMap(op, (0, None, 0), 3), (0, None, 0), 2
            )
            assert fun(c, c, c).op == ir.VMap(ir.VMap(op, (0, 0, 0), 3), (0, 0, 0), 2)

            try:
                fun(a, b, c)
                assert False
            except ValueError as e:
                assert str(e).startswith("Can't broadcast")

            try:
                fun(b, a, c)
                assert False
            except ValueError as e:
                assert str(e).startswith("Can't broadcast")


def test_tertiary_op_numpy():
    from pangolin.interface import student_t, beta_binomial, constant
    from pangolin import ir, interface

    with interface.broadcasting_numpy():

        for fun, op in [(student_t, ir.StudentT()), (beta_binomial, ir.BetaBinomial())]:
            a = constant(1.5)
            b = constant([1, 2, 3])
            c = constant([[1, 2, 3], [4, 5, 6]])

            assert fun(a, a, a).op == op
            assert fun(a, b, a).op == ir.VMap(op, (None, 0, None), 3)
            assert fun(a, b, c).op == ir.VMap(
                ir.VMap(op, (None, 0, 0), 3), (None, None, 0), 2
            )
            assert fun(b, a, b).op == ir.VMap(op, (0, None, 0), 3)
            assert fun(b, a, c).op == ir.VMap(
                ir.VMap(op, (0, None, 0), 3), (None, None, 0), 2
            )
            assert fun(c, a, c).op == ir.VMap(
                ir.VMap(op, (0, None, 0), 3), (0, None, 0), 2
            )
            assert fun(c, c, c).op == ir.VMap(ir.VMap(op, (0, 0, 0), 3), (0, 0, 0), 2)


def test_autoregressive_broadcasting():
    from pangolin import interface as pi
    from jax import numpy as jnp

    with pi.broadcasting_simple():

        start = jnp.zeros(2)
        scale = pi.constant(1.0)

        z = pi.autoregressive(lambda last: pi.normal(last, scale), 100)(start)
        assert z.shape == (100, 2)

        z = pi.autoregressive(lambda last: pi.normal(last, scale), 100)([0.0, 0.5])
        assert z.shape == (100, 2)

        z = pi.autoregressive(lambda last: pi.normal(last, 1.0), 100)(start)
        assert z.shape == (100, 2)


def test_broadcasting_manager():
    from pangolin import interface as pi

    assert pi.config.broadcasting == pi.Broadcasting.SIMPLE

    with pi.broadcasting_off():
        assert pi.config.broadcasting == pi.Broadcasting.OFF

    with pi.broadcasting_simple():
        assert pi.config.broadcasting == pi.Broadcasting.SIMPLE

    with pi.broadcasting_numpy():
        assert pi.config.broadcasting == pi.Broadcasting.NUMPY
