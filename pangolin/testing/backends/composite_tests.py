# from pangolin.interface import *
from pangolin.ir import *
import numpyro.handlers
import numpy as np
import pytest
import scipy.special
from pangolin.jax_backend import ancestor_sample_flat
import jax
from pangolin import ir


class CompositeTests:
    """
    Intended to be used as a mixin
    """

    def test_add(self):
        # x -> x+x

        op = ir.Composite(1, [ir.Add()], [[0, 0]])

        expected = 3.0
        # out = eval_op(op, [1.5])
        # assert np.allclose(out, expected)

        x = ir.RV(ir.Constant(1.5))
        y = ir.RV(op, x)
        assert isinstance(y.op, ir.Composite)

        [out] = self.ancestor_sample_flat([y], None)
        assert np.allclose(expected, out)

    def test_add_mul(self):
        # x,y -> (x+x)*y
        op = ir.Composite(2, [ir.Add(), ir.Mul()], [[0, 0], [2, 1]])

        x = ir.RV(ir.Constant(3.3))
        y = ir.RV(ir.Constant(4.4))
        z = ir.RV(op, x, y)
        assert isinstance(z.op, ir.Composite)

        expected = (3.3 + 3.3) * 4.4
        # out = eval_op(z.op, [3.3, 4.4])
        # assert np.allclose(expected, out)

        [out] = ancestor_sample_flat([z], None)
        assert np.allclose(expected, out)

    def test_bernoulli(self):
        op = ir.Composite(1, [ir.Bernoulli()], [[0]])

        x = ir.RV(ir.Constant(0.7))
        y = ir.RV(op, x)
        assert isinstance(y.op, ir.Composite)

        key = jax.random.PRNGKey(0)
        value = sample_op(y.op, key, [0.7])

        l = log_prob_op(y.op, value, [0.7])

        numpyro_dist = numpyro.distributions.Bernoulli(0.7)
        assert isinstance(numpyro_dist, numpyro.distributions.Distribution)
        expected = numpyro_dist.log_prob(value)

        assert np.allclose(l, expected)

    def test_exponential():
        op = ir.Composite(1, [ir.Exponential()], [[0]])

        x = ir.RV(ir.Constant(0.7))
        y = ir.RV(op, x)
        assert isinstance(y.op, ir.Composite)

        key = jax.random.PRNGKey(0)
        value = sample_op(y.op, key, [0.7])

        l = log_prob_op(y.op, value, [0.7])

        numpyro_dist = numpyro.distributions.Exponential(0.7)
        expected = numpyro_dist.log_prob(value)

        assert np.allclose(l, expected)

    def test_add_normal():
        # z ~ Normal(x+y, y)
        op = ir.Composite(2, [ir.Add(), ir.Normal()], [[0, 1], [2, 1]])

        x = ir.RV(ir.Constant(0.3))
        y = ir.RV(ir.Constant(0.1))
        z = ir.RV(op, x, y)

        key = jax.random.PRNGKey(0)

        value = sample_op(z.op, jax.random.split(key)[1], [0.3, 0.1])
        [value2] = ancestor_sample_flat([z], key)

        assert value == value2

        l = log_prob_op(z.op, value, [0.3, 0.1])
        l2 = ancestor_log_prob_flat([z], [value])

        tmp = 0.3 + 0.1
        numpyro_dist = numpyro.distributions.Normal(tmp, 0.1)
        expected = numpyro_dist.log_prob(value)
        assert np.allclose(l, expected)
        assert np.allclose(l2, expected)


# def test_composite_deterministic():
#     @composite
#     def f(x):
#         a = x + 2
#         b = x * x
#         return a + b

#     x = constant(1.5)
#     y = f(x)
#     assert isinstance(y.op, ir.Composite)

#     expected = (1.5 + 2) + (1.5**2)

#     [ys] = sample_flat([y], [], [], niter=100)
#     assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)

#     ys = sample(y, None, None, niter=100)
#     assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)


# def test_composite_random():
#     @composite
#     def f(x):
#         a = x + 2
#         b = x * x
#         return normal(a**b, 1e-5)

#     x = constant(1.5)
#     y = f(x)
#     assert isinstance(y.op, ir.Composite)

#     expected = (1.5 + 2) ** (1.5**2)

#     [ys] = sample_flat([y], [], [], niter=100)
#     assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)

#     ys = sample(y, None, None, niter=100)
#     assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)


# def test_composite_simple_const_rv():
#     x = constant(0.5)
#     noise = constant(1e-3)

#     @composite
#     def f(last):
#         return normal(last, noise)  # +1

#     y = f(x)

#     def testfun(E_y):
#         return np.abs(E_y - 0.5) < 0.1

#     inf_until_match(E, y, [], [], testfun)
