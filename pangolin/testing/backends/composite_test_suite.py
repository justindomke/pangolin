# from pangolin.interface import *
from pangolin.ir import *
import numpyro.handlers
import numpy as np
import pytest
import scipy.special
from pangolin.jax_backend import ancestor_sample_flat
import jax
from pangolin import ir


class DeterministicTestSuite:
    """
    This class assumes a fixture named 'ancestor_sample_flat' will be available at runtime.
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

        [out] = ancestor_sample_flat([y], None)
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


# class BackendTestSuite(
