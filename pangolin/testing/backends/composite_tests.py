# from pangolin.interface import *
from pangolin.ir import *
import numpyro.handlers
import numpy as np
import pytest
import scipy.special
from pangolin.jax_backend import ancestor_sample_flat
import jax
from pangolin import ir
from .base import MixinBase
from pangolin.testing import test_util
from pangolin import interface as pi


class CompositeTests(MixinBase):
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

        def testfun(samps):
            [y_samps] = samps
            return np.abs(np.mean(y_samps) - 0.7) < 0.05

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [y], testfun)

        [y_samp] = self.ancestor_sample_flat([y])

        assert y_samp.shape == ()

        l = self.ancestor_log_prob_flat([y], [y_samp])
        numpyro_dist = numpyro.distributions.Bernoulli(0.7)
        expected = numpyro_dist.log_prob(y_samp)

        assert np.allclose(l, expected)

    def test_exponential(self):
        op = ir.Composite(1, [ir.Exponential()], [[0]])

        x = ir.RV(ir.Constant(0.23))
        y = ir.RV(op, x)
        assert isinstance(y.op, ir.Composite)

        def testfun(samps):
            [y_samps] = samps
            return np.abs(np.mean(y_samps) - 1 / 0.23) < 0.05

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [y], testfun)

        [y_samp] = self.ancestor_sample_flat([y])
        assert y_samp.shape == ()

        l = self.ancestor_log_prob_flat([y], [y_samp])
        numpyro_dist = numpyro.distributions.Exponential(0.23)
        expected = numpyro_dist.log_prob(y_samp)
        assert np.allclose(l, expected)

    def test_add_normal(self):
        # z ~ Normal(x+y, y)
        op = ir.Composite(2, [ir.Add(), ir.Normal()], [[0, 1], [2, 1]])

        x = ir.RV(ir.Constant(0.3))
        y = ir.RV(ir.Constant(0.1))
        z = ir.RV(op, x, y)

        expected_mean = 0.4
        expected_std = 0.1

        def testfun(samps):
            [z_samps] = samps
            return np.abs(np.mean(z_samps) - expected_mean) < 0.05 and np.abs(np.std(z_samps) - expected_std) < 0.05

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [z], testfun)

        [z_samp] = self.ancestor_sample_flat([z])
        assert z_samp.shape == ()

        l = self.ancestor_log_prob_flat([z], [z_samp])
        numpyro_dist = numpyro.distributions.Normal(0.4, 0.1)
        expected = numpyro_dist.log_prob(z_samp)
        assert np.allclose(l, expected)

    def test_composite_deterministic(self):
        @pi.composite
        def f(x):
            a = x + 2
            b = x * x
            return a + b

        x = pi.constant(1.5)
        y = f(x)
        assert isinstance(y.op, ir.Composite)

        expected_mean = (1.5 + 2) + (1.5**2)

        def testfun(samps):
            [y_samps] = samps
            return np.abs(np.mean(y_samps) - expected_mean) < 0.05

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [y], testfun)

    def test_composite_random(self):
        @pi.composite
        def f(x):
            a = x + 2
            b = x * x
            return pi.normal(a**b, 1e-5)

        x = pi.constant(1.5)
        y = f(x)
        assert isinstance(y.op, ir.Composite)

        expected_mean = (1.5 + 2) ** (1.5**2)

        def testfun(samps):
            [y_samps] = samps
            return np.abs(np.mean(y_samps) - expected_mean) < 0.05

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [y], testfun)

        [y_samp] = self.ancestor_sample_flat([y])
        l = self.ancestor_log_prob_flat([y], [y_samp])
        a_samp = 1.5 + 2
        b_samp = 1.5 * 1.5
        numpyro_dist = numpyro.distributions.Normal(a_samp**b_samp, 1e-5)
        expected = numpyro_dist.log_prob(y_samp)
        assert np.allclose(l, expected)

    def test_composite_simple_const_rv(self):
        x = pi.constant(0.5)
        noise = pi.constant(1e-3)

        @pi.composite
        def f(last):
            return pi.normal(last, noise)  # +1

        y = f(x)

        def testfun(samps):
            [y_samps] = samps
            E_y = np.mean(y_samps)
            return np.abs(E_y - 0.5) < 0.05

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [y], testfun)
