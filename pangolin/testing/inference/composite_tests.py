from pangolin import ir
import numpy as np
import pytest
import scipy.special
import jax
from scipy import stats
import random
from pangolin.testing import test_util


class CompositeTests:
    """
    Intended to be used as a mixin
    """

    # off because no discrete vars

    # def test_bernoulli():
    #     op = ir.Composite(1, [ir.Bernoulli()], [[0]])

    #     x = ir.RV(ir.Constant(0.7))
    #     y = ir.RV(op, x)
    #     assert isinstance(y.op, ir.Composite)

    #     def testfun(E_y):
    #         return np.abs(E_y - 0.7) < 0.01

    #     inf_until_match(E, y, [], [], testfun)

    # off because no constrained vars (yet)

    # def test_exponential():
    #     op = ir.Composite(1, [ir.Exponential()], [[0]])

    #     x = ir.RV(ir.Constant(0.7))
    #     y = ir.RV(op, x)
    #     assert isinstance(y.op, ir.Composite)

    #     def testfun(E_y):
    #         return np.abs(E_y - 1 / 0.7) < 0.01

    #     inf_until_match(E, y, [], [], testfun)

    # def test_exponential():
    #     op = ir.Composite(1, [ir.Exponential()], [[0]])

    #     x = ir.RV(ir.Constant(0.7))
    #     y = ir.RV(op, x)
    #     assert isinstance(y.op, ir.Composite)

    #     def testfun(E_y):
    #         return np.abs(E_y - 1 / 0.7) < 0.01

    #     inf_until_match(E, y, [], [], testfun)

    def test_add_normal(self):
        # z ~ Normal(x+y, y)
        op = ir.Composite(2, [ir.Add(), ir.Normal()], [[0, 1], [2, 1]])

        x = ir.RV(ir.Constant(0.3))
        y = ir.RV(ir.Constant(0.1))
        z = ir.RV(op, x, y)

        def testfun(samps):
            [samps_z] = samps
            E_z = np.mean(samps_z)
            return np.abs(E_z - 0.4) < 0.01

        test_util.inf_until_match(self.sample_flat, [z], [], [], testfun)

    def test_add(self):
        # x -> x+x

        op = ir.Composite(1, [ir.Add()], [[0, 0]])

        x = ir.RV(ir.Constant(1.5))
        y = ir.RV(op, x)
        assert isinstance(y.op, ir.Composite)

        expected = 3.0
        [samps] = self.sample_flat([y], [], [], niter=1)

        assert np.allclose(expected, samps[0])

    def test_add_mul(self):
        # x,y -> (x+x)*y
        op = ir.Composite(2, [ir.Add(), ir.Mul()], [[0, 0], [2, 1]])

        x = ir.RV(ir.Constant(3.3))
        y = ir.RV(ir.Constant(4.4))
        z = ir.RV(op, x, y)
        assert isinstance(z.op, ir.Composite)

        expected = (3.3 + 3.3) * 4.4
        [samps] = self.sample_flat([z], [], [], niter=1)

        assert np.allclose(expected, samps[0])
