from pangolin import ir
import numpy as np
import pytest
import scipy.special
import jax
from scipy import stats
import random
from pangolin.testing import test_util
from .base import MixinBase


class SimplePosteriorTests(MixinBase):
    """
    Intended to be used as a mixin
    """

    def test_simple(self):
        loc = ir.RV(ir.Constant(0))
        scale = ir.RV(ir.Constant(1))
        x = ir.RV(ir.Normal(), loc, scale)
        [x_samps] = self.sample_flat([x], [], [], niter=37)
        assert x_samps.shape == (37,)

        def testfun(samps):
            [x_samps] = samps
            return np.abs(np.mean(x_samps) - 0) < 0.05 and np.abs(np.std(x_samps) - 1) < 0.05

        test_util.inf_until_match(self.sample_flat, [x], [], [], testfun)

    def test_conditioning(self):
        loc = ir.RV(ir.Constant(0))
        scale = ir.RV(ir.Constant(1))
        x = ir.RV(ir.Normal(), loc, scale)
        y = ir.RV(ir.Normal(), x, scale)
        [x_samps] = self.sample_flat([x], [y], [1], niter=82)
        assert x_samps.shape == (82,)

        def testfun(samps):
            [x_samps] = samps
            return np.abs(np.mean(x_samps) - 0.5) < 0.05 and np.abs(np.var(x_samps) - 0.5) < 0.05

        test_util.inf_until_match(self.sample_flat, [x], [y], [1], testfun)

    def test_nonrandom(self):
        loc = ir.RV(ir.Constant(0))
        scale = ir.RV(ir.Constant(1))
        x = ir.RV(ir.Normal(), loc, scale)
        y = ir.RV(ir.Add(), x, x)
        [x_samps, y_samps] = self.sample_flat([x, y], [], [], niter=103)
        assert x_samps.shape == (103,)
        assert y_samps.shape == (103,)
        assert np.allclose(y_samps, x_samps * 2)

    def test_nonrandom_conditioning(self):
        loc = ir.RV(ir.Constant(0))
        scale = ir.RV(ir.Constant(1))
        x = ir.RV(ir.Normal(), loc, scale)
        z = ir.RV(ir.Normal(), x, scale)
        y1 = ir.RV(ir.Add(), x, x)
        y2 = ir.RV(ir.Mul(), x, x)
        [x_samps, y1_samps, y2_samps] = self.sample_flat([x, y1, y2], [z], [1.0], niter=49)
        assert x_samps.shape == y1_samps.shape == y2_samps.shape == (49,)

        def testfun(samps):
            [x_samps, y1_samps, y2_samps] = samps
            return (
                np.abs(np.mean(x_samps) - 0.5) < 0.05
                and np.abs(np.var(x_samps) - 0.5) < 0.05
                and np.allclose(y1_samps, x_samps * 2)
                and np.allclose(y2_samps, x_samps**2)
            )

        test_util.inf_until_match(self.sample_flat, [x, y1, y2], [z], [1.0], testfun)

    def test_nonrandom_from_given(self):
        loc = ir.RV(ir.Constant(0))
        scale = ir.RV(ir.Constant(1))
        x = ir.RV(ir.Normal(), loc, scale)
        z = ir.RV(ir.Normal(), x, scale)
        y = ir.RV(ir.Mul(), z, scale)
        [x_samps, y_samps] = self.sample_flat([x, y], [z], [1.0], niter=201)
        assert x_samps.shape == y_samps.shape == (201,)

        def testfun(samps):
            [x_samps, y_samps] = samps
            return (
                np.abs(np.mean(x_samps) - 0.5) < 0.05
                and np.abs(np.var(x_samps) - 0.5) < 0.05
                and np.allclose(y_samps, 1.0)
            )

        test_util.inf_until_match(self.sample_flat, [x, y], [z], [1.0], testfun)

    def test_given_in_output(self):
        loc = ir.RV(ir.Constant(0))
        scale = ir.RV(ir.Constant(1))
        x = ir.RV(ir.Normal(), loc, scale)
        z = ir.RV(ir.Normal(), x, scale)
        y = ir.RV(ir.Mul(), z, scale)
        [y_samps, z_samps] = self.sample_flat([y, z], [z], [2.3], niter=100)
        assert y_samps.shape == z_samps.shape == (100,)
        assert np.allclose(z_samps, 2.3)
        assert np.allclose(y_samps, 2.3)
