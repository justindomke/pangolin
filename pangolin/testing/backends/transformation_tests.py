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


class TransformationTests(MixinBase):
    """
    Intended to be used as a mixin
    """

    def test_lognormal(self):
        x = pi.lognormal(0, 1)
        y = pi.tforms.exp(pi.normal)(0, 1)

        val = 0.01 + 1 / np.random.rand()
        expected = self.ancestor_log_prob_flat([x], [val])
        actual = self.ancestor_log_prob_flat([y], [val])

        assert np.allclose(expected, actual)

        def testfun(samps):
            [x_samps, y_samps] = samps
            x_samps = np.asarray(x_samps, copy=True)
            y_samps = np.asarray(y_samps, copy=True)

            return (
                np.abs(np.mean(y_samps) - np.mean(x_samps)) < 0.05 and np.abs(np.std(y_samps) - np.std(x_samps)) < 0.05
            )

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [x, y], testfun)

    def test_log_of_exponential(self):
        x = pi.log(pi.exponential(1.5))
        y = pi.tforms.log(pi.exponential)(1.5)

        def testfun(samps):
            [x_samps, y_samps] = samps
            x_samps = np.asarray(x_samps, copy=True)
            y_samps = np.asarray(y_samps, copy=True)

            return (
                np.abs(np.mean(y_samps) - np.mean(x_samps)) < 0.05 and np.abs(np.std(y_samps) - np.std(x_samps)) < 0.05
            )

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [x, y], testfun)

    def test_logit_beta(self):
        x = pi.logit(pi.beta(2, 2))
        y = pi.tforms.logit(pi.beta)(2, 2)

        def testfun(samps):
            [x_samps, y_samps] = samps
            x_samps = np.asarray(x_samps, copy=True)
            y_samps = np.asarray(y_samps, copy=True)

            return (
                np.abs(np.mean(y_samps) - np.mean(x_samps)) < 0.05 and np.abs(np.std(y_samps) - np.std(x_samps)) < 0.05
            )

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [x, y], testfun)

    def test_inv_logit_normal(self):
        x = pi.inv_logit(pi.normal(0, 1))
        y = pi.tforms.inv_logit(pi.normal)(0, 1)

        def testfun(samps):
            [x_samps, y_samps] = samps
            x_samps = np.asarray(x_samps, copy=True)
            y_samps = np.asarray(y_samps, copy=True)

            return (
                np.abs(np.mean(y_samps) - np.mean(x_samps)) < 0.05 and np.abs(np.std(y_samps) - np.std(x_samps)) < 0.05
            )

        test_util.ancestor_sample_until_match(self.ancestor_sample_flat, [x, y], testfun)
