from pangolin import ir
import numpy as np
import pytest
import scipy.special
import jax
from scipy import stats
import random
from pangolin.testing import test_util
from .base import MixinBase
from pangolin import interface as pi


def wishart_mean_std(nu, S):
    mean = nu * S

    # Get the diagonal elements (v_ii)
    diag_S = np.diag(S)

    # Compute v_ij^2
    term1 = S**2

    # Compute v_ii * v_jj using outer product
    term2 = np.outer(diag_S, diag_S)

    # Calculate Variance
    variance_matrix = nu * (term1 + term2)

    # Return Standard Deviation
    return mean, np.sqrt(variance_matrix)


class TransformationTests(MixinBase):
    """
    Intended to be used as a mixin. Tests if backends can make use of full transforms.
    """

    # def test_unconstrain_spd(self):
    #     x = pi.tforms.unconstrain_spd(pi.wishart)(2, np.eye(2))

    #     expected_mean, expected_std = wishart_mean_std(2, np.eye(2))

    #     def testfun(samps):
    #         [x_samps] = samps
    #         return np.abs(np.mean(x_samps, axis=0) - expected_mean) < 0.05  # and np.abs(np.var(x_samps) - 0.5) < 0.05

    #     test_util.inf_until_match(self.sample_flat, [x], [], [], testfun)
