from pangolin.ir import *
import numpy as np
import pytest
import scipy.special
import jax
from scipy import stats
import random
from pangolin.testing import test_util
from base import HasInferenceProps


def rands_from_ranges(ranges):
    dims = np.random.choice([2, 3, 4])  # for matrices / vectors
    out = []
    for domain in ranges:
        if domain == "real":
            new = np.random.randn()
        elif domain == "positive":
            new = 0.1 + np.abs(np.random.randn())
        elif domain == "vector":
            new = np.random.randn(dims)
        elif domain == "matrix":
            new = np.random.randn(dims, dims)
        elif domain == "positive-definite":
            factor = np.random.randn(dims, dims)
            new = factor @ factor.T + 0.1 * np.eye(dims)
        elif isinstance(domain, set):
            new = random.choice(tuple(domain))
        elif isinstance(domain, tuple):
            lo, hi = domain
            new = lo + np.random.rand() * (hi - lo)
        elif domain == "simplex":
            scores = np.random.randn(dims)
            new = np.exp(scores - scipy.special.logsumexp(scores))
        elif domain == "positive-vector":
            new = 0.1 + np.abs(np.random.randn(dims))
        else:
            raise NotImplementedError()
        out.append(new)
    return out


# compare to scipy, using scipy's bizarre parameterization choices

testdata = [
    (Normal, scipy.stats.norm, ["real", "positive"]),
    (NormalPrec, lambda a, b: scipy.stats.norm(a, 1 / b**2), ["real", (0.1, 10.0)]),
    (Lognormal, lambda a, b: scipy.stats.lognorm(s=b, scale=np.exp(a)), ["real", "positive"]),
    (Cauchy, scipy.stats.cauchy, ["real", "positive"]),
    (Bernoulli, scipy.stats.bernoulli, [(0, 1)]),
    (BernoulliLogit, lambda a: scipy.stats.bernoulli(1 / (1 + np.exp(-a))), ["real"]),
    (Beta, scipy.stats.beta, ["positive", "positive"]),
    (Binomial, scipy.stats.binom, [{1, 2, 5, 10}, (0, 1)]),
    (
        Categorical,
        lambda p: scipy.stats.rv_discrete(name="categorical", values=(np.arange(len(p)), p)),
        ["simplex"],
    ),
    (Uniform, lambda a, b: scipy.stats.uniform(a, b - a), [(-0.75, -0.25), (0.25, 0.75)]),
    (BetaBinomial, scipy.stats.betabinom, [{1, 2, 5, 10}, "positive", "positive"]),
    (Exponential, lambda inv_scale: scipy.stats.expon(scale=1 / inv_scale), ["positive"]),
    (Gamma, lambda a, b: scipy.stats.gamma(a=a, loc=0, scale=1 / b), ["positive", "positive"]),
    (Poisson, scipy.stats.poisson, ["positive"]),
    (StudentT, scipy.stats.t, [{1.5, 2, 5}, "real", "positive"]),
    (Dirichlet, scipy.stats.dirichlet, ["positive-vector"]),
    (Multinomial, scipy.stats.multinomial, [{1, 2, 5, 10}, "simplex"]),
    (MultiNormal, scipy.stats.multivariate_normal, ["vector", "positive-definite"]),
    (Wishart, scipy.stats.wishart, [(5.1, 7.1), "positive-definite"]),
]

# Cauchy won't be tested since it has no mean


def get_mean(scipy_rv) -> None | np.ndarray:
    if isinstance(scipy_rv, scipy.stats._multivariate.multivariate_normal_frozen):
        # scipy is inconsistent about multivariate normal
        # randomly has rv.mean instead of rv.mean() lolwhat
        return scipy_rv.mean
    elif hasattr(scipy_rv, "mean") and not np.any(np.isnan(scipy_rv.mean())):
        return scipy_rv.mean()
    else:
        return None


def get_std(scipy_rv):
    if hasattr(scipy_rv, "std") and not np.any(np.isnan(scipy_rv.std())) and not np.any(np.isinf(scipy_rv.std())):
        return scipy_rv.std()
    else:
        return None


def get_cov(scipy_rv):
    if isinstance(scipy_rv, scipy.stats._multivariate.multivariate_normal_frozen):
        # scipy is inconsistent about multivariate normal
        # randomly has rv.cov instead of rv.cov()
        return scipy_rv.cov
    if hasattr(scipy_rv, "cov") and not np.any(np.isnan(scipy_rv.cov())) and not np.any(np.isinf(scipy_rv.cov())):
        return scipy_rv.cov()
    else:
        return None


class DistributionTests(HasInferenceProps):
    """
    Intended to be used as a mixin
    """

    @pytest.mark.parametrize("pangolin_op, scipy_fun, ranges", testdata)
    def test_random_op_sampling(self, pangolin_op, scipy_fun, ranges):
        if pangolin_op in self._ops_without_sampling_support:
            pytest.skip("Skipping because backend does not support this")

        for reps in range(1):
            inputs = rands_from_ranges(ranges)

            input_rvs = [RV(Constant(x)) for x in inputs]
            output_rv = RV(pangolin_op(), *input_rvs)

            scipy_rv = scipy_fun(*inputs)

            expected_mean = get_mean(scipy_rv)
            expected_std = get_std(scipy_rv)
            expected_cov = get_cov(scipy_rv)

            def testfun(samps_list):
                [samps] = samps_list
                samps = np.array(samps, copy=True)  # cast from JAX or pytorch or whatever

                match = True

                if expected_mean is not None:
                    empirical_mean = np.mean(samps, axis=0)
                    match = np.all(
                        np.abs(empirical_mean - expected_mean) / (0.1 + np.linalg.norm(expected_mean)) < 0.25
                    )

                # if expected_std is not None:
                #     empirical_std = np.std(samps, axis=0)
                #     match &= np.all(np.abs(empirical_std - expected_std) / (0.1 + expected_std) < 0.25)

                # if expected_cov is not None:
                #     empirical_cov = np.cov(samps.T)
                #     match &= np.all(
                #         np.abs(empirical_cov - expected_cov) / (0.1 + np.linalg.norm(expected_cov)) < 0.25
                #     )

                return match

            test_util.inf_until_match(
                self.sample_flat, [output_rv], [], [], testfun, niter_start=1000, niter_max=100000
            )
