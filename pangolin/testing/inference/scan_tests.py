from pangolin import ir
import numpy as np
import pytest
import scipy.special
import jax
from scipy import stats
import random
from pangolin.testing import test_util
from pangolin import interface as pi
from .base import InferenceTestBase
from typing import Callable


class ScanTests(InferenceTestBase):
    """
    Intended to be used as a mixin
    """

    def test_repeated_exp(self):
        length = 5

        op = ir.Scan(ir.Exp(), length, in_axes=[])
        # out = get_numpyro_val(op, 0.1, is_observed=False)

        last = 0.1
        expected = []
        for i in range(length):
            last = np.exp(last)
            expected.append(last)
        expected = np.array(expected)

        # assert np.allclose(out, expected)

        x = ir.RV(ir.Constant(0.1))
        y = ir.RV(op, x)

        out = self.sample_flat([y], [], [], num_samples=1)
        assert np.allclose(expected, out[0])

    def test_repeated_exp_with_dummy(self):
        length = 5

        op = ir.Scan(ir.Exp(), length, in_axes=[])
        # out = get_numpyro_val(op, 0.1, is_observed=False)

        last = 0.1
        expected = []
        for i in range(length):
            last = np.exp(last)
            expected.append(last)
        expected = np.array(expected)

        x = ir.RV(ir.Constant(0.1))
        y = ir.RV(op, x)
        dummy = ir.RV(ir.Normal(), x, x)  # so there's something to sample!

        def testfun(samps):
            [ys] = samps
            return all(np.allclose(samp, expected) for samp in ys)

        test_util.inf_until_match(self.sample_flat, [y], [dummy], [1.5], testfun)

    def test_autoregressive_simple(self):
        x = pi.constant(0.5)
        length = 12
        y = pi.scan(lambda last: pi.normal(last + 1, 1e-4), length)(x)
        assert isinstance(y.op, ir.Scan)

        def testfun(samps):
            [ys] = samps
            last_y = ys[-1, :]
            expected = 0.5 + np.arange(1, length + 1)
            return np.max(np.abs(last_y - expected)) < 0.1

        test_util.inf_until_match(self.sample_flat, [y], [], [], testfun)

    def test_autoregressive_simple_const_rv(self):
        x = pi.constant(0.5)
        length = 12
        noise = pi.constant(1e-4)
        y = pi.scan(lambda last: pi.normal(last + 1, noise), length)(x)

        assert isinstance(y.op, ir.Scan)
        base_op = y.op.base_op
        assert isinstance(base_op, ir.Composite)
        assert base_op == ir.Composite(
            2,
            (
                ir.Constant(1),
                ir.Add(),
                ir.Normal(),
            ),
            [[], [0, 2], [3, 1]],
        )
        assert y.op == ir.Scan(base_op, length, [None])

        def testfun(samps):
            [ys] = samps
            last_y = ys[-1, :]
            expected = 0.5 + np.arange(1, length + 1)
            return np.max(np.abs(last_y - expected)) < 0.1

        test_util.inf_until_match(self.sample_flat, [y], [], [], testfun)

    def test_autoregressive_nonrandom(self):
        x = pi.normal(0.0, 1e-5)
        length = 12
        y = pi.scan(lambda last: last + 1, length)(x)

        assert isinstance(y.op, ir.Scan)

        def testfun(samps):
            [ys] = samps
            last_y = ys[-1, :]
            expected = np.arange(1, length + 1)
            return np.max(np.abs(last_y - expected)) < 0.1

        test_util.inf_until_match(self.sample_flat, [y], [], [], testfun)

    def test_autoregressive_varying_increments(self):
        x = pi.constant(0.0)
        length = 12
        increment = np.random.randn(length)
        increment_rv = pi.constant(increment)
        for y in [
            pi.scan(lambda last, inc: pi.normal(last + inc, 1e-4), length)(x, increment_rv),
            pi.scan(lambda last, inc: pi.normal(last + inc, 1e-4), length, in_axes=0)(
                x,
                increment_rv,
            ),
        ]:

            assert isinstance(y.op, ir.Scan)

            def testfun(samps):
                [ys] = samps
                last_y = ys[-1, :]
                expected = np.cumsum(increment)
                return np.max(np.abs(last_y - expected)) < 0.1

            test_util.inf_until_match(self.sample_flat, [y], [], [], testfun)

    def test_autoregressive_matmul(self):
        ndim = 5
        length = 3
        x0 = np.random.randn(ndim)
        x = pi.vmap(pi.normal, [0, None], ndim)(pi.constant(x0), pi.constant(1e-5))
        A = np.random.randn(ndim, ndim)
        # RV.__array_priority__ makes NumPy defer to RV.__rmatmul__, which returns an RV.
        y = pi.scan(lambda last: A @ last, length)(x)  # pyright: ignore[reportArgumentType]

        def testfun(samps):
            [ys] = samps
            last_y = ys[-1, :]
            assert last_y.shape == (length, ndim)
            out = last_y[-1, :]
            expected = x0
            for i in range(length):
                expected = A @ expected
            return np.max(np.abs(out - expected)) < 0.1

        test_util.inf_until_match(self.sample_flat, [y], [], [], testfun)

    def test_autoregressive_matmul_A_rv(self):
        ndim = 5
        length = 3
        x0 = np.random.randn(ndim)
        x = pi.vmap(pi.normal, [0, None], ndim)(pi.constant(x0), pi.constant(1e-5))
        A = np.random.randn(ndim, ndim)
        A_rv = pi.constant(A)
        y = pi.scan(lambda last: A_rv @ last, length)(x)

        def testfun(samps):
            [ys] = samps
            last_y = ys[-1, :]
            assert last_y.shape == (length, ndim)
            out = last_y[-1, :]
            expected = x0
            for i in range(length):
                expected = A @ expected
            return np.max(np.abs(out - expected)) < 0.1

        test_util.inf_until_match(self.sample_flat, [y], [], [], testfun)

    def test_double_autoregressive(self):
        randwalk = pi.scan(lambda last, input: pi.normal(last + input, 1e-7), 10)
        # randwalk is a function that takes a scalar RV and a length-10 RV and produces a length-10 RV

        z = randwalk(pi.constant(0.0), pi.constant(np.arange(10)))
        assert z.shape == (10,)
        [zs] = self.sample_flat([z], [], [], num_samples=1)
        assert zs.shape == (1, 10)

        vecwalk = pi.scan(lambda last: randwalk(pi.constant(0.0), last), 5)
        # vecwalk is a fnction that takes a length-10 RV and produces a 5x10 RV

        u = vecwalk(pi.constant(np.zeros(10)))
        assert u.shape == (5, 10)
        [us] = self.sample_flat([u], [], [], num_samples=1)
        assert us.shape == (1, 5, 10)


### belong as an interface test

# def test_autoregressive_const_rv_mapped():
#     x = coerce_to_rv(0.5)
#     length = 12
#     noises = coerce_to_rv(np.random.rand(length))
#     op = ir.Scan(ir.Normal(), length, (0,))
#     y = ir.RV(op, x, noises)
#     ys = sample(y)
#     #print(f"{ys=}")

# def test_autoregressive_const_rv_unmapped():
#     x = coerce_to_rv(0.5)
#     length = 12
#     noise = coerce_to_rv(1e-4)
#     op = ir.Scan(ir.Normal(), length, (None,))
#     y = ir.RV(op, x, noise)

#     ys = sample(y)
#     #print(f"{ys=}")


# def long_bernoulli_chain_expected(p,length,x_obs):
#     m = [np.array([1-p,p])]
#     for i in range(length):
#         #p0 = .95 * m[-1][0] + .05 * m[-1][1]
#         #p1 = .05 * m[-1][0] + .95 * m[-1][1]
#         #m.append(np.array([p0, p1]))
#         q = .05 * m[-1][0] + .95 * m[-1][1]
#         m.append(np.array([1-q, q]))
#     m = np.array(m)

#     if x_obs is None:
#         n = 1
#     else:

#         if x_obs == 0:
#             q = .05
#         else:
#             q = .95
#         n = [np.array([1-q, q])]
#         for i in range(length):
#             q = .05 * n[-1][0] + .95 * n[-1][1]
#             n.append(np.array([1 - q, q]))
#         n = np.array(list(reversed(n)))

#     probs = m*n
#     probs = probs / np.sum(probs, axis=1, keepdims=True)
#     return probs[:,1]

# # def test_discrete_autoregressive_ancestor():
# #     p = np.random.rand()
# #     length = 5
# #     #x_obs = np.random.randint(2)
# #
# #     expected_Ez = long_bernoulli_chain_expected(p, length, None)
# #
# #     z0 = bernoulli(p)
# #     z = autoregressive(lambda last: bernoulli(0.05 + 0.9 * last), length)(z0)
# #     x = bernoulli(0.05 + 0.9 * z[-1])
# #
# #     def testfun(expectations):
# #         Ez0, Ez = expectations
# #         Ez = np.append(Ez0, Ez)
# #         print(f"         {Ez=}")
# #         print(f"{expected_Ez=}")
# #         return np.all(np.abs(Ez - expected_Ez) < .01)
# #
# #     inf_until_match(E, (z0,z), [], [], testfun)

# def test_latent_discrete_autoregressive_forbidden():
#     p = np.random.rand()
#     length = 5
#     x_obs = np.random.randint(2)

#     z0 = bernoulli(p)
#     z = autoregressive(lambda last: bernoulli(0.05 + 0.9 * last), length)(z0)
#     x = bernoulli(0.05 + 0.9 * z[-1])

#     try:
#         E(z,x,x_obs)
#         assert False
#     except ValueError as e:
#         assert str(e).startswith("Can't have non-observed autoregressive over discrete variables")
