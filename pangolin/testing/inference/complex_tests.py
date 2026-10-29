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


class ComplexTests(InferenceTestBase):
    """
    Intended to be used as a mixin
    """

    def test_scan(self):
        # warmup
        increasing = pi.scan(lambda x: x + 1, 10)
        y = increasing(pi.constant(0))

        expected = np.arange(1, 11)

        [y_samps] = self.sample_flat([y], [], [], num_samples=1)  # type:ignore
        out = y_samps[0]

        assert out.shape == expected.shape
        assert np.allclose(expected, out)

    def test_scan_mapped(self):
        # warmup
        a = np.random.randn(10)
        increasing = pi.scan(lambda x, ai: x + ai, 10)
        y = increasing(pi.constant(0.0), pi.constant(a))

        expected = np.cumsum(a)

        [y_samps] = self.sample_flat([y], [], [], num_samples=1)  # type:ignore
        out = y_samps[0]

        assert out.shape == expected.shape
        assert np.allclose(expected, out)

    def test_vmap_scan_deterministic(self):
        increasing = pi.scan(lambda x: x + 1, 10)
        y = pi.vmap(increasing)(pi.constant([1, 2, 3, 4, 5]))

        expected = np.array([1, 2, 3, 4, 5])[:, None] + np.arange(1, 11)[None, :]

        [y_samps] = self.sample_flat([y], [], [], num_samples=1)  # type:ignore
        out = y_samps[0]

        assert out.shape == expected.shape
        assert np.allclose(expected, out)

    def test_vmap_scan_deterministic_mapped(self):
        a = np.random.randn(10)

        for increasing in [
            pi.scan(lambda x, ai: x + ai, 10, in_axes=0),
            pi.scan(lambda x, ai: x + ai, 10),
        ]:
            y = pi.vmap(increasing, [0, None])(
                pi.constant([1.0, 2, 3, 4, 5]),
                pi.constant(a),
            )

            expected = np.array([1, 2, 3, 4, 5])[:, None] + np.cumsum(a)[None, :]

            [y_samps] = self.sample_flat([y], [], [], num_samples=1)  # type:ignore
            out = y_samps[0]

            assert out.shape == expected.shape
            assert np.allclose(expected, out)

    def test_vmap_scan_deterministic_mapped2(self):
        a = np.random.randn(5, 10)

        for increasing in [
            pi.scan(lambda x, ai: x + ai, 10, in_axes=0),
            pi.scan(lambda x, ai: x + ai, 10),
        ]:
            initial = pi.constant([1.0, 2, 3, 4, 5])
            increments = pi.constant(a)
            for y in [
                pi.vmap(increasing, 0)(initial, increments),
                pi.vmap(increasing)(initial, increments),
                pi.vmap(increasing, axis_size=5)(initial, increments),
            ]:
                expected = np.array([1, 2, 3, 4, 5])[:, None] + np.cumsum(a, axis=1)

                [y_samps] = self.sample_flat([y], [], [], num_samples=1)  # type:ignore
                out = y_samps[0]

                assert out.shape == expected.shape
                assert np.allclose(expected, out, rtol=1e-5, atol=1e-5)

    def test_scan_vmap(self):
        a = np.random.randn(5)
        b = np.random.randn(5)

        vmap_add = pi.vmap(pi.add)
        a_rv = pi.constant(a)

        y = pi.scan(lambda carry: vmap_add(carry, a_rv), length=10)(pi.constant(b))

        expected = b[None, :] + np.cumsum(a[None, :] + np.zeros(10)[:, None], axis=0)

        [y_samps] = self.sample_flat([y], [], [], num_samples=1)  # type:ignore
        out = y_samps[0]

        assert y.shape == out.shape == expected.shape == (10, 5)
        assert np.allclose(expected, out)

    def test_scan_vmap_mapped(self):
        a = np.random.randn(10, 5)
        b = np.random.randn(5)

        vmap_add = pi.vmap(pi.add)

        y = pi.scan(lambda carry, a_i: vmap_add(carry, a_i), length=10, in_axes=0)(
            pi.constant(b),
            pi.constant(a),
        )

        expected_op = ir.Scan(ir.VMap(ir.Add(), (0, 0), 5), 10, (0,))

        assert y.op == expected_op

        expected = b[None, :] + np.cumsum(a, axis=0)

        [y_samps] = self.sample_flat([y], [], [], num_samples=1)
        out = y_samps[0]

        assert y.shape == out.shape == expected.shape == (10, 5)
        assert np.allclose(expected, out, rtol=1e-5, atol=1e-5)

    # def test_autoregressive_vmap_mapped2(self):
    #     "test currently unsupported because jax backend can't handle in_axes > 0"

    #     a = np.random.randn(5, 10)
    #     b = np.random.randn(5)

    #     # autoregressive interface can't handle this
    #     # vmap_add = pi.vmap(pi.add)
    #     # y = pi.autoregressive(lambda carry, a_i: vmap_add(carry, a_i), length=10, in_axes=1)(b, a)

    #     # this shuould be the correct op

    #     # fmt: off
    #     expected_op = ir.Autoregressive(
    #                     ir.Composite(
    #                         2,
    #                         (
    #                             ir.VMap(
    #                                 ir.Add(),
    #                                 (0, 0),
    #                                 5),
    #                         ),
    #                         ((0, 1),)
    #                     ),
    #                     10,
    #                     (1,),
    #                     0)
    #     # fmt: on

    #     y = ir.RV(expected_op, pi.constant(b), pi.constant(a))

    #     expected = b[None, :] + np.cumsum(a.T, axis=0)

    #     [y_samps] = self.sample_flat([y], [], [], niter=1)  # type:ignore
    #     out = y_samps[0]

    #     assert y.shape == out.shape == expected.shape == (10, 5)
    #     assert np.allclose(expected, out)

    def test_downstream_sampling(self):
        """
        Simple test that you can condition upstream
        """

        x = pi.normal(0, 1)
        y = pi.normal(x, 1)

        def testfun(x_samps):
            E_x = np.mean(x_samps)
            return np.abs(E_x - 3.1) < 0.01

        test_util.inf_until_match(self.sample_flat, [y], [x], [3.1], testfun)
