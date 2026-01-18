from pangolin.ir import *
import numpyro.handlers
import numpy as np
import pytest
import scipy.special
import jax
from .base import MixinBase


def rands_from_ranges(ranges):
    dims = np.random.choice([2, 5, 10])  # for matrices / vectors
    out = []
    for domain in ranges:
        if domain == "real":
            new = np.random.randn()
        elif domain == "positive":
            new = np.abs(np.random.randn())
        elif domain == "vector":
            new = np.random.randn(dims)
        elif domain == "matrix":
            new = np.random.randn(dims, dims)
        elif domain == "rectangular-matrix":
            new = np.random.randn(dims, dims + 2)
        elif domain == "positive-definite":
            factor = np.random.randn(dims, dims)
            new = factor @ factor.T
        elif isinstance(domain, tuple):
            lo, hi = domain
            new = lo + np.random.rand() * (hi - lo)
        else:
            raise NotImplementedError()
        out.append(new)
    return out


testdata = [
    (Add, lambda a, b: a + b, ["real", "real"]),
    (Sub, lambda a, b: a - b, ["real", "real"]),
    (Mul, lambda a, b: a * b, ["real", "real"]),
    (Div, lambda a, b: a / b, ["real", "real"]),
    (Pow, lambda a, b: a**b, ["positive", "real"]),
    (Abs, np.abs, ["real"]),
    (Cos, np.cos, ["real"]),
    (Sin, np.sin, ["real"]),
    (Tan, np.tan, ["real"]),
    (Arccos, np.arccos, [(-0.999, 0.999)]),
    (Arcsin, np.arcsin, [(-0.999, 0.999)]),
    (Arctan, np.arctan, ["real"]),
    (Arccosh, np.arccosh, [(1, 100)]),
    (Arcsinh, np.arcsinh, [(-100, 100)]),
    (Arctanh, np.arctanh, [(-0.999, 0.999)]),
    (Exp, np.exp, ["real"]),
    (InvLogit, scipy.special.expit, ["real"]),
    (Log, np.log, ["positive"]),
    (Loggamma, scipy.special.loggamma, ["positive"]),
    (Logit, scipy.special.logit, [(0, 1)]),
    (Step, lambda a: np.heaviside(a, 0.5), ["real"]),
    (Matmul, np.dot, ["vector", "vector"]),
    (Matmul, np.dot, ["vector", "matrix"]),
    (Matmul, np.dot, ["matrix", "vector"]),
    (Matmul, np.dot, ["matrix", "matrix"]),
    (Inv, np.linalg.inv, ["matrix"]),
    (Cholesky, np.linalg.cholesky, ["positive-definite"]),
    (Transpose, np.transpose, ["rectangular-matrix"]),
    (Identity, lambda a: a, ["real"]),
    (Diag, np.diag, ["matrix"]),
    (DiagMatrix, np.diag, ["vector"]),
]


class DeterministicTests(MixinBase):
    """
    Intended to be used as a mixin
    """

    @pytest.mark.parametrize("pangolin_op, numpy_fun, ranges", testdata)
    def test_deterministic_op(self, pangolin_op, numpy_fun, ranges):
        for reps in range(5):
            inputs = rands_from_ranges(ranges)

            input_rvs = [RV(Constant(x)) for x in inputs]
            output_rv = RV(pangolin_op(), *input_rvs)

            output_pangolin = self.ancestor_sample_flat([output_rv], None)
            output_numpy = numpy_fun(*inputs)
            assert np.allclose(output_pangolin, output_numpy, atol=1e-5, rtol=1e-5)
