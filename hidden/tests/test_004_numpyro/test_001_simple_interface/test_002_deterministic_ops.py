from pangolin.interface import *
from pangolin.inference.numpyro.model import get_model_flat
import numpyro.handlers
import numpy as np
import pytest
import scipy.special


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
        elif isinstance(domain, tuple):
            lo, hi = domain
            new = lo + np.random.rand() * (hi - lo)
        else:
            raise NotImplementedError()
        out.append(new)
    return out


testdata = [
    (add, lambda a, b: a + b, ["real", "real"]),
    (sub, lambda a, b: a - b, ["real", "real"]),
    (mul, lambda a, b: a * b, ["real", "real"]),
    (div, lambda a, b: a / b, ["real", "real"]),
    (pow, lambda a, b: a**b, ["positive", "real"]),
    (sqrt, np.sqrt, ["positive"]),
    (abs, np.abs, ["real"]),
    (cos, np.cos, ["real"]),
    (sin, np.sin, ["real"]),
    (tan, np.tan, ["real"]),
    (arccos, np.arccos, [(-0.999, 0.999)]),
    (arcsin, np.arcsin, [(-0.999, 0.999)]),
    (arctan, np.arctan, ["real"]),
    (arccosh, np.arccosh, [(1, 100)]),
    (arcsinh, np.arcsinh, [(-100, 100)]),
    (arctanh, np.arctanh, [(-0.999, 0.999)]),
    (exp, np.exp, ["real"]),
    (inv_logit, scipy.special.expit, ["real"]),
    (expit, scipy.special.expit, ["real"]),
    (sigmoid, scipy.special.expit, ["real"]),
    (log, np.log, ["positive"]),
    (loggamma, scipy.special.loggamma, ["positive"]),
    (logit, scipy.special.logit, [(0, 1)]),
    (step, lambda a: np.heaviside(a, 0.5), ["real"]),
    (matmul, np.dot, ["vector", "vector"]),
    (matmul, np.dot, ["vector", "matrix"]),
    (matmul, np.dot, ["matrix", "vector"]),
    (matmul, np.dot, ["matrix", "matrix"]),
    (inv, np.linalg.inv, ["matrix"]),
]


@pytest.mark.parametrize("pangolin_fun, numpy_fun, ranges", testdata)
def test_op(pangolin_fun, numpy_fun, ranges):
    for reps in range(5):
        inputs = rands_from_ranges(ranges)
        output_rv = pangolin_fun(*inputs)
        model, var_to_name = get_model_flat([output_rv], [], [])
        with numpyro.handlers.seed(rng_seed=0):
            samples = model()
        output_pangolin = samples[var_to_name[output_rv]]
        output_numpy = numpy_fun(*inputs)
        assert np.allclose(output_pangolin, output_numpy, atol=1e-5, rtol=1e-5)
