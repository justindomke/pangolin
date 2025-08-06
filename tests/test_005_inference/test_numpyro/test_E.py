from pangolin.interface import *
from pangolin.inference.numpyro import (
    E,
    std,
)
import numpy as np
from util import inf_until_match, sample_until_match, sample_flat_until_match


def test_E():
    x = normal(2.5, 3)

    def testfun(Ex):
        return np.abs(Ex - 2.5) < 0.1

    inf_until_match(E, x, None, None, testfun)


def test_std():
    x = normal(2.5, 3)

    def testfun(std_x):
        return np.abs(std_x - 3) < 0.1

    inf_until_match(std, x, None, None, testfun)


def test_E_vector():
    loc = np.array([0, 1, 2])
    scale = np.array([2, 3, 4])

    x = vmap(normal)(loc, scale)

    def testfun(Ex):
        return np.linalg.norm(Ex - loc) < 0.1

    inf_until_match(E, x, None, None, testfun)


def test_E_matrix():
    loc = np.array([[0, 1, 2], [3, 4, 5]])
    scale = np.array([2, 3, 4])

    x = vmap(vmap(normal), [0, None])(loc, scale)

    def testfun(Ex):
        print(f"{Ex=}")
        return np.linalg.norm(Ex - loc) < 0.1

    inf_until_match(E, x, None, None, testfun)


def test_E_pytree():
    d = {}
    d["x"] = normal(1.5, 2)
    d["y"] = normal(3.3, 4)

    def testfun(E_d):
        return np.abs(E_d["x"] - 1.5) + np.abs(E_d["y"] - 3.3) < 0.1

    inf_until_match(E, d, None, None, testfun)


def test_std_pytree():
    d = {}
    d["x"] = normal(1.5, 2)
    d["y"] = normal(3.3, 4)

    def testfun(std_d):
        return np.abs(std_d["x"] - 2) + np.abs(std_d["y"] - 4) < 0.1

    inf_until_match(std, d, None, None, testfun)


def test_std_vector():
    loc = np.array([0, 1, 2])
    scale = np.array([2, 3, 4])

    x = vmap(normal)(loc, scale)

    def testfun(Ex):
        return np.linalg.norm(Ex - scale) < 0.1

    inf_until_match(std, x, None, None, testfun)
