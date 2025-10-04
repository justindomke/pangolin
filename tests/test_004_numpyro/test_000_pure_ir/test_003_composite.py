import numpyro.handlers

# from pangolin.interface import *
from pangolin.inference.numpyro import (
    sample_flat,
    sample,
    E,
)
from pangolin.inference.numpyro.handlers import get_numpyro_val
import numpy as np
from pangolin import ir
from pangolin.inference.numpyro.test_util import inf_until_match


def test_add():
    # x -> x+x

    op = ir.Composite(1, [ir.Add()], [[0, 0]])

    x = ir.RV(ir.Constant(1.5))
    y = ir.RV(op, x)
    assert isinstance(y.op, ir.Composite)

    expected = 3.0
    out = get_numpyro_val(y.op, 1.5, is_observed=False)

    assert np.allclose(expected, out)


def test_add_mul():
    # x,y -> (x+x)*y
    op = ir.Composite(2, [ir.Add(), ir.Mul()], [[0, 0], [2, 1]])

    x = ir.RV(ir.Constant(3.3))
    y = ir.RV(ir.Constant(4.4))
    z = ir.RV(op, x, y)
    assert isinstance(z.op, ir.Composite)

    expected = (3.3 + 3.3) * 4.4
    out = get_numpyro_val(z.op, 3.3, 4.4, is_observed=False)

    assert np.allclose(expected, out)


# def test_bernoulli():
#     op = ir.Composite(1, [ir.Bernoulli()], [[0]])

#     x = ir.RV(ir.Constant(0.7))
#     y = ir.RV(op, x)
#     assert isinstance(y.op, ir.Composite)

#     def testfun(E_y):
#         return np.abs(E_y - 0.7) < 0.01

#     inf_until_match(E, y, [], [], testfun)


def test_exponential():
    op = ir.Composite(1, [ir.Exponential()], [[0]])

    x = ir.RV(ir.Constant(0.7))
    y = ir.RV(op, x)
    assert isinstance(y.op, ir.Composite)

    def testfun(E_y):
        return np.abs(E_y - 1 / 0.7) < 0.01

    inf_until_match(E, y, [], [], testfun)


def test_add_normal():
    # z ~ Normal(x+y, y)
    op = ir.Composite(2, [ir.Add(), ir.Normal()], [[0, 1], [2, 1]])

    x = ir.RV(ir.Constant(0.3))
    y = ir.RV(ir.Constant(0.1))
    z = ir.RV(op, x, y)

    def testfun(E_z):
        return np.abs(E_z - 0.4) < 0.01

    inf_until_match(E, z, [], [], testfun)
