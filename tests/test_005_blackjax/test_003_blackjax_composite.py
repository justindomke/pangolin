import numpyro.handlers

# from pangolin.interface import *
from pangolin.blackjax import sample, E, inf_until_match
import numpy as np
from pangolin import ir


# off because no discrete vars

# def test_bernoulli():
#     op = ir.Composite(1, [ir.Bernoulli()], [[0]])

#     x = ir.RV(ir.Constant(0.7))
#     y = ir.RV(op, x)
#     assert isinstance(y.op, ir.Composite)

#     def testfun(E_y):
#         return np.abs(E_y - 0.7) < 0.01

#     inf_until_match(E, y, [], [], testfun)


# off because no constrained vars (yet)

# def test_exponential():
#     op = ir.Composite(1, [ir.Exponential()], [[0]])

#     x = ir.RV(ir.Constant(0.7))
#     y = ir.RV(op, x)
#     assert isinstance(y.op, ir.Composite)

#     def testfun(E_y):
#         return np.abs(E_y - 1 / 0.7) < 0.01

#     inf_until_match(E, y, [], [], testfun)


def test_add_normal():
    # z ~ Normal(x+y, y)
    op = ir.Composite(2, [ir.Add(), ir.Normal()], [[0, 1], [2, 1]])

    x = ir.RV(ir.Constant(0.3))
    y = ir.RV(ir.Constant(0.1))
    z = ir.RV(op, x, y)

    def testfun(E_z):
        return np.abs(E_z - 0.4) < 0.01

    inf_until_match(E, z, [], [], testfun)
