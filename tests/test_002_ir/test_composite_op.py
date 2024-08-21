import pytest
from pangolin.ir.composite import Composite

# from pangolin.ir import *
from pangolin import ir
import numpy as np

fslice = slice(None)  # full slice


def test_plain_normal():
    op = Composite(2, (ir.Normal(),), ((0, 1),))
    assert op.get_shape((), ()) == ()
    assert op.random


def test_single_input():
    op = Composite(1, (ir.Normal(),), ((0, 0),))
    assert op.get_shape(()) == ()
    assert op.random


def test_diag_normal():
    # def f(x0,x1):
    #     x2 = np.arange(5)
    #     x3 = x1 * x2
    #     x4 = normal(x0, x3)
    op = Composite(
        2,
        (ir.Constant(np.arange(5)), ir.VMap(ir.Mul(), (None, 0)), ir.VMap(ir.Normal(), (0, 0))),
        ((), (1, 2), (0, 3)),
    )
    assert op.get_shape((5,), ()) == (5,)
    assert op.random


def test_multivariate_normal():
    "Multivariate normal with mean zero and cov a*I"
    scalar_matrix_mul = ir.VMap(ir.VMap(ir.Mul(),(None,0)),(None,0))

    op = Composite(1,(ir.Constant(np.ones(5)), ir.Constant(np.eye(5)), scalar_matrix_mul,
                      ir.MultiNormal()),
                   ((),(),(0,2),(1,3)))
    assert op.get_shape(()) == (5,)
    assert op.random

# def test_composite():
#     def f(a):
#         b = a * a
#         c = exponential(b)
#         return c
#
#     x = interface.normal(0, 1)
#     z = interface.composite(f)(x)
#     assert z.shape == ()
#     assert z.parents == (x,)
#     op = z.cond_dist
#     assert isinstance(op, Composite)
#     assert op.num_inputs == 1
#     assert op.cond_dists == [mul, exponential]
#     assert op.par_nums == [[0, 0], [1]]
#
#
# def test_composite_inside_vmap():
#     def f(a):
#         return normal_scale(a, a * a)
#
#     x = normal_scale(0, 1)
#     z = interface.vmap(interface.composite(f), None, 3)(x)
#     interface.print_upstream(z)
#
#
# def test_vmap_inside_composite():
#     def f(a):
#         b = a * a
#         return interface.vmap(normal_scale, None, 3)(a, b)
#
#     x = normal_scale(0, 1)
#     z = interface.composite(f)(x)
#     interface.print_upstream(z)
