from cleanpangolin.interface import *
from cleanpangolin import ir
from collections.abc import Callable

from cleanpangolin.interface.vmap import (
    convert_args,
    generated_nodes,
    vmap_dummy_args,
    AbstractOp,
    vmap_eval,
    vmap
)
import numpy as np

def test_vmap_eval_simple():
    def flat_fun(x, y):
        return [x * y]

    in_axes = (0, 0)
    axis_size = 3
    x = makerv([1, 2, 3])
    y = makerv([4, 5, 6])
    [z] = vmap_eval(flat_fun, in_axes, axis_size, x, y)

    assert z.op == ir.VMap(ir.Mul(), (0, 0), 3)
    assert z.parents == (x, y)


def test_vmap_eval_more():
    def flat_fun(x, y):
        tmp1 = x * x
        tmp2 = y + y
        return [tmp1**tmp2, tmp1, tmp2]

    in_axes = (0, 0)
    axis_size = 3
    x = makerv([1, 2, 3])
    y = makerv([4, 5, 6])
    [z, tmp1, tmp2] = vmap_eval(flat_fun, in_axes, axis_size, x, y)
    assert z.op == ir.VMap(ir.Pow(), (0,0), 3)
    assert z.parents == (tmp1, tmp2)
    assert tmp1.op == ir.VMap(ir.Mul(), (0,0), 3)
    assert tmp1.parents == (x,x)
    assert tmp2.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert tmp2.parents == (y, y)

def test_vmap_eval_half_mapped():
    def flat_fun(x, y):
        tmp1 = x * x
        tmp2 = y + y
        return [tmp1**tmp2, tmp1, tmp2]

    in_axes = (0, None)
    axis_size = 3
    x = makerv([1, 2, 3])
    y = makerv(4)
    [z, tmp1, tmp2] = vmap_eval(flat_fun, in_axes, axis_size, x, y)
    assert z.op == ir.VMap(ir.Pow(), (0,0), 3)
    assert z.parents == (tmp1, tmp2)
    assert tmp1.op == ir.VMap(ir.Mul(), (0,0), 3)
    assert tmp1.parents == (x,x)
    assert tmp2.op == ir.VMap(ir.Add(), (None, None), 3)
    assert tmp2.parents == (y, y)

def test_vmap_eval_closure():
    z = normal(0,1)
    def flat_fun(xi):
        y = z*z
        return [y]
    in_axes = (0,)
    axis_size = 3
    x = makerv([1,2,3])
    try:
        [y] = vmap_eval(flat_fun, in_axes, axis_size, x)
        assert False
    except ValueError:
        pass


def test_vmap_eval_constant():
    # test that constant is not vmapped (used to require a special case)
    def flat_fun(xi):
        yi = makerv(2)
        return [xi * yi]

    in_axes = (0,)
    axis_size = 3
    x = makerv([1, 2, 3])
    [z] = vmap_eval(flat_fun, in_axes, axis_size, x)
    assert z.op == ir.VMap(ir.Mul(), (0,None), 3)
    assert z.parents[0] == x
    assert z.parents[1].op == Constant(2)


def test_no_redundant_deterministic():
    def flat_fun(xi):
        yi = xi * 2
        zi = yi + 3
        return [zi]

    in_axes = (None,)
    axis_size = 3
    x = makerv(1)
    [z] = vmap_eval(flat_fun, in_axes, axis_size, x)
    assert z.op == ir.VMap(ir.Add(), (None,None), 3)
    assert z.parents[0].op == ir.Mul()
    assert z.parents[1].op == ir.Constant(3)
    assert z.parents[0].parents[0] == x
    assert z.parents[0].parents[1].op == Constant(2)



def test_double_vmap_eval():
    def flat_fun1(xij):
        return [exp(xij)]

    xij = makerv(2)
    [yij] = flat_fun1(xij)
    assert yij.op == ir.Exp()

    def vec_fun1(xi):
        [yi] = vmap_eval(flat_fun1, (0,), None, xi)
        return [yi]

    xi = makerv([1,2,3])
    [yi] = vec_fun1(xi)
    assert yi.op == ir.VMap(ir.Exp(),(0,),3)

    def vec_fun2(x):
        [y] = vmap_eval(vec_fun1, (0,), None, x)
        return [y]

    x = makerv([[1,2,3],[4,5,6]])
    [y] = vec_fun2(x)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(),(0,),3),(0,),2)