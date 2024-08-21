from pangolin.interface import *
from pangolin import ir
from collections.abc import Callable

from pangolin.interface.vmap import (
    convert_args,
    generated_nodes,
    vmap_dummy_args,
    AbstractOp,
    # vmap_eval,
    vmap,
    vmap_flat,
)
import numpy as np


def test_simple():
    def flat_fun(x, y):
        return [x * y]

    in_axes = (0, 0)
    axis_size = 3
    x = makerv([1, 2, 3])
    y = makerv([4, 5, 6])
    [z] = vmap_flat(flat_fun, in_axes, axis_size)(x, y)

    assert z.op == ir.VMap(ir.Mul(), (0, 0), 3)
    assert z.parents == (x, y)


def test_more():
    def flat_fun(x, y):
        tmp1 = x * x
        tmp2 = y + y
        return [tmp1**tmp2, tmp1, tmp2]

    in_axes = (0, 0)
    axis_size = 3
    x = makerv([1, 2, 3])
    y = makerv([4, 5, 6])
    [z, tmp1, tmp2] = vmap_flat(flat_fun, in_axes, axis_size)(x, y)
    assert z.op == ir.VMap(ir.Pow(), (0, 0), 3)
    assert z.parents == (tmp1, tmp2)
    assert tmp1.op == ir.VMap(ir.Mul(), (0, 0), 3)
    assert tmp1.parents == (x, x)
    assert tmp2.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert tmp2.parents == (y, y)


def test_half_mapped():
    def flat_fun(x, y):
        tmp1 = x * x
        tmp2 = y + y
        return [tmp1**tmp2, tmp1, tmp2]

    in_axes = (0, None)
    axis_size = 3
    x = makerv([1, 2, 3])
    y = makerv(4)
    [z, tmp1, tmp2] = vmap_flat(flat_fun, in_axes, axis_size)(x, y)
    assert z.op == ir.VMap(ir.Pow(), (0, 0), 3)
    assert z.parents == (tmp1, tmp2)
    assert tmp1.op == ir.VMap(ir.Mul(), (0, 0), 3)
    assert tmp1.parents == (x, x)
    assert tmp2.op == ir.VMap(ir.Add(), (None, None), 3)
    assert tmp2.parents == (y, y)


def test_closure():
    z = normal(0, 1)

    def flat_fun(xi):
        y = z * z
        return [y]

    in_axes = (0,)
    axis_size = 3
    x = makerv([1, 2, 3])
    [y] = vmap_flat(flat_fun, in_axes, axis_size)(x)
    assert y.op == ir.VMap(ir.Mul(), (None, None), 3)
    assert y.parents == (z, z)


def test_constant():
    # test that constant is not vmapped (used to require a special case)
    def flat_fun(xi):
        yi = makerv(2)
        return [xi * yi]

    in_axes = (0,)
    axis_size = 3
    x = makerv([1, 2, 3])
    [z] = vmap_flat(flat_fun, in_axes, axis_size)(x)
    assert z.op == ir.VMap(ir.Mul(), (0, None), 3)
    assert z.parents[0] == x
    assert z.parents[1].op == ir.Constant(2)


def test_no_redundant_deterministic():
    def flat_fun(xi):
        yi = xi * 2
        zi = yi + 3
        return [zi]

    in_axes = (None,)
    axis_size = 3
    x = makerv(1)
    [z] = vmap_flat(flat_fun, in_axes, axis_size)(x)
    assert z.op == ir.VMap(ir.Add(), (None, None), 3)
    assert z.parents[0].op == ir.Mul()
    assert z.parents[1].op == ir.Constant(3)
    assert z.parents[0].parents[0] == x
    assert z.parents[0].parents[1].op == ir.Constant(2)


def test_double():
    def flat_fun1(xij):
        return [exp(xij)]

    xij = makerv(2)
    [yij] = flat_fun1(xij)
    assert yij.op == ir.Exp()

    def vec_fun1(xi):
        [yi] = vmap_flat(flat_fun1, (0,), None)(xi)
        return [yi]

    xi = makerv([1, 2, 3])
    [yi] = vec_fun1(xi)
    assert yi.op == ir.VMap(ir.Exp(), (0,), 3)

    def vec_fun2(x):
        [y] = vmap_flat(vec_fun1, (0,), None)(x)
        return [y]

    x = makerv([[1, 2, 3], [4, 5, 6]])
    [y] = vec_fun2(x)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), (0,), 3), (0,), 2)


def test_double_nicer():
    def flat_fun1(xij):
        return [exp(xij)]

    xij = makerv(2)
    [yij] = flat_fun1(xij)
    assert yij.op == ir.Exp()

    vec_fun1 = vmap_flat(flat_fun1, (0,), None)
    vec_fun2 = vmap_flat(vec_fun1, (0,), None)

    x = makerv([[1, 2, 3], [4, 5, 6]])
    [y] = vec_fun2(x)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), (0,), 3), (0,), 2)


def test_1():
    "should fail because of incoherent axes sizes"
    try:
        y = vmap_flat(lambda loc, scale: normal(loc, scale), (None, None), 5)(
            np.zeros(3),
            np.ones(3),
        )
        assert False
    except AssertionError as e:
        assert True


def test_2():
    [y] = vmap_flat(lambda loc, scale: [normal(loc, scale)], (0, None), 3)(np.zeros(3), 1)
    assert y.shape == (3,)


def test_3():
    def f(x):
        return [normal(x, x)]

    [y] = vmap_flat(f, (0,), None)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)


def test_4():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        return [normal(loc, scale)]

    [y] = vmap_flat(f, (0,), None)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)


def test_6():
    def f():
        return [normal(0, 1)]

    [x] = vmap_flat(f, (), 3)()
    assert x.shape == (3,)


def test_7():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return [y, x, z]

    y, x, z = vmap_flat(f, (0,), 3)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert x.shape == (3,)
    assert z.shape == (3,)


def test_8():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return [y, x, z]

    y, x, z = vmap_flat(f, (0,), None)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert x.shape == (3,)
    assert z.shape == (3,)
