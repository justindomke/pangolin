from pangolin import interface as pi, ir


import numpy as np
import pytest

from pangolin.interface import constant
from pangolin.interface import exp
from pangolin.interface import normal
from pangolin.functional.vmapping import vmap


def test_simple():
    def fun(x, y):
        return x * y

    x = constant([1, 2, 3])
    y = constant([4, 5, 6])
    z = vmap(fun, (0, 0), 3)(x, y)
    assert z.op == ir.VMap(ir.Mul(), (0, 0), 3)
    assert z.parents == (x, y)


def test_more():
    def fun(x, y):
        tmp1 = x * x
        tmp2 = y + y
        return tmp1**tmp2, tmp1, tmp2

    x = constant([1, 2, 3])
    y = constant([4, 5, 6])
    z, tmp1, tmp2 = vmap(fun, (0, 0), 3)(x, y)
    assert z.op == ir.VMap(ir.Pow(), (0, 0), 3)
    assert z.parents == (tmp1, tmp2)
    assert tmp1.op == ir.VMap(ir.Mul(), (0, 0), 3)
    assert tmp1.parents == (x, x)
    assert tmp2.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert tmp2.parents == (y, y)


def test_half_mapped():
    def fun(x, y):
        tmp1 = x * x
        tmp2 = y + y
        return tmp1**tmp2, tmp1, tmp2

    x = constant([1, 2, 3])
    y = constant(4)
    z, tmp1, tmp2 = vmap(fun, (0, None), 3)(x, y)
    assert z.op == ir.VMap(ir.Pow(), (0, 0), 3)
    assert z.parents == (tmp1, tmp2)
    assert tmp1.op == ir.VMap(ir.Mul(), (0, 0), 3)
    assert tmp1.parents == (x, x)
    assert tmp2.op == ir.VMap(ir.Add(), (None, None), 3)
    assert tmp2.parents == (y, y)


def test_closure():
    z = normal(0, 1)

    def fun(x):
        return z * z

    x = constant([1, 2, 3])
    y = vmap(fun, 0, 3)(x)
    assert y.op == ir.VMap(ir.Mul(), (None, None), 3)
    assert y.parents == (z, z)


def test_constant():
    def fun(x):
        y = constant(2)
        return x * y

    x = constant([1, 2, 3])
    z = vmap(fun, 0, 3)(x)
    assert z.op == ir.VMap(ir.Mul(), (0, None), 3)
    assert z.parents[0] == x
    assert z.parents[1].op == ir.Constant(2)


def test_no_redundant_deterministic():
    def fun(x):
        y = x * 2
        return y + 3

    x = constant(1)
    z = vmap(fun, None, 3)(x)
    assert z.op == ir.VMap(ir.Add(), (None, None), 3)
    assert z.parents[0].op == ir.Mul()
    assert z.parents[1].op == ir.Constant(3)
    assert z.parents[0].parents[0] == x
    assert z.parents[0].parents[1].op == ir.Constant(2)


def test_double():
    def fun(x):
        return exp(x)

    xij = constant(2)
    assert fun(xij).op == ir.Exp()

    vec_fun1 = vmap(fun, 0)
    xi = constant([1, 2, 3])
    yi = vec_fun1(xi)
    assert yi.op == ir.VMap(ir.Exp(), (0,), 3)

    vec_fun2 = vmap(vec_fun1, 0)
    x = constant([[1, 2, 3], [4, 5, 6]])
    y = vec_fun2(x)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), (0,), 3), (0,), 2)


def test_double_nicer():
    def fun(x):
        return exp(x)

    xij = constant(2)
    assert fun(xij).op == ir.Exp()

    vec_fun1 = vmap(fun, 0)
    vec_fun2 = vmap(vec_fun1, 0)
    x = constant([[1, 2, 3], [4, 5, 6]])
    y = vec_fun2(x)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), (0,), 3), (0,), 2)


def test_1():
    with pi.overload(broadcasting="off"):
        with pytest.raises(ValueError):
            vmap(lambda loc, scale: normal(loc, scale), (None, None), 5)(
                constant(np.zeros(3)), constant(np.ones(3))
            )


def test_2():
    y = vmap(lambda loc, scale: normal(loc, scale), (0, None), 3)(
        constant(np.zeros(3)), constant(1)
    )
    assert y.shape == (3,)


def test_3():
    y = vmap(lambda x: normal(x, x), 0)(constant(np.array([3.3, 4.4, 5.5])))
    assert y.shape == (3,)


def test_4():
    def fun(x):
        loc = x * 1.1
        scale = x**2.2
        return normal(loc, scale)

    y = vmap(fun, 0)(constant(np.array([3.3, 4.4, 5.5])))
    assert y.shape == (3,)


def test_6():
    x = vmap(lambda: normal(0, 1), (), 3)()
    assert x.shape == (3,)


def test_7():
    def fun(x):
        loc = x * 1.1
        scale = x**2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return y, x, z

    y, x, z = vmap(fun, 0, 3)(constant(np.array([3.3, 4.4, 5.5])))
    assert y.shape == (3,)
    assert x.shape == (3,)
    assert z.shape == (3,)


def test_8():
    def fun(x):
        loc = x * 1.1
        scale = x**2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return y, x, z

    y, x, z = vmap(fun, 0)(constant(np.array([3.3, 4.4, 5.5])))
    assert y.shape == (3,)
    assert x.shape == (3,)
    assert z.shape == (3,)
