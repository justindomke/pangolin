from cleanpangolin.interface.vmap import (
    convert_args,
    AbstractRV,
    generated_nodes,
    vmap_dummy_args,
    AbstractOp,
    vmap_eval,
)
from cleanpangolin.interface import *
from cleanpangolin import ir
from collections.abc import Callable


def test_convert_args_independent():
    x0 = normal(0, 1)
    y0 = normal(2, 3)
    x1, y1 = convert_args(AbstractRV, x0, y0)
    assert isinstance(x1, AbstractRV)
    assert isinstance(y1, AbstractRV)
    assert x1.parents == x0.parents
    assert y1.parents == y0.parents
    assert x1.op == ir.Normal()
    assert y1.op == ir.Normal()


def test_get_abstract_args_dependent():
    x0 = normal(0, 1)
    y0 = normal(x0, 2)
    x1, y1 = convert_args(AbstractRV, x0, y0)
    assert isinstance(x1, AbstractRV)
    assert isinstance(y1, AbstractRV)
    assert x1.parents == x0.parents
    assert y1.parents == (x1, y0.parents[1])
    assert x1.op == ir.Normal()
    assert y1.op == ir.Normal()


def test_get_abstract_args_closure():
    z = normal(0, 1) ** 2
    x0 = normal(0, z)
    y0 = normal(x0, z)
    x1, y1 = convert_args(AbstractRV, x0, y0)
    assert isinstance(x1, AbstractRV)
    assert isinstance(y1, AbstractRV)
    assert x1.parents == (x0.parents[0], z)
    assert y1.parents == (x1, z)
    assert x1.op == ir.Normal()
    assert y1.op == ir.Normal()


def test_generated_nodes1():
    def fun(x, y):
        return [x * 2, y + 3]

    x0 = makerv(1)
    y0 = makerv(2)
    generated, out = generated_nodes(fun, x0, y0)
    assert len(generated) == 2
    assert generated[0].op == ir.Mul()
    assert generated[0].parents[0] == x0
    assert generated[0].parents[1].op == Constant(2)
    assert generated[1].op == ir.Add()
    assert generated[1].parents[0] == y0
    assert generated[1].parents[1].op == Constant(3)


def test_generated_nodes2():
    def fun(x, y):
        tmp = normal(x, y)
        z = cauchy(tmp, y)
        return [z]

    x0 = makerv(1)
    y0 = makerv(2)
    generated, out = generated_nodes(fun, x0, y0)
    assert len(generated) == 2
    # tmp
    assert generated[0].op == ir.Normal()
    assert generated[0].parents == (x0, y0)
    # z
    assert generated[1].op == ir.Cauchy()
    assert generated[1].parents == (generated[0], y0)
    # z again
    assert out[0] == generated[1]


def test_generated_nodes_closure():
    x = makerv(1)

    def fun(y):
        tmp = x * 3  # shouldn't be included
        z = y + tmp
        return [z]

    y0 = makerv(2)
    generated, out = generated_nodes(fun, y0)
    assert len(generated) == 1
    # z
    z = generated[0]
    assert z.op == ir.Add()
    assert z.parents[0] == y0
    # tmp
    tmp = z.parents[1]
    assert tmp.op == ir.Mul()
    assert tmp.parents[0] == x
    assert tmp.parents[1].op == Constant(3)
    # z again
    assert out[0] == z


def test_generated_nodes_ignored_input():
    x = makerv(1)

    def fun(y):
        return [2 * x]

    y0 = makerv(3)
    try:
        # should fail because output independent of y
        generated, out = generated_nodes(fun, y0)
        assert False
    except ValueError:
        pass
    #assert generated == []
    #assert out[0].op == ir.Mul()
    #assert out[0].parents[0].op == Constant(2)
    #assert out[0].parents[1] == x


def test_generated_nodes_passthrough():
    x = makerv(1)

    def fun(x):
        return [x]

    try:
        generated, out = generated_nodes(fun, x)
        assert False
    except ValueError:
        pass


def test_generated_nodes_switcharoo():
    def fun(x, y):
        return [y, x]

    x0 = makerv(1)
    y0 = makerv(2)

    try:
        generated, out = generated_nodes(fun, x0, y0)
        assert False
    except ValueError:
        pass


def test_non_flat():
    def fun(x, y):
        return x

    x0 = makerv(1)
    y0 = makerv(2)

    try:
        generated_nodes(fun, x0, y0)
        assert False
    except ValueError:
        pass


def test_vmap_dummy_args():
    in_axes = (0,)
    axis_size = None
    args = (makerv([1, 2, 3]),)
    dummy_args, axis_size = vmap_dummy_args(in_axes, axis_size, *args)
    assert len(dummy_args) == 1
    assert axis_size == 3
    assert dummy_args[0].shape == ()


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
