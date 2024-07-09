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


def test_vmap_generated_nodes1():
    # both inputs explicitly given
    a = makerv(0)
    b = makerv(1)
    f = lambda a, b: [normal(a, b)]
    generated, out = generated_nodes(f, a, b)
    assert len(generated) == 1
    assert generated[0].op == ir.Normal()


def test_vmap_generated_nodes2():
    # b captured as a closure
    a = makerv(0)
    b = makerv(1)
    f = lambda a: [normal(a, b)]
    generated, out = generated_nodes(f, a)
    assert len(generated) == 1
    assert generated[0].op == ir.Normal()


def test_vmap_generated_nodes3():
    # a captured as a closure
    a = makerv(0)
    b = makerv(1)
    f = lambda b: [normal(a, b)]
    generated, out = generated_nodes(f, b)
    assert len(generated) == 1
    assert generated[0].op == ir.Normal()


# illegal under current rules
# def test_vmap_generated_nodes4():
#     # both a and b captured as a closure
#     a = makerv(0)
#     b = makerv(1)
#     f = lambda: [normal(a, b)]
#     generated, out = generated_nodes(f)
#     assert len(generated) == 1
#     assert generated[0].op == ir.Normal()


def test_vmap_generated_nodes5():
    def fun(a, b):
        loc = a + b
        scale = 1
        return [normal(loc, scale)]

    a = makerv(0)
    b = makerv(1)

    # both a and b given
    f = lambda a, b: fun(a, b)
    nodes = list(generated_nodes(f, a, b)[0])

    assert len(nodes) == 2
    assert nodes[0].op == ir.Add()
    assert nodes[1].op == ir.Normal()

    # b captured with closure
    f = lambda a: fun(a, b)
    nodes = list(generated_nodes(f, a)[0])
    assert len(nodes) == 2
    assert nodes[0].op == add
    assert nodes[2].op == ir.Normal()

    # # neither a nor b captured
    # f = lambda: fun(a, b)
    # nodes = list(vmap_generated_nodes(f)[0])
    # assert len(nodes) == 3
    # assert nodes[0].cond_dist == add
    # assert isinstance(nodes[1].cond_dist, Constant)
    # assert nodes[2].cond_dist == normal_scale

