from pangolin.interface import *
from pangolin.interface.autoregressive import autoregressive, autoregressive_flat, repeat
import numpy as np
from pangolin import ir

def test_simple_flat():
    def fun(x):
        return exponential(x)

    u = makerv(2.2)
    y = autoregressive_flat(fun, 10)(u)
    base_op = ir.Composite(1, [ir.Exponential()], [[0]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [])
    assert y.parents == (u,)


def test_normal():
    def fun(x):
        return normal(x, 1.1)

    u = makerv(2.2)
    y = autoregressive_flat(fun, 10)(u)
    base_op = ir.Composite(1, [ir.Constant(1.1), ir.Normal()], [[], [0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [])


def test_normal_external_scale():
    scale = makerv(1.1)

    def fun(x):
        return normal(x, scale)

    u = makerv(2.2)
    y = autoregressive_flat(fun, 10)(u)
    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [None])
    assert y.parents == (u, scale)


def test_normal_with_param():
    def fun(x, scale):
        return normal(x, scale)

    u = makerv(2.2)
    scales = makerv(np.arange(1, 11))
    y = autoregressive_flat(fun, 10)(u, scales)
    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [0])
    assert y.parents == (u, scales)


def test_normal_with_param_no_length():
    def fun(x, scale):
        return normal(x, scale)

    u = makerv(2.2)
    scales = makerv(np.arange(1, 11))
    y = autoregressive_flat(fun)(u, scales)
    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [0])
    assert y.parents == (u, scales)


def test_pass_constants():
    def fun(x, scale):
        return normal(x, scale)

    y = autoregressive_flat(fun)(2.2, np.arange(1, 11))
    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [0])
    assert y.parents[0].op == ir.Constant(2.2)
    assert y.parents[1].op == ir.Constant(np.arange(1, 11))


def test_simple_non_flat():
    def fun(x):
        return exponential(x)

    u = makerv(2.2)
    y = autoregressive(fun, 10)(u)
    base_op = ir.Composite(1, [ir.Exponential()], [[0]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [])
    assert y.parents == (u,)


def test_normal_non_flat():
    def fun(x):
        return normal(x, 1.1)

    u = makerv(2.2)
    y = autoregressive(fun, 10)(u)
    base_op = ir.Composite(1, [ir.Constant(1.1), ir.Normal()], [[], [0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [])


def test_normal_with_param_non_flat():
    def fun(x, scale):
        return normal(x, scale)

    u = makerv(2.2)
    scales = makerv(np.arange(1, 11))
    y = autoregressive(fun, 10)(u, scales)

    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [0])
    assert y.parents == (u, scales)


def test_normal_with_param_no_length_non_flat():
    def fun(x, scale):
        return normal(x, scale)

    u = makerv(2.2)
    scales = makerv(np.arange(1, 11))
    y = autoregressive(fun, None)(u, scales)
    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [0])
    assert y.parents == (u, scales)


def test_pass_constants_non_flat():
    def fun(x, scale):
        return normal(x, scale)

    y = autoregressive(fun)(2.2, makerv(np.arange(1, 11)))
    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [0])
    assert y.parents[0].op == ir.Constant(2.2)
    assert y.parents[1].op == ir.Constant(np.arange(1, 11))


def test_normal_with_unmapped_param_non_flat():
    def fun(x, scale):
        return normal(x, scale)

    u = makerv(2.2)
    scale = makerv(3.3)
    y = autoregressive(fun, 10, None)(u, scale)

    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [None])
    assert y.parents == (u, scale)


def test_dict_non_flat():
    def fun(x, params):
        return normal(x, params["scale"])

    u = makerv(2.2)
    params = {"scale": makerv(3.3)}
    y = autoregressive(fun, 10, None)(u, params)

    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [None])
    assert y.parents == (u, params["scale"])


def test_interesting_autoregressive():
    def fun(last, x, y, stuff):
        a = x[0] + stuff["hi"]
        b = x[1] * stuff["yo"]
        c = y[0][0] / y[0][1]
        d = a / b
        e = c**d
        f = last - e
        return f

    init = 0.5
    x = [1, 2]
    y = [[3, 4], 5]
    stuff = {"hi": 6, "yo": 7}
    y = autoregressive(fun, length=10, in_axes=None)(init, x, y, stuff)
    base_op = ir.Composite(
        8,
        [ir.Add(), ir.Mul(), ir.Div(), ir.Div(), ir.Pow(), ir.Sub()],
        [[1, 6], [2, 7], [3, 4], [8, 9], [10, 11], [0, 12]],
    )
    assert y.op.base_op == base_op
    op = ir.Autoregressive(base_op, 10, [None] * 7, 0)
    assert y.op == op


def test_interesting_closure_autoregressive():
    x = [makerv(1), makerv(2)]
    y = [[makerv(3), makerv(4)], makerv(5)]
    stuff = {"hi": makerv(6), "yo": makerv(7)}

    def fun(last):
        a = x[0] + stuff["hi"]
        b = x[1] * stuff["yo"]
        c = y[0][0] / y[0][1]
        d = a / b
        e = c**d
        f = last - e
        return f

    init = 0.5
    y = autoregressive(fun, length=10)(init)
    base_op = ir.Composite(
        7,  # now, y[1][0] is invisible, never included
        [ir.Add(), ir.Mul(), ir.Div(), ir.Div(), ir.Pow(), ir.Sub()],
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [0, 11]],  # now included as encountered
    )
    assert y.op.base_op == base_op
    op = ir.Autoregressive(base_op, 10, [None] * 6, 0)  # one less input
    assert y.op == op


def test_decorator():
    @repeat(length=10)
    def fun(last):
        return normal(last, 2.2)

    x = fun(0)
    base_op = ir.Composite(1, [ir.Constant(2.2), ir.Normal()], [[], [0, 1]])
    assert x.op.base_op == base_op
    op = ir.Autoregressive(base_op, 10, [], 0)
    assert x.op == op


def test_snappy_syntax():
    y = repeat(10)(lambda last: normal(last, 7.5))(0)
    base_op = ir.Composite(1, [ir.Constant(7.5), ir.Normal()], [[], [0, 1]])
    assert y.op.base_op == base_op
    op = ir.Autoregressive(base_op, 10, [], 0)
    assert y.op == op


def test_vmap_inside_autoregressive():
    @repeat(10)
    def f(last):
        return vmap(normal, (0, None))(last, 7.5)

    y = f(np.zeros(5))
    base_op = ir.Composite(
        1, [ir.Constant(7.5), ir.VMap(ir.Normal(), (0, None), 5)], [[], [0, 1]]
    )
    assert y.op.base_op == base_op
    op = ir.Autoregressive(base_op, 10, [], 0)
    assert y.op == op

def test_autoregressive_inside_vmap():
    @repeat(10)
    def f(last):
        return normal(last, 7.5)

    y = vmap(f)(np.zeros(5))
    print(f"{y.op=}")
    base_op = ir.Composite(1, [ir.Constant(7.5), ir.Normal()], [[], [0, 1]])
    assert y.op.base_op.base_op == base_op
    auto_op = ir.Autoregressive(base_op, 10, [], 0)
    assert y.op.base_op == auto_op
    op = ir.VMap(auto_op, (0,), 5)
    assert y.op == op
