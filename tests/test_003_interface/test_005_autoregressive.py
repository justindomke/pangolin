from pangolin.interface import *
from pangolin.interface.autoregressing import (
    autoregressive,
    autoregressive_flat,
    autoregress,
)
import numpy as np
from pangolin import ir

# from pangolin import sample


def test_simple_flat():
    def fun(x):
        return exponential(x)

    u = constant(2.2)
    y = autoregressive_flat(fun, 10, ())(u)
    base_op = ir.Composite(1, [ir.Exponential()], [[0]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [])
    assert y.parents == (u,)


def test_normal():
    def fun(x):
        return normal(x, 1.1)

    u = constant(2.2)
    y = autoregressive_flat(fun, 10, ())(u)
    base_op = ir.Composite(1, [ir.Constant(1.1), ir.Normal()], [[], [0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [])


def test_normal_external_scale():
    scale = constant(1.1)

    def fun(x):
        return normal(x, scale)

    u = constant(2.2)
    y = autoregressive_flat(fun, 10, ())(u)
    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [None])
    assert y.parents == (u, scale)


def test_normal_with_param():
    def fun(x, scale):
        return normal(x, scale)

    u = constant(2.2)
    scales = constant(np.arange(1, 11))
    y = autoregressive_flat(fun, 10, (0,))(u, scales)
    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [0])
    assert y.parents == (u, scales)


def test_pass_constants():
    def fun(x, scale):
        return normal(x, scale)

    y = autoregressive_flat(fun, 10, (0,))(2.2, np.arange(1, 11))
    base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [0])
    assert y.parents[0].op == ir.Constant(2.2)
    assert y.parents[1].op == ir.Constant(np.arange(1, 11))


def test_simple_non_flat():
    def fun(x):
        return exponential(x)

    u = constant(2.2)
    y = autoregressive(fun, 10)(u)
    base_op = ir.Composite(1, [ir.Exponential()], [[0]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [])
    assert y.parents == (u,)


def test_normal_non_flat():
    def fun(x):
        return normal(x, 1.1)

    u = constant(2.2)
    y = autoregressive(fun, 10)(u)
    base_op = ir.Composite(1, [ir.Constant(1.1), ir.Normal()], [[], [0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Autoregressive(base_op, 10, [])


def test_normal_with_param_non_flat():
    def fun(x, scale):
        return normal(x, scale)

    u_np = 2.2
    u_rv = constant(u_np)

    scales_np = np.arange(1, 11)
    scales_rv = constant(scales_np)

    for u in [u_np, u_rv]:
        for length in [10, None]:
            for scales in [scales_rv, scales_np]:
                y = autoregressive(fun, length)(u, scales)

                base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
                assert y.op.base_op == base_op
                assert y.op == ir.Autoregressive(base_op, 10, [0])

                assert y.parents[0].op == u_rv.op == ir.Constant(u_np)
                assert y.parents[1].op == scales_rv.op == ir.Constant(scales_np)

                if u is u_rv:
                    assert y.parents[0] is u_rv
                if scales is scales_rv:
                    assert y.parents[1] is scales_rv


def test_normal_with_unmapped_param_non_flat():
    def fun(x, scale):
        return normal(x, scale)

    u_np = 2.2
    u_rv = constant(u_np)

    scales_np = 3.3
    scales_rv = constant(scales_np)

    length = 10

    for u in [u_np, u_rv]:
        for scales in [scales_rv, scales_np]:
            for in_axes in [None]:
                y = autoregressive(fun, length, in_axes)(u, scales)

                base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
                assert y.op.base_op == base_op
                assert y.op == ir.Autoregressive(base_op, 10, [None])

                assert y.parents[0].op == u_rv.op == ir.Constant(u_np)
                assert y.parents[1].op == scales_rv.op == ir.Constant(scales_np)

                if u is u_rv:
                    assert y.parents[0] is u_rv
                if scales is scales_rv:
                    assert y.parents[1] is scales_rv


# def test_normal_with_unmapped_param_non_flat():
#     def fun(x, scale):
#         return normal(x, scale)

#     u = constant(2.2)
#     scale = constant(3.3)
#     y = autoregressive(fun, 10, None)(u, scale)

#     base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
#     assert y.op.base_op == base_op
#     assert y.op == ir.Autoregressive(base_op, 10, [None])
#     assert y.parents == (u, scale)


def test_dict_non_flat():
    def fun(x, params):
        return normal(x, params["scale"])

    # u = constant(2.2)
    # params = {"scale": constant(3.3)}

    u_np = 2.2
    u_rv = constant(2.2)

    for u in [u_np, u_rv]:
        for params in [{"scale": constant(3.3)}]:
            y = autoregressive(fun, 10, None)(u, params)

            base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
            assert y.op.base_op == base_op
            assert y.op == ir.Autoregressive(base_op, 10, [None])
            # assert y.parents == (u, params["scale"])

            assert y.parents[0].op == u_rv.op == ir.Constant(2.2)
            assert y.parents[1].op == ir.Constant(3.3)


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
    x = [constant(1), constant(2)]
    y = [[constant(3), constant(4)], constant(5)]
    stuff = {"hi": constant(6), "yo": constant(7)}

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
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [0, 11],
        ],  # now included as encountered
    )
    assert y.op.base_op == base_op
    op = ir.Autoregressive(base_op, 10, [None] * 6, 0)  # one less input
    assert y.op == op


def test_decorator():
    @autoregress(length=10)
    def fun(last):
        return normal(last, 2.2)

    x = fun(0)
    assert isinstance(x.op, ir.Autoregressive)
    base_op = ir.Composite(1, [ir.Constant(2.2), ir.Normal()], [[], [0, 1]])
    assert x.op.base_op == base_op
    op = ir.Autoregressive(base_op, 10, [], 0)
    assert x.op == op


def test_snappy_syntax():
    y = autoregress(10)(lambda last: normal(last, 7.5))(0)
    base_op = ir.Composite(1, [ir.Constant(7.5), ir.Normal()], [[], [0, 1]])
    assert y.op.base_op == base_op
    op = ir.Autoregressive(base_op, 10, [], 0)
    assert y.op == op


def test_vmap_inside_autoregressive():
    @autoregress(10)
    def f(last):
        return vmap(normal, (0, None))(last, 7.5)

    y = f(np.zeros(5))
    base_op = ir.Composite(1, [ir.Constant(7.5), ir.VMap(ir.Normal(), (0, None), 5)], [[], [0, 1]])
    assert y.op.base_op == base_op
    op = ir.Autoregressive(base_op, 10, [], 0)
    assert y.op == op


def test_autoregressive_inside_vmap():
    @autoregress(10)
    def f(last):
        return normal(last, 7.5)

    y = vmap(f)(np.zeros(5))
    base_op = ir.Composite(1, [ir.Constant(7.5), ir.Normal()], [[], [0, 1]])
    assert y.op.base_op.base_op == base_op
    auto_op = ir.Autoregressive(base_op, 10, [], 0)
    assert y.op.base_op == auto_op
    op = ir.VMap(auto_op, (0,), 5)
    assert y.op == op


# def test_double_autoregressive_cumsum():
#     # use double autoregressive to define cumsum of cumsum

#     cumsum = autoregressive(lambda last, input: last + input)
#     vecwalk = autoregressive(lambda last: cumsum(0.0, last), 5)

#     a = np.random.randn(10)
#     u = vecwalk(a)
#     u_sample = sample(u,niter=None)

#     tmp = np.cumsum(a)
#     for i in range(5):
#         assert np.allclose(u_sample[i],tmp)
#         tmp = np.cumsum(tmp)


# def test_double_autoregressive_cumsum_random():
#     # use double autoregressive to define cumsum of cumsum, use random dists

#     cumsum = autoregressive(lambda last, input: normal(last + input, 1e-7))
#     vecwalk = autoregressive(lambda last: cumsum(0.0,last), 5)

#     a = np.random.randn(10)
#     u = vecwalk(a)
#     u_sample = sample(u,niter=None)

#     tmp = np.cumsum(a)
#     for i in range(5):
#         assert np.allclose(u_sample[i],tmp, atol=1e-3, rtol=1e-3)
#         tmp = np.cumsum(tmp)
