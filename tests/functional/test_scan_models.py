from pangolin import interface as pi, ir


from pangolin.interface import *
from pangolin.functional.scanning import scan
import numpy as np

# from pangolin import sample


def test_simple_flat():
    def fun(x):
        return exponential(x)

    u = constant(2.2)
    y = scan(fun, 10)(pi.default.coerce(u))
    base_op = ir.Exponential()
    assert y.op.base_op == base_op
    assert y.op == ir.Scan(base_op, 10, [])
    assert y.parents == (u,)


def test_normal_flat():
    def fun(x):
        return normal(x, 1.1)

    u = constant(2.2)
    y = scan(fun, 10)(pi.default.coerce(u))
    base_op = ir.Composite(1, (ir.Constant(1.1), ir.Normal()), [[], [0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Scan(base_op, 10, [])


def test_normal_external_scale_flat():
    scale = constant(1.1)

    def fun(x):
        return normal(x, scale)

    u = constant(2.2)
    y = scan(fun, 10)(pi.default.coerce(u))
    base_op = ir.Normal()
    assert y.op.base_op == base_op
    assert y.op == ir.Scan(base_op, 10, [None])
    assert y.parents == (u, scale)


def test_normal_with_param_flat():
    def fun(x, scale):
        return normal(x, scale)

    u = constant(2.2)
    scales = constant(np.arange(1, 11))
    y = scan(fun, 10, 0)(pi.default.coerce(u), scales)
    base_op = ir.Normal()
    assert y.op.base_op == base_op
    assert y.op == ir.Scan(base_op, 10, [0])
    assert y.parents == (u, scales)


def test_scan_flat_explicit_inputs_precede_closure_captures():
    captured = constant(10)

    def fun(carry, explicit):
        return carry + explicit - captured

    carry = constant(0)
    explicit = constant(np.arange(3))
    result = scan(fun, 3, 0)(pi.default.coerce(carry), explicit)

    base_op = ir.Composite(3, (ir.Add(), ir.Sub()), ((0, 1), (3, 2)))
    assert result.op == ir.Scan(base_op, 3, (0, None))
    assert result.parents == (carry, explicit, captured)


def test_pass_constants_flat():
    def fun(x, scale):
        return normal(x, scale)

    y = scan(fun, 10, 0)(pi.default.coerce(2.2), constant(np.arange(1, 11)))
    base_op = ir.Normal()
    assert y.op.base_op == base_op
    assert y.op == ir.Scan(base_op, 10, [0])
    assert y.parents[0].op == ir.Constant(2.2)
    assert y.parents[1].op == ir.Constant(np.arange(1, 11))


def test_simple_non_flat():
    def fun(x):
        return exponential(x)

    u = constant(2.2)
    y = scan(fun, 10)(pi.default.coerce(u))
    base_op = ir.Exponential()
    assert y.op.base_op == base_op
    assert y.op == ir.Scan(base_op, 10, [])
    assert y.parents == (u,)


def test_normal_non_flat():
    def fun(x):
        return normal(x, 1.1)

    u = constant(2.2)
    y = scan(fun, 10)(pi.default.coerce(u))
    base_op = ir.Composite(1, (ir.Constant(1.1), ir.Normal()), [[], [0, 1]])
    assert y.op.base_op == base_op
    assert y.op == ir.Scan(base_op, 10, [])


def test_normal_with_param_non_flat():
    def fun(x, scale):
        return normal(x, scale)

    u_np = 2.2
    u_rv = constant(u_np)

    scales_np = np.arange(1, 11)
    scales_rv = constant(scales_np)

    for u in [u_np, u_rv]:
        y = scan(fun, 10)(pi.default.coerce(u), scales_rv)

        base_op = ir.Normal()
        assert y.op.base_op == base_op
        assert y.op == ir.Scan(base_op, 10, [0])

        assert y.parents[0].op == u_rv.op == ir.Constant(u_np)
        assert y.parents[1] is scales_rv

        if u is u_rv:
            assert y.parents[0] is u_rv


def test_normal_with_unmapped_param_non_flat():
    def fun(x, scale):
        return normal(x, scale)

    u_np = 2.2
    u_rv = constant(u_np)

    scales_np = 3.3
    scales_rv = constant(scales_np)

    length = 10

    for u in [u_np, u_rv]:
        y = scan(fun, length, None)(pi.default.coerce(u), scales_rv)

        base_op = ir.Normal()
        assert y.op.base_op == base_op
        assert y.op == ir.Scan(base_op, 10, [None])

        assert y.parents[0].op == u_rv.op == ir.Constant(u_np)
        assert y.parents[1] is scales_rv

        if u is u_rv:
            assert y.parents[0] is u_rv


# def test_normal_with_unmapped_param_non_flat():
#     def fun(x, scale):
#         return normal(x, scale)

#     u = constant(2.2)
#     scale = constant(3.3)
#     y = scan(fun, 10, None)(u, scale)

#     base_op = ir.Composite(2, [ir.Normal()], [[0, 1]])
#     assert y.op.base_op == base_op
#     assert y.op == ir.Scan(base_op, 10, [None])
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
            y = scan(fun, 10, None)(pi.default.coerce(u), params)

            base_op = ir.Normal()
            assert y.op.base_op == base_op
            assert y.op == ir.Scan(base_op, 10, [None])
            # assert y.parents == (u, params["scale"])

            assert y.parents[0].op == u_rv.op == ir.Constant(2.2)
            assert y.parents[1].op == ir.Constant(3.3)


def test_interesting_scan():
    def fun(last, x, y, stuff):
        a = x[0] + stuff["hi"]
        b = x[1] * stuff["yo"]
        c = y[0][0] / y[0][1]
        d = a / b
        e = c**d
        f = last - e
        return f

    init = 0.5
    x = [constant(1), constant(2)]
    y = [[constant(3), constant(4)], constant(5)]
    stuff = {"hi": constant(6), "yo": constant(7)}
    y = scan(fun, length=10, in_axes=None)(pi.default.coerce(init), x, y, stuff)
    base_op = ir.Composite(
        7,
        (ir.Add(), ir.Mul(), ir.Div(), ir.Div(), ir.Pow(), ir.Sub()),
        [[1, 5], [2, 6], [3, 4], [7, 8], [9, 10], [0, 11]],
    )
    assert y.op.base_op == base_op
    op = ir.Scan(base_op, 10, [None] * 6)
    assert y.op == op


def test_interesting_closure_scan():
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
    y = scan(fun, length=10)(pi.default.coerce(init))
    base_op = ir.Composite(
        7,  # now, y[1][0] is invisible, never included
        (ir.Add(), ir.Mul(), ir.Div(), ir.Div(), ir.Pow(), ir.Sub()),
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
    op = ir.Scan(base_op, 10, [None] * 6)  # one less input
    assert y.op == op


def test_vmap_inside_scan():
    def f(last):
        return vmap(normal, (0, None))(last, constant(7.5))

    f = scan(f, 10)
    y = f(constant(np.zeros(5)))
    base_op = ir.Composite(1, (ir.Constant(7.5), ir.VMap(ir.Normal(), [0, None], 5)), [[], [0, 1]])
    assert y.op.base_op == base_op
    op = ir.Scan(base_op, 10, [])
    assert y.op == op


def test_scan_inside_vmap():
    def f(last):
        return normal(last, 7.5)

    f = scan(f, 10)
    y = vmap(f)(constant(np.zeros(5)))
    base_op = ir.Composite(1, (ir.Constant(7.5), ir.Normal()), [[], [0, 1]])
    assert y.op.base_op.base_op == base_op
    auto_op = ir.Scan(base_op, 10, [])
    assert y.op.base_op == auto_op
    op = ir.VMap(auto_op, (0,), 5)
    assert y.op == op


# def test_double_autoregressive_cumsum():
#     # use double autoregressive to define cumsum of cumsum

#     cumsum = scan(lambda last, input: last + input)
#     vecwalk = scan(lambda last: cumsum(0.0, last), 5)

#     a = np.random.randn(10)
#     u = vecwalk(a)
#     u_sample = sample(u,niter=None)

#     tmp = np.cumsum(a)
#     for i in range(5):
#         assert np.allclose(u_sample[i],tmp)
#         tmp = np.cumsum(tmp)


# def test_double_autoregressive_cumsum_random():
#     # use double autoregressive to define cumsum of cumsum, use random dists

#     cumsum = scan(lambda last, input: normal(last + input, 1e-7))
#     vecwalk = scan(lambda last: cumsum(0.0,last), 5)

#     a = np.random.randn(10)
#     u = vecwalk(a)
#     u_sample = sample(u,niter=None)

#     tmp = np.cumsum(a)
#     for i in range(5):
#         assert np.allclose(u_sample[i],tmp, atol=1e-3, rtol=1e-3)
#         tmp = np.cumsum(tmp)
