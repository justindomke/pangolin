from pangolin.simple_interface import (
    makerv,
    normal,
    autoregressive,
    ir,
    vmap,
    bernoulli,
)
from pangolin.inference.numpyro.test_util import (
    inf_until_match,
    sample_until_match,
    sample_flat_until_match,
)
import numpy as np
from pangolin.inference.numpyro.handlers import get_numpyro_val
from pangolin.inference.numpyro import sample, E


def test_autoregressive_val_nonrandom():
    def f(last):
        return last * 1.3

    x = makerv(1.1)
    length = 5
    y = autoregressive(f, length)(x)
    out = get_numpyro_val(y.op, 1.5, is_observed=False)

    last = 1.5
    expected = []
    for i in range(length):
        last = f(last)
        expected.append(last)
    expected = np.array(expected)

    assert np.allclose(out, expected)


def test_autoregressive_simple():
    x = makerv(0.5)
    length = 12
    y = autoregressive(lambda last: normal(last + 1, 1e-4), length)(x)

    assert isinstance(y.op, ir.Autoregressive)

    def testfun(ys):
        last_y = ys[-1, :]
        expected = 0.5 + np.arange(1, length + 1)
        return np.max(np.abs(last_y - expected)) < 0.1

    sample_flat_until_match([y], [], [], testfun)
    sample_until_match(y, None, None, testfun)


def test_autoregressive_const_rv_mapped():
    x = makerv(0.5)
    length = 12
    noises = makerv(np.random.rand(length))
    op = ir.Autoregressive(ir.Normal(), length, (0,), 0)
    y = ir.RV(op, x, noises)
    ys = sample(y)
    # print(f"{ys=}")


def test_autoregressive_const_rv_unmapped():
    x = makerv(0.5)
    length = 12
    noise = makerv(1e-4)
    op = ir.Autoregressive(ir.Normal(), length, (None,), 0)
    y = ir.RV(op, x, noise)

    ys = sample(y)
    # print(f"{ys=}")


def test_autoregressive_simple_const_rv():
    x = makerv(0.5)
    length = 12
    noise = makerv(1e-4)
    y = autoregressive(lambda last: normal(last + 1, noise), length)(x)

    assert isinstance(y.op, ir.Autoregressive)
    base_op = y.op.base_op
    assert isinstance(base_op, ir.Composite)
    assert base_op == ir.Composite(
        2, [ir.Constant(1), ir.Add(), ir.Normal()], [[], [0, 2], [3, 1]]
    )
    assert y.op == ir.Autoregressive(base_op, length, [None], 0)

    def testfun(ys):
        last_y = ys[-1, :]
        expected = 0.5 + np.arange(1, length + 1)
        return np.max(np.abs(last_y - expected)) < 0.1

    sample_flat_until_match([y], [], [], testfun)
    sample_until_match(y, None, None, testfun)


def test_autoregressive_nonrandom():
    x = normal(0.0, 1e-5)
    length = 12
    y = autoregressive(lambda last: last + 1, length)(x)

    assert isinstance(y.op, ir.Autoregressive)

    def testfun(ys):
        last_y = ys[-1, :]
        expected = np.arange(1, length + 1)
        return np.max(np.abs(last_y - expected)) < 0.1

    sample_flat_until_match([y], [], [], testfun)
    sample_until_match(y, None, None, testfun)


def test_autoregressive_varying_increments():
    x = makerv(0.0)
    length = 12
    increment = np.random.randn(length)
    y = autoregressive(lambda last, inc: normal(last + inc, 1e-4), length)(x, increment)

    assert isinstance(y.op, ir.Autoregressive)

    def testfun(ys):
        last_y = ys[-1, :]
        expected = np.cumsum(increment)
        return np.max(np.abs(last_y - expected)) < 0.1

    sample_flat_until_match([y], [], [], testfun)
    sample_until_match(y, None, None, testfun)


def test_autoregressive_matmul():
    ndim = 5
    length = 3
    x0 = np.random.randn(ndim)
    x = vmap(normal, [0, None], ndim)(x0, 1e-5)
    A = np.random.randn(ndim, ndim)
    y = autoregressive(lambda last: A @ last, length)(x)

    def testfun(ys):
        last_y = ys[-1, :]
        assert last_y.shape == (length, ndim)
        out = last_y[-1, :]
        expected = x0
        for i in range(length):
            expected = A @ expected
        return np.max(np.abs(out - expected)) < 0.1

    sample_flat_until_match([y], [], [], testfun)
    sample_until_match(y, None, None, testfun)


def test_autoregressive_matmul_A_rv():
    ndim = 5
    length = 3
    x0 = np.random.randn(ndim)
    x = vmap(normal, [0, None], ndim)(x0, 1e-5)
    A = np.random.randn(ndim, ndim)
    A_rv = makerv(A)
    y = autoregressive(lambda last: A_rv @ last, length)(x)

    def testfun(ys):
        last_y = ys[-1, :]
        assert last_y.shape == (length, ndim)
        out = last_y[-1, :]
        expected = x0
        for i in range(length):
            expected = A @ expected
        return np.max(np.abs(out - expected)) < 0.1

    sample_flat_until_match([y], [], [], testfun)
    sample_until_match(y, None, None, testfun)


def long_bernoulli_chain_expected(p, length, x_obs):
    m = [np.array([1 - p, p])]
    for i in range(length):
        # p0 = .95 * m[-1][0] + .05 * m[-1][1]
        # p1 = .05 * m[-1][0] + .95 * m[-1][1]
        # m.append(np.array([p0, p1]))
        q = 0.05 * m[-1][0] + 0.95 * m[-1][1]
        m.append(np.array([1 - q, q]))
    m = np.array(m)

    if x_obs is None:
        n = 1
    else:

        if x_obs == 0:
            q = 0.05
        else:
            q = 0.95
        n = [np.array([1 - q, q])]
        for i in range(length):
            q = 0.05 * n[-1][0] + 0.95 * n[-1][1]
            n.append(np.array([1 - q, q]))
        n = np.array(list(reversed(n)))

    probs = m * n
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs[:, 1]


# def test_discrete_autoregressive_ancestor():
#     p = np.random.rand()
#     length = 5
#     #x_obs = np.random.randint(2)
#
#     expected_Ez = long_bernoulli_chain_expected(p, length, None)
#
#     z0 = bernoulli(p)
#     z = autoregressive(lambda last: bernoulli(0.05 + 0.9 * last), length)(z0)
#     x = bernoulli(0.05 + 0.9 * z[-1])
#
#     def testfun(expectations):
#         Ez0, Ez = expectations
#         Ez = np.append(Ez0, Ez)
#         print(f"         {Ez=}")
#         print(f"{expected_Ez=}")
#         return np.all(np.abs(Ez - expected_Ez) < .01)
#
#     inf_until_match(E, (z0,z), [], [], testfun)


# def test_latent_discrete_autoregressive_forbidden():
#     p = np.random.rand()
#     length = 5
#     x_obs = np.random.randint(2)

#     z0 = bernoulli(p)
#     z = autoregressive(lambda last: bernoulli(0.05 + 0.9 * last), length)(z0)
#     x = bernoulli(0.05 + 0.9 * z[-1])

#     try:
#         E(z, x, x_obs)
#         assert False
#     except ValueError as e:
#         assert str(e).startswith(
#             "Can't have non-observed autoregressive over discrete variables"
#         )


# def test_double_autoregressive():
#     # def inner_f(last, input):
#     #     return normal(last+input,1e-7)
#     #
#     # def outer_f(last_chain):
#     #     return autoregressive(inner_f, 10)(jnp.zeros(10), last_chain)
#
#     # define random walk over scalars with vector input
#     randwalk = autoregressive(lambda last, input: normal(last+input,1e-7), 10)
#     z = randwalk(0.0, np.arange(10))
#     assert z.shape == (10,)
#     zs = sample(z,niter=1)
#     assert zs.shape == (1,10)
#
#     vecwalk = autoregressive(lambda last: randwalk(0.0, last), 5)
#     u = vecwalk(np.zeros(10))
#     assert u.shape == (5, 10)
#     us = sample(u,niter=1)
#     assert us.shape == (1, 5, 10)


def test_double_autoregressive():

    # define random walk over scalars with vector input
    randwalk = autoregressive(lambda last, input: normal(last + input, 1e-7), 10)
    z = randwalk(0.0, np.arange(10))
    assert z.shape == (10,)
    zs = sample(z, niter=1)
    assert zs.shape == (1, 10)

    vecwalk = autoregressive(lambda last: randwalk(0.0, last), 5)
    u = vecwalk(np.zeros(10))
    assert u.shape == (5, 10)
    us = sample(u, niter=1)
    assert us.shape == (1, 5, 10)

    # z = autoregressive(outer_f, 5)(0.0)
