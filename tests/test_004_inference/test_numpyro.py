import jax.random

from pangolin import ir
from pangolin.interface import *
from pangolin.inference.numpyro import (
    numpyro_var,
    get_model_flat,
    simple_handlers,
    numpyro_handlers,
    sample_flat,
    sample,
    E,
    std,
    var,
    ancestor_sample_flat,
)
from numpyro import distributions as numpyro_dist
from numpyro import infer as numpyro_infer
import numpyro.handlers
import numpy as np
import inspect
from jax import numpy as jnp
import pytest
import scipy.special


def rands_from_ranges(ranges):
    dims = np.random.choice([2, 5, 10])  # for matrices / vectors
    out = []
    for domain in ranges:
        if domain == "real":
            new = np.random.randn()
        elif domain == "positive":
            new = np.abs(np.random.randn())
        elif domain == "vector":
            new = np.random.randn(dims)
        elif domain == "matrix":
            new = np.random.randn(dims, dims)
        elif isinstance(domain, tuple):
            lo, hi = domain
            new = lo + np.random.rand() * (hi - lo)
        else:
            raise NotImplementedError()
        out.append(new)
    return out


testdata = [
    (add, lambda a, b: a + b, ["real", "real"]),
    (sub, lambda a, b: a - b, ["real", "real"]),
    (mul, lambda a, b: a * b, ["real", "real"]),
    (div, lambda a, b: a / b, ["real", "real"]),
    (pow, lambda a, b: a**b, ["positive", "real"]),
    (sqrt, np.sqrt, ["positive"]),
    (abs, np.abs, ["real"]),
    (cos, np.cos, ["real"]),
    (sin, np.sin, ["real"]),
    (tan, np.tan, ["real"]),
    (arccos, np.arccos, [(-0.999, 0.999)]),
    (arcsin, np.arcsin, [(-0.999, 0.999)]),
    (arctan, np.arctan, ["real"]),
    (arccosh, np.arccosh, [(1, 100)]),
    (arcsinh, np.arcsinh, [(-100, 100)]),
    (arctanh, np.arctanh, [(-0.999, 0.999)]),
    (exp, np.exp, ["real"]),
    (inv_logit, scipy.special.expit, ["real"]),
    (expit, scipy.special.expit, ["real"]),
    (sigmoid, scipy.special.expit, ["real"]),
    (log, np.log, ["positive"]),
    (log_gamma, scipy.special.loggamma, ["positive"]),
    (logit, scipy.special.logit, [(0, 1)]),
    (step, lambda a: np.heaviside(a, 0.5), ["real"]),
    (matmul, np.dot, ["vector", "vector"]),
    (matmul, np.dot, ["vector", "matrix"]),
    (matmul, np.dot, ["matrix", "vector"]),
    (matmul, np.dot, ["matrix", "matrix"]),
    (inv, np.linalg.inv, ["matrix"]),
]


@pytest.mark.parametrize("pangolin_fun, numpy_fun, ranges", testdata)
def test_op(pangolin_fun, numpy_fun, ranges):
    for reps in range(5):
        inputs = rands_from_ranges(ranges)
        output_rv = pangolin_fun(*inputs)
        model, var_to_name = get_model_flat([output_rv], [], [])
        with numpyro.handlers.seed(rng_seed=0):
            samples = model()
        output_pangolin = samples[var_to_name[output_rv]]
        output_numpy = numpy_fun(*inputs)
        assert np.allclose(output_pangolin, output_numpy, atol=1e-5, rtol=1e-5)


# def test_scalar_fun(pangolin_fun, numpy_fun, range):
#    run_tests_on_scalar_op(pangolin_fun, numpy_fun, [range])


def test_all_op_types_have_handlers():
    excluded_op_types = [ir.Op, ir.VecMatOp]

    for name in dir(ir):
        op_type = getattr(ir, name)
        if inspect.isclass(op_type):
            if issubclass(op_type, ir.Op) and op_type not in excluded_op_types:
                if (op_type not in simple_handlers) and (op_type not in numpyro_handlers):
                    raise Warning(f"No handler for {op_type} {op_type().name=}")


def test_numpyro_var_add():
    assert numpyro_var(ir.Add(), 2, 3) == 5


def test_numpyro_var_normal():
    out = numpyro_var(ir.Normal(), 2, 3)
    assert out.loc == 2
    assert out.scale == 3
    assert isinstance(out, numpyro_dist.Normal)


def test_numpyro_var_normal_prec():
    out = numpyro_var(ir.NormalPrec(), 2, 2)
    assert out.loc == 2
    assert out.scale == 1 / 4  # exact because using powers of 2 yay floating point
    assert isinstance(out, numpyro_dist.Normal)


def test_get_model_flat_deterministic():
    x = makerv(1.5)
    y = x**3
    model, var_to_name = get_model_flat([x, y], [], [])
    with numpyro.handlers.seed(rng_seed=0):
        out = model()
    x_samps = out[var_to_name[x]]
    y_samps = out[var_to_name[y]]
    assert x_samps.shape == ()
    assert y_samps.shape == ()
    assert np.allclose(x_samps, 1.5)
    assert np.allclose(y_samps, 1.5**3)


def test_get_model_flat_single_normal():
    """
    Simplest test to show that you can get a "normal" numpyro model out of the model and do
    "normal" numpyro stuff with it.
    """

    x = normal(0, 1)
    model, var_to_name = get_model_flat([x], [], [])

    # The above model code is equivalent to:
    # def model():
    #     v0 = numpyro.sample('v0', numpyro_dist.Normal(0,1))
    #     return {'v0':v0}
    # var_to_name = {x:'v0'}

    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.plate("multiple_samples", 10000):
            out = model()

    x_samps = out[var_to_name[x]]
    assert x_samps.shape == (10000,)
    assert np.abs(np.mean(x_samps)) < 0.05
    assert np.abs(np.var(x_samps) - 1) < 0.05


def test_get_model_flat_pair_normals():
    """
    Simplest test to show that you can get a "normal" numpyro model out of the model and do
    "normal" numpyro stuff with it.
    """

    x = normal(0, 1)
    y = normal(x, 1)
    model, var_to_name = get_model_flat([x, y], [], [])

    # The above model code is equivalent to:
    # def model():
    #     v0 = numpyro.sample('v0', numpyro_dist.Normal(0,1))
    #     v1 = numpyro.sample('v1', numpyro_dist.Normal(0,1))
    #     return {'v0':v0,'v1':v1}
    # var_to_name = {x:'v0',y:'v1'}

    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.plate("multiple_samples", 10000):
            out = model()

    x_samps = out[var_to_name[x]]
    y_samps = out[var_to_name[y]]
    assert x_samps.shape == (10000,)
    assert np.abs(np.mean(x_samps)) < 0.05
    assert np.abs(np.var(x_samps) - 1) < 0.05
    assert np.abs(np.mean(y_samps)) < 0.05
    assert np.abs(np.var(y_samps) - 2) < 0.05
    assert np.abs(np.mean((y_samps - x_samps) ** 2) - 1) < 0.05


def test_get_model_flat_vmap_deterministic():
    x = makerv(1.5)
    y = vmap(lambda x: x**3, None, 5)(x)
    model, var_to_name = get_model_flat([x, y], [], [])
    with numpyro.handlers.seed(rng_seed=0):
        out = model()
    x_samps = out[var_to_name[x]]
    y_samps = out[var_to_name[y]]
    assert x_samps.shape == ()
    assert y_samps.shape == (5,)
    assert np.allclose(x_samps, 1.5)
    assert np.allclose(y_samps, 1.5**3)


def test_get_model_flat_pair_normals_mcmc():
    """
    Simplest test to show that you can get a "normal" numpyro model out of the model and do
    "normal" numpyro stuff with it.
    """

    x = normal(0, 1)
    y = normal(x, 1)
    model, var_to_name = get_model_flat([y], [x], [3.5])

    # The above model code is equivalent to:
    # def model():
    #     v0 = numpyro.sample('v0', numpyro_dist.Normal(0,1))
    #     v1 = numpyro.sample('v1', numpyro_dist.Normal(0,1))
    #     return {'v0':v0,'v1':v1}
    # var_to_name = {x:'v0',y:'v1'}

    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.plate("multiple_samples", 10000):
            out = model()

    y_samps = out[var_to_name[y]]
    assert np.abs(np.mean(y_samps) - 3.5) < 0.05
    assert np.abs(np.var(y_samps) - 1) < 0.05


def test_get_model_flat_pair_normals_conditioned_at_bottom():
    """
    Simplest test to show that you can get a "normal" numpyro model out of the model and do
    "normal" numpyro stuff with it.
    """

    x = normal(0, 1)
    y = normal(x, 1)
    y_obs = 3.5
    nsamps = 1000
    model, var_to_name = get_model_flat([x], [y], [y_obs])

    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=nsamps // 2, num_samples=nsamps)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    mcmc.print_summary()
    out = mcmc.get_samples()

    x_samps = out[var_to_name[x]]
    assert x_samps.shape == (nsamps,)
    assert np.abs(np.mean(x_samps) - y_obs / 2) < 0.05
    assert np.abs(np.var(x_samps) - 0.5) < 0.05


def inf_until_match(inf, vars, given, vals, testfun, niter_start=1000, niter_max=100000):
    from time import time

    niter = niter_start
    while niter <= niter_max:
        t0 = time()
        out = inf(vars, given, vals, niter=niter)
        t1 = time()
        print(f"{niter=} {t1 - t0}")
        if testfun(out):
            assert True
            return
        else:
            niter *= 2
    assert False


import functools

sample_until_match = functools.partial(inf_until_match, sample)


def sample_flat_until_match(vars, given, vals, testfun, niter_start=1000, niter_max=100000):
    new_testfun = lambda stuff: testfun(stuff[0])
    return inf_until_match(sample_flat, vars, given, vals, new_testfun, niter_start, niter_max)


def test_sample_flat():
    x = normal(0, 1)

    def testfun(xs):
        return np.abs(np.mean(xs)) < 0.05 and np.abs(np.std(xs) - 1) < 0.05

    sample_flat_until_match([x], [], [], testfun)


def test_sample():
    x = normal(0, 1)

    def testfun(xs):
        return np.abs(np.mean(xs)) < 0.05 and np.abs(np.std(xs) - 1) < 0.05

    sample_until_match(x, None, None, testfun)


def test_discrete():
    x = bernoulli(0.5)
    y = bernoulli(x * 0.9 + (1 - x) * 0.1)

    # p(x=1|y=1) \propto 0.5 * .9 = 0.45
    # p(x=0|y=1) \propto 0.5 * .1 = 0.05
    # p(x=0|y=1) = .9
    # p(x=1|y=1) = .1

    def testfun(xs):
        return np.abs(np.mean(xs) < 0.9) < 0.01

    sample_flat_until_match([x], [y], [1], testfun)
    sample_until_match(x, y, 1, testfun)


def test_composite():
    x = makerv(1.5)

    @composite
    def f(x):
        a = x + 2
        b = x * x
        return normal(a**b, 1e-5)

    y = f(x)
    assert isinstance(y.op, ir.Composite)

    expected = (1.5 + 2) ** (1.5**2)

    [ys] = sample_flat([y], [], [], niter=100)
    assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)

    ys = sample(y, None, None, niter=100)
    assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)


def test_composite_simple_const_rv():
    x = makerv(0.5)
    noise = makerv(1e-3)

    @composite
    def f(last):
        return normal(last, noise)  # +1

    y = f(x)

    print_upstream(y)

    def testfun(E_y):
        print(f"{E_y=}")
        return np.abs(E_y - 0.5) < 0.1

    inf_until_match(E, y, [], [], testfun)


def test_autoregressive_simple():
    x = makerv(0.5)
    length = 12
    y = autoregressive(lambda last: normal(last + 1, 1e-4), length)(x)

    print_upstream(y)

    assert isinstance(y.op, ir.Autoregressive)

    def testfun(ys):
        last_y = ys[-1, :]
        expected = 0.5 + np.arange(1, length + 1)
        print(f"{last_y=}")
        print(f"{expected=}")
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
    print(f"{ys=}")

def test_autoregressive_const_rv_unmapped():
    x = makerv(0.5)
    length = 12
    noise = makerv(1e-4)
    op = ir.Autoregressive(ir.Normal(), length, (None,), 0)
    y = ir.RV(op, x, noise)

    ys = sample(y)
    print(f"{ys=}")


def test_autoregressive_simple_const_rv():
    x = makerv(0.5)
    length = 12
    noise = makerv(1e-4)
    y = autoregressive(lambda last: normal(last + 1, noise), length)(x)

    assert isinstance(y.op, ir.Autoregressive)
    base_op = y.op.base_op
    assert isinstance(base_op, ir.Composite)
    assert base_op == ir.Composite(2, [ir.Constant(1), ir.Add(), ir.Normal()], [[], [0, 2], [3, 1]])
    assert y.op == ir.Autoregressive(base_op, length, [None], 0)

    print_upstream(y)

    def testfun(ys):
        last_y = ys[-1, :]
        expected = 0.5 + np.arange(1, length + 1)
        print(f"{last_y=}")
        print(f"{expected=}")
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
        print(f"{last_y=}")
        print(f"{expected=}")
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
        print(f"{last_y=}")
        print(f"{expected=}")
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
    print_upstream(y)

    def testfun(ys):
        last_y = ys[-1, :]
        assert last_y.shape == (length, ndim)
        out = last_y[-1, :]
        expected = x0
        for i in range(length):
            expected = A @ expected
        print(f"{out=}")
        print(f"{expected=}")
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
    print_upstream(y)

    def testfun(ys):
        last_y = ys[-1, :]
        assert last_y.shape == (length, ndim)
        out = last_y[-1, :]
        expected = x0
        for i in range(length):
            expected = A @ expected
        print(f"{out=}")
        print(f"{expected=}")
        return np.max(np.abs(out - expected)) < 0.1

    sample_flat_until_match([y], [], [], testfun)
    sample_until_match(y, None, None, testfun)


def test_pytrees():
    "Can we sample if we store RVs in dicts or whatever"
    niter = 57
    d = {}
    d["hello"] = []
    d["hello"].append(normal(0, 1))
    d["hello"].append(normal(d["hello"][0], 2))
    samples = sample(d, niter=niter)
    assert isinstance(samples, dict)
    assert set(samples.keys()) == {"hello"}
    assert isinstance(samples["hello"], list)
    assert len(samples["hello"]) == 2
    assert samples["hello"][0].shape == (niter,)
    assert samples["hello"][1].shape == (niter,)


def test_E():
    x = normal(2.5, 3)

    def testfun(Ex):
        return np.abs(Ex - 2.5) < 0.1

    inf_until_match(E, x, None, None, testfun)


def test_std():
    x = normal(2.5, 3)

    def testfun(std_x):
        print(f"{std_x=}")
        return np.abs(std_x - 3) < 0.1

    inf_until_match(std, x, None, None, testfun)


def test_E_vector():
    loc = np.array([0, 1, 2])
    scale = np.array([2, 3, 4])

    x = vmap(normal)(loc, scale)

    def testfun(Ex):
        print(f"{Ex=}")
        return np.linalg.norm(Ex - loc) < 0.1

    inf_until_match(E, x, None, None, testfun)


def test_E_matrix():
    loc = np.array([[0, 1, 2], [3, 4, 5]])
    scale = np.array([2, 3, 4])

    x = vmap(vmap(normal), [0, None])(loc, scale)

    def testfun(Ex):
        print(f"{Ex=}")
        return np.linalg.norm(Ex - loc) < 0.1

    inf_until_match(E, x, None, None, testfun)


def test_E_pytree():
    d = {}
    d["x"] = normal(1.5, 2)
    d["y"] = normal(3.3, 4)

    def testfun(E_d):
        print(f"{E_d=}")
        return np.abs(E_d["x"] - 1.5) + np.abs(E_d["y"] - 3.3) < 0.1

    inf_until_match(E, d, None, None, testfun)


def test_std_pytree():
    d = {}
    d["x"] = normal(1.5, 2)
    d["y"] = normal(3.3, 4)

    def testfun(std_d):
        print(f"{std_d=}")
        return np.abs(std_d["x"] - 2) + np.abs(std_d["y"] - 4) < 0.1

    inf_until_match(std, d, None, None, testfun)


def test_std_vector():
    loc = np.array([0, 1, 2])
    scale = np.array([2, 3, 4])

    x = vmap(normal)(loc, scale)

    def testfun(Ex):
        print(f"{Ex=}")
        return np.linalg.norm(Ex - scale) < 0.1

    inf_until_match(std, x, None, None, testfun)


def test_random_index():
    a = bernoulli(0.5)
    b = makerv([0.1, 0.2])
    c = b[a]

    def testfun(Ec):
        print(f"{Ec=}")
        return np.abs(Ec - 0.15) < 0.05

    cs = sample(c)

    # inf_until_match(E, c, None, None, testfun)

def test_ancestor_sample_flat():
    x = normal(0,1)
    [xs] = ancestor_sample_flat([x],niter=1000)
