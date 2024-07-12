import jax.random

from cleanpangolin import ir
from cleanpangolin.interface import *
from cleanpangolin.inference.numpyro import (
    numpyro_var,
    get_model_flat,
    simple_handlers,
    numpyro_handlers,
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
    dims = np.random.choice([2,5,10]) # for matrices / vectors
    out = []
    for domain in ranges:
        if domain == "real":
            new = np.random.randn()
        elif domain == "positive":
            new = np.abs(np.random.randn())
        elif domain == "vector":
            new = np.random.randn(dims)
        elif domain == "matrix":
            new = np.random.randn(dims,dims)
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
    (arccos, np.arccos, [(-.999,.999)]),
    (arcsin, np.arcsin, [(-.999,.999)]),
    (arctan, np.arctan, ["real"]),
    (arccosh, np.arccosh, [(1,100)]),
    (arcsinh, np.arcsinh, [(-100,100)]),
    (arctanh, np.arctanh, [(-.999,.999)]),
    (exp, np.exp, ["real"]),
    (inv_logit, scipy.special.expit, ["real"]),
    (expit, scipy.special.expit, ["real"]),
    (sigmoid, scipy.special.expit, ["real"]),
    (log, np.log, ["positive"]),
    (log_gamma, scipy.special.loggamma, ["positive"]),
    (logit, scipy.special.logit, [(0,1)]),
    (step, lambda a: np.heaviside(a,0.5), ["real"]),
    (matmul, np.dot, ["vector","vector"]),
    (matmul, np.dot, ["vector","matrix"]),
    (matmul, np.dot, ["matrix","vector"]),
    (matmul, np.dot, ["matrix","matrix"]),
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


