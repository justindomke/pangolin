import jax.random

from pangolin.interface import *
from pangolin.inference.numpyro.model import (
    get_model_flat,
)
import numpyro.handlers
import numpy as np


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
