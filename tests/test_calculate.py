import pangolin.interface as interface
import numpy as np
from numpyro import distributions as dist

# import pangolin.calculate as calculate
import jax.random

from pangolin import calculate, inference_numpyro

def test_ancestor_sampling():
    niter = 53
    calc = calculate.Calculate("numpyro", niter=niter)

    x = interface.normal(0,1)
    xs = calc.sample(x,mode='ancestor')
    assert xs.shape == (niter,)

def test_given_ancestor_sampling():
    niter = 53
    calc = calculate.Calculate("numpyro", niter=niter)

    x = interface.normal(0,1)
    y = interface.normal(x,1)
    ys = calc.sample(y,x,0.0,mode='ancestor')
    assert ys.shape == (niter,)

def test_invalid_ancestor_sampling():
    niter = 53
    calc = calculate.Calculate("numpyro", niter=niter)

    x = interface.normal(0,1)
    y = interface.normal(x,1)
    xs = calc.sample(x,y,1.0,mode='ancestor')
    assert xs.shape == (niter,)


def test_numpyro():
    niter = 53

    # inf = inference_numpyro.NumpyroInference(niter=53)
    # calc = calculate.Calculate(inf)
    calc = calculate.Calculate("numpyro", niter=53)

    x = interface.normal(0, 1)
    xs = calc.sample(x,mode='mcmc')
    assert xs.shape == (niter,)

    xs = calc.sample(x, mode='ancestor')
    assert xs.shape == (niter,)

    # Ex = calc.E(x)
    #
    # assert Ex.shape == ()
