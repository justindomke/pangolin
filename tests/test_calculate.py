import pangolin.interface as interface
import numpy as np
from numpyro import distributions as dist

# import pangolin.calculate as calculate
import jax.random

from pangolin import calculate, inference_numpyro


def test_numpyro():
    niter = 53

    # inf = inference_numpyro.NumpyroInference(niter=53)
    # calc = calculate.Calculate(inf)
    calc = calculate.Calculate("numpyro", niter=53)

    x = interface.normal(0, 1)
    xs = calc.sample(x)

    assert xs.shape == (niter,)

    Ex = calc.E(x)

    assert Ex.shape == ()
