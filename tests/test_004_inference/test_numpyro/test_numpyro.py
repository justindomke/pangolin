import jax.random

from pangolin import ir
from pangolin.interface import *
from pangolin.inference.numpyro.handlers import numpyro_handlers
from pangolin.inference.numpyro import (
    # numpyro_var,
    # get_model_flat,
    # simple_handlers,
    # numpyro_handlers,
    #sample_flat,
    sample,
    E,
    std,
    var,
    #ancestor_sample_flat,
    # numpyro_vmap_var_random,
    # vmap_numpyro_pars,
)
from pangolin.inference.numpyro.sampling import ancestor_sample_flat, sample_flat
from numpyro import distributions as numpyro_dist
from numpyro import infer as numpyro_infer
import numpyro.handlers
import numpy as np
import inspect
from jax import numpy as jnp
import pytest
import scipy.special
import pangolin as pg
from util import inf_until_match, sample_until_match, sample_flat_until_match


def test_all_op_types_have_handlers():
    excluded_op_types = [ir.Op, ir.VecMatOp, ir.VMap]

    for name in dir(ir):
        op_type = getattr(ir, name)
        if inspect.isclass(op_type):
            if issubclass(op_type, ir.Op) and op_type not in excluded_op_types:
                if (op_type not in numpyro_handlers):
                    #raise Warning(f"No handler for {op_type} {op_type().name=}")
                    raise Warning(f"No handler for {op_type}")




def test_sample_flat():
    x = normal(0, 1)

    def testfun(xs):
        print(f"{np.mean(xs)=} {np.var(xs)=}")
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

