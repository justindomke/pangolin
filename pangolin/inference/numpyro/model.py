# from . import interface, dag, util, inference
import jax.tree_util
import functools
from jax import numpy as jnp
from numpyro import distributions as dist
from typing import Sequence
import numpyro
from jax import lax
from numpyro.distributions import util as dist_util
from jax.scipy import special as jspecial
from jax import nn as jnn
# import numpy as np
from pangolin.ir.rv import RV
from pangolin import dag, util, ir
from pangolin.interface.base import OperatorRV
from numpy.typing import ArrayLike
from pangolin.interface import RV_or_ArrayLike
from pangolin.inference import inference_util
import numpy as np

def get_model_flat(vars: list[RV], given: list[RV], vals: list[RV_or_ArrayLike]):
    """
    Given a "flat" specification of an inference problem, get a numpyro model.

    The basic algorithm here is quite simple:
    1. Get all variables upstream of `vars` and `given`
    2. Define a new model function that goes through those variables in order, and does the
    appropriate numpyro operations for each

    Parameters
    ----------
    vars
        A flat sequence of random variables to define a model over
    given
        A flat sequence of random variables to condition on
    vals
        A flat sequence of constants (should have `len(vals)==len(given)` and matching shapes for
        each element)

    Returns
    -------
    model
        a new numpyro model, which can be used just like any other numpyro model
    names
        the names of the random variables in the model corresponding to `vars`
    """
    from .handlers import get_numpyro_rv

    if not isinstance(vars, list):
        raise ValueError("vars must be list")
    if not isinstance(given, list):
        raise ValueError("given must be list")
    if not isinstance(vals, list):
        raise ValueError("vals must be list")
    if not all(isinstance(a, RV) for a in vars):
        raise ValueError("all elements of vars must be RVs")
    if not all(isinstance(a, RV) for a in given):
        raise ValueError("all elements of given must be RVs")
    if len(given) != len(vals):
        raise ValueError(f"length of given ({len(given)}) does not match length of vals ({len(vals)})")

    vals = [jnp.array(a) for a in vals]

    for var, val in zip(given, vals):
        if not util.is_numeric_numpy_array(val):
            raise ValueError("given val {val} not numeric")
        if var.shape != val.shape:
            raise ValueError("given var {var} with shape {var.shape} does not match corresponding given val {val} with shape {val.shape}")


    all_vars = dag.upstream_nodes(tuple(vars) + tuple(given))

    name_to_var = {}
    var_to_name = {}
    varnum = 0
    for var in all_vars:
        name = f"v{varnum}"
        varnum += 1
        name_to_var[name] = var
        var_to_name[var] = name

    def model():
        var_to_numpyro_rv = {}
        name_to_numpyro_rv = {}
        for var in all_vars:
            assert isinstance(var, RV)
            name = var_to_name[var]
            numpyro_pars = [var_to_numpyro_rv[p] for p in var.parents]

            if var in given:
                obs = vals[given.index(var)]
            else:
                obs = None

            numpyro_rv = get_numpyro_rv(var.op, name, obs, *numpyro_pars)

            var_to_numpyro_rv[var] = numpyro_rv
            name_to_numpyro_rv[name] = numpyro_rv
        return name_to_numpyro_rv

    return model, var_to_name

def is_continuous(op: ir.Op):
    continuous_dists = (
    ir.Normal, ir.Beta, ir.Cauchy, ir.Exponential, ir.Dirichlet, ir.Gamma, ir.LogNormal,
    ir.MultiNormal, ir.Poisson, ir.StudentT, ir.Uniform)
    discrete_dists = (
    ir.Bernoulli, ir.BernoulliLogit, ir.BetaBinomial, ir.Binomial, ir.Categorical, ir.Multinomial)

    if not op.random:
        raise ValueError("is_continuous only handles random ops")
    elif isinstance(op, ir.VMap):
        return is_continuous(op.base_op)
    elif isinstance(op, ir.Composite):
        return is_continuous(op.ops[-1])
    elif isinstance(op, ir.Autoregressive):
        return is_continuous(op.base_op)
    elif isinstance(op, continuous_dists):
        return True
    elif isinstance(op, discrete_dists):
        return False
    else:
        raise NotImplementedError(f"is_continuous doesn't not know to handle {op}")
