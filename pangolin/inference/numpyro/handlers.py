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

from inference.old_numpyro import is_continuous
# import numpy as np
from pangolin.ir.rv import RV
from pangolin import dag, util, ir
from pangolin.interface.interface import OperatorRV
from numpy.typing import ArrayLike
from pangolin.interface import RV_or_array
from pangolin.inference import inference_util
import numpy as np

from .handler_registry import numpyro_handlers, register_handler
from .vmap import get_numpyro_rv_discrete_latent

def get_numpyro_val(op: ir.Op, *numpyro_pars, is_observed):
    fun = numpyro_handlers[type(op)]
    return fun(op,*numpyro_pars, is_observed=is_observed)

def get_numpyro_rv(op: ir.Op, name: str, obs, *numpyro_pars):
    #handler = numpyro_handlers[type(op)]
    #return handler(op, name, obs, *numpyro_pars)
    is_observed = obs is not None

    # special case for RANDOM+DISCRETE+LATENT VMAP
    if isinstance(op, ir.VMap) and op.random and not is_continuous(op) and not is_observed:
       return get_numpyro_rv_discrete_latent(op, name, obs, *numpyro_pars)

    numpyro_val = get_numpyro_val(op, *numpyro_pars, is_observed = is_observed)

    if op.random:
        if obs is None:
            return numpyro.sample(name, numpyro_val)
        else:
            # TODO: Why doesn't this check always work?
            # assert obs.shape == numpyro_var.shape, f"{obs.shape} vs {numpyro_var.shape}"
            return numpyro.sample(name, numpyro_val, obs=obs)
    else:
        if obs is not None:
            raise ValueError("Can't have observation for non-random op {op}")
        return numpyro.deterministic(name, numpyro_val)

simple_dists = {
    ir.Normal: dist.Normal,
    ir.NormalPrec: lambda loc, prec: dist.Normal(loc, 1 / prec**2),
    ir.Bernoulli: dist.Bernoulli,
    ir.BernoulliLogit: dist.BernoulliLogits,
    ir.Beta: dist.Beta,
    ir.BetaBinomial: lambda n, a, b: dist.BetaBinomial(a, b, n),  # numpyro has a different order
    ir.Binomial: dist.Binomial,
    ir.Categorical: dist.Categorical,
    ir.Cauchy: dist.Cauchy,
    ir.Exponential: dist.Exponential,
    ir.Dirichlet: dist.Dirichlet,
    ir.Gamma: dist.Gamma,
    ir.LogNormal: dist.LogNormal,
    ir.Multinomial: dist.Multinomial,
    ir.MultiNormal: dist.MultivariateNormal,
    ir.Poisson: dist.Poisson,
    ir.StudentT: dist.StudentT,
    ir.Uniform: dist.Uniform,
}


def register_simple_dist_handler(op_class):
    d = simple_dists[op_class]

    @register_handler(op_class)
    def handle(op, *numpyro_pars, is_observed):
        # op is ignored, because simple!
        return d(*numpyro_pars)


for op_class in simple_dists:
    register_simple_dist_handler(op_class)

simple_funs = {
    ir.Add: lambda a, b: a + b,
    ir.Sub: lambda a, b: a - b,
    ir.Mul: lambda a, b: a * b,
    ir.Div: lambda a, b: a / b,
    ir.Pow: lambda a, b: a**b,
    ir.Abs: jnp.abs,
    ir.Arccos: jnp.arccos,
    ir.Arccosh: jnp.arccosh,
    ir.Arcsin: jnp.arcsin,
    ir.Arcsinh: jnp.arcsinh,
    ir.Arctan: jnp.arctan,
    ir.Arctanh: jnp.arctanh,
    ir.Cos: jnp.cos,
    ir.Cosh: jnp.cosh,
    ir.Exp: jnp.exp,
    ir.InvLogit: dist.transforms.SigmoidTransform(),
    ir.Log: jnp.log,
    ir.Loggamma: jspecial.gammaln,
    ir.Logit: jspecial.logit,
    ir.Sin: jnp.sin,
    ir.Sinh: jnp.sinh,
    ir.Step: lambda x: jnp.heaviside(x, 0.5),
    ir.Tan: jnp.tan,
    ir.Tanh: jnp.tanh,
    ir.MatMul: jnp.matmul,
    ir.Inv: jnp.linalg.inv,
    ir.Softmax: jnn.softmax,
}


def register_simple_fun_handler(op_class):
    # TODO: merge with dist handler
    f = simple_funs[op_class]

    @register_handler(op_class)
    def handle(op, *numpyro_pars, is_observed):
        return f(*numpyro_pars)


for op_class in simple_funs:
    register_simple_fun_handler(op_class)


@register_handler(ir.Constant)
def handle_constant(op: ir.Constant, *, is_observed):
    value = jnp.array(op.value)  # return a jax array, not a numpy array
    return value


@register_handler(ir.Index)
def handle_index(op: ir.Index, val, *indices, is_observed):
    stuff = []
    i = 0
    for my_slice in op.slices:
        if my_slice:
            stuff.append(my_slice)
        else:
            stuff.append(indices[i])
            i += 1
    stuff = tuple(stuff)
    return val[stuff]


@register_handler(ir.Sum)
def handle_sum(op: ir.Sum, val, *, is_observed):
    return jnp.sum(val, axis=op.axis)


@register_handler(ir.Composite)
def handle_composite(op: ir.Composite, *numpyro_parents, is_observed):
    vals = list(numpyro_parents)
    assert len(numpyro_parents) == op.num_inputs
    for n, (my_cond_dist, my_par_nums) in enumerate(zip(op.ops, op.par_nums, strict=True)):
        my_parents = [vals[i] for i in my_par_nums]
        my_is_observed = is_observed and n == len(op.ops)
        new_val = get_numpyro_val(my_cond_dist, *my_parents, is_observed=my_is_observed)
        vals.append(new_val)
    return vals[-1]
