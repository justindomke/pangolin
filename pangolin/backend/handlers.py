"""
Plan:

* `sample_op(op, rng_key, parent_values)`
* `log_prob_op(op, value, parent_values)`
* `ancestor_sample_flat(vars, rng_key, given_vars, given_values)`
* `ancestor_log_prob_flat(vars, values, given_vars, given_values)`
"""

from jax import numpy as jnp
import jax.tree_util
import numpy as np
from typing import Sequence, Any, Callable, TypeAlias, TYPE_CHECKING
from pangolin import ir
from pangolin.ir import Op, RV
from numpy.typing import ArrayLike
from numpyro import distributions as dist
from jax.scipy import special as jspecial
from jax import nn as jnn
from pangolin import dag


# log_prob_funs is a dict mapping op *types* to functions of the form log_prob(op, value, parent_values)


# class Handler:
#     def sample(self, key, parent_values: Sequence[ArrayLike]):
#         raise NotImplementedError()

#     def log_prob(self, value, parent_values: Sequence[ArrayLike]):
#         raise NotImplementedError()


simple_dists = {
    ir.Normal: dist.Normal,
    ir.NormalPrec: lambda loc, prec: dist.Normal(loc, 1 / prec**2),
    ir.Bernoulli: dist.Bernoulli,
    ir.BernoulliLogit: dist.BernoulliLogits,
    ir.Beta: dist.Beta,
    ir.BetaBinomial: lambda n, a, b: dist.BetaBinomial(
        a, b, n
    ),  # numpyro has a different order
    ir.Binomial: dist.Binomial,
    ir.Categorical: dist.Categorical,
    ir.Cauchy: dist.Cauchy,
    ir.Exponential: dist.Exponential,
    ir.Dirichlet: dist.Dirichlet,
    ir.Gamma: dist.Gamma,
    ir.Lognormal: dist.LogNormal,
    ir.Multinomial: dist.Multinomial,
    ir.MultiNormal: dist.MultivariateNormal,
    ir.Poisson: dist.Poisson,
    ir.StudentT: dist.StudentT,
    ir.Uniform: dist.Uniform,
}

log_prob_handlers = {}
sample_handlers = {}

for op_class in simple_dists:
    bind = simple_dists[op_class]

    def my_log_prob(op, value, parent_values):
        bound_dist: dist.Distribution = bind(parent_values)
        return bound_dist.log_prob(value)

    log_prob_handlers[op_class] = my_log_prob

    def my_sample(op, key, parent_values):
        bound_dist: dist.Distribution = bind(parent_values)
        return bound_dist.sample(key)

    sample_handlers[op_class] = my_sample


# class NumpyroHandler(Handler):
#     """
#     Given a function to bind a set of parent values into a Numpyro dist, create a Handler
#     """

#     def __init__(self, binder: Callable):
#         self.binder = binder

#     def bind(self, parent_values: Sequence[ArrayLike]):
#         return self.binder(*parent_values)

#     def sample(self, key, parent_values: Sequence[ArrayLike]):
#         return self.bind(parent_values).sample(key)

#     def log_prob(self, value: ArrayLike, parent_values: Sequence[ArrayLike]):
#         return self.bind(parent_values).log_prob(value)


# simple_dist_handlers = {
#     op_class: NumpyroHandler(simple_dists[op_class]) for op_class in simple_dists
# }


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
    ir.Matmul: jnp.matmul,
    ir.Inv: jnp.linalg.inv,
    ir.Softmax: jnn.softmax,
    ir.SimpleIndex: ir.index_orthogonal_no_slices,
}

for op_class in simple_funs:
    fun = simple_funs[op_class]
    sample_handlers[op_class] = lambda op, key, parent_values: fun(*parent_values)


# class DeterministicHandler(Handler):
#     """
#     Given a (jax) function to compute the output, compute the output
#     """

#     def __init__(self, fun: Callable):
#         self.fun = fun

#     def sample(self, key, parent_values: Sequence[ArrayLike]):
#         return self.fun(*parent_values)

#     # def log_prob(self, value: ArrayLike, parent_values: Sequence[ArrayLike]):
#     #     return 0.0  # should this raise an error?


# simple_fun_handlers = {
#     op_class: DeterministicHandler(simple_funs[op_class]) for op_class in simple_funs
# }


# def get_handler(op):
#     op_class = type(op)
#     if op_class in simple_dist_handlers:
#         return simple_dist_handlers[op_class]
#     elif op_class in simple_fun_handlers:
#         return simple_fun_handlers[op_class]
#     else:
#         raise NotImplementedError("")


def sample_op(op: Op, key, parent_values: Sequence[ArrayLike]):
    """
    Given a single `Op` and parent values, draw a sample.
    """
    # handler = get_handler(op)
    # return handler.sample(key, parent_values)
    op_class = type(op)
    return sample_handlers[op_class](op, key, parent_values)


def log_prob_op(
    op: Op,
    value: ArrayLike,
    parent_values: Sequence[ArrayLike],
):
    """
    Given a single `Op`, evaluate log_prob.
    """
    if not op.random:
        raise ValueError("Cannot evaluate log_prob for non-random op")
    op_class = type(op)
    return log_prob_handlers[op_class](op, value, parent_values)


def ancestor_sample_flat(vars: Sequence[RV], key):
    all_vars = dag.upstream_nodes(vars)
    all_values = {}
    for var in all_vars:
        parent_values = [all_values[p] for p in var.parents]
        key, subkey = jax.random.split(key)
        all_values[var] = sample_op(var.op, subkey, parent_values)
    return [all_values[var] for var in vars]
