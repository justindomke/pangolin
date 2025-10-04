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


class Handler:
    def sample(self, key, parent_values: Sequence[ArrayLike]):
        pass

    def log_prob(self, value, parent_values: Sequence[ArrayLike]):
        pass


class NumpyroHandler(Handler):
    """
    Given a function to bind a set of parent values into a Numpyro dist, create a Handler
    """

    def __init__(self, binder: Callable):
        self.binder = binder

    def bind(self, parent_values: Sequence[ArrayLike]):
        return self.binder(*parent_values)

    def sample(self, key, parent_values: Sequence[ArrayLike]):
        return self.bind(parent_values).sample(key)

    def log_prob(self, value: ArrayLike, parent_values: Sequence[ArrayLike]):
        return self.bind(parent_values).log_prob(value)


class DeterministicHandler(Handler):
    """
    Given a (jax) function to compute the output, compute the output
    """

    def __init__(self, fun: Callable):
        self.fun = fun

    def sample(self, key, parent_values: Sequence[ArrayLike]):
        return self.fun(*parent_values)

    def log_prob(self, value: ArrayLike, parent_values: Sequence[ArrayLike]):
        return 0.0  # should this raise an error?


simple_dist_binders = {
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

simple_dist_handlers = {
    op_class: NumpyroHandler(simple_dist_binders[op_class])
    for op_class in simple_dist_binders
}


def get_handler(op):
    op_class = type(op)
    if op_class in simple_dist_handlers:
        return simple_dist_handlers[op_class]
    else:
        raise NotImplementedError("")


# def register_simple_dist_handler(op_class):
#     d = simple_dists[op_class]

#     @register_handler(op_class)
#     def handle(op, *numpyro_pars, is_observed):
#         # op is ignored, because simple!
#         return d(*numpyro_pars)

# def sample_op(op: Op, rng_key, parent_values: Sequence[ArrayLike]):
#     """
#     Given a single `Op` and parent values, draw a sample.
#     """
#     pass


# def log_prob_op(
#     op: Op,
#     value: ArrayLike,
#     parent_values: Sequence[ArrayLike],
# ):
#     """
#     Given a single `Op`, evaluate log_prob.
#     """
#     pass
