# from . import interface, dag, util, inference
# import jax.tree_util
# import functools
from jax import numpy as jnp
from numpyro import distributions as dist
from typing import Sequence, Callable

# import numpyro
# from jax import lax
# from numpyro.distributions import util as dist_util
# from jax.scipy import special as jspecial
from jax import nn as jnn

# import numpy as np
from pangolin.ir import RV
from pangolin import dag, util, ir

# from pangolin.interface.base import OperatorRV
from numpy.typing import ArrayLike

# from pangolin.interface import RV_or_ArrayLike
# from pangolin.inference import inference_util
# import numpy as np

RV_or_ArrayLike = RV | ArrayLike


def get_model_flat(
    vars: list[RV], given: list[RV], vals: list[ArrayLike]
) -> tuple[Callable, dict[RV, str]]:
    """
    This is the core function that makes inference possible. Given a "flat" specification of an inference problem, get a Numpyro model. The basic algorithm is quite simple:
    1. Get all variables upstream of `vars` and `given`
    2. Define a new model function that goes through those variables in order, and does the
    appropriate numpyro operations for each

    Parameters
    ----------
    vars: list[RV]
        A flat sequence of random variables to define a model over
    given: list[RV]
        A flat sequence of random variables to condition on
    vals: list[ArrayLike]
        A flat sequence of constants (should have `len(vals)==len(given)` and matching shapes for
        each element)

    Returns
    -------
    model: Callable
        a new numpyro model, which can be used just like any other numpyro model with normal numpyro utilities
    names: dict[RV, str]
        a dict mapping all upstream RVs to the names of the random variables in the model corresponding to `vars`

    Examples
    --------
    >>> x = ir.RV(ir.Constant(0.5))
    >>> y = ir.RV(ir.Constant(3.0))
    >>> z = ir.RV(ir.Mul(), x, y)
    >>> model, names = get_model_flat([z], [], []) # [x, y, z] same
    >>> names == {x: 'v0', y: 'v1', z: 'v2'}
    True
    >>> with numpyro.handlers.seed(rng_seed = jax.random.PRNGKey(0)):
    ...     model()
    {'v0': Array(0.5, dtype=float32), 'v1': Array(3., dtype=float32), 'v2': Array(1.5, dtype=float32)}

    >>> x = ir.RV(ir.Constant(0.5))
    >>> y = ir.RV(ir.Normal(), x, x)
    >>> model, names = get_model_flat([y], [], []) # [x, y] same
    >>> names == {x: 'v0', y: 'v1'} # names maps rvs to strings
    True
    >>> with numpyro.handlers.seed(rng_seed = jax.random.PRNGKey(0)):
    ...     model()
    {'v0': Array(0.5, dtype=float32), 'v1': Array(-0.12576944, dtype=float32)}

    >>> x = ir.RV(ir.Constant(0.5))
    >>> y = ir.RV(ir.Normal(), x, x)
    >>> z = ir.RV(ir.Normal(), y, x)
    >>> model, names = get_model_flat([y], [z], [2.0]) # [y] same
    >>> names == {x: 'v0', y: 'v1', z: 'v2'}
    True
    >>> from numpyro.infer import MCMC, NUTS
    >>> kernel = NUTS(model)
    >>> mcmc = MCMC(kernel, num_warmup=10, num_samples=3)
    >>> mcmc.run(jax.random.PRNGKey(0))
    >>> samples = mcmc.get_samples()
    >>> samples['v0']
    Array([0.5, 0.5, 0.5], dtype=float32)
    >>> samples['v1']
    Array([1.0675064, 1.1308542, 1.10018  ], dtype=float32)

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
        raise ValueError(
            f"length of given ({len(given)}) does not match length of vals ({len(vals)})"
        )

    for var, val0 in zip(given, vals, strict=True):
        val = jnp.array(val0)
        if not util.is_numeric_numpy_array(val):
            raise ValueError("given val {val} not numeric")
        if var.shape != val.shape:
            raise ValueError(
                "given var {var} with shape {var.shape} does not match corresponding given val {val} with shape {val.shape}"
            )

    all_vars = dag.upstream_nodes(tuple(vars) + tuple(given))

    name_to_var: dict[str, RV] = {}
    var_to_name: dict[RV, str] = {}
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
