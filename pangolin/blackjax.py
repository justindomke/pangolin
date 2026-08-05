"""
This module defines a convenient interface to call
`Blackjax <https://blackjax-devs.github.io/blackjax/>`_
to do inference. You could of course just call `pangolin.jax_backend.ancestor_log_prob`
to get a plain jax function and then call Blackjax yourself. But this module abstracts
away all the details.
"""

from __future__ import annotations
from jax import numpy as jnp
import jax.tree_util
import numpy as np
from typing import Any, Callable, TypeAlias, TYPE_CHECKING, Type, Sequence, List, Optional, Iterable
from pangolin import ir
from pangolin.ir import Op, RV, ArrayLike
from numpy.typing import ArrayLike
from numpyro import distributions as dist
from jax.scipy import special as jspecial
from jax import nn as jnn
from pangolin import dag, util
from pangolin import jax_backend
import blackjax
from jaxtyping import PyTree
import functools
from pangolin.calculate import Calculate

__all__ = ["sample", "E", "var", "std", "run_nuts", "nuts", "blackjax_calculate"]


from pangolin.jax_backend.bijectors import default_bijector_dict


################################################################################
# NUTS
################################################################################


def nuts_inference_loop(rng_key, kernel, initial_states, num_samples):
    @jax.jit
    def one_step(states, rng_key):
        states, infos = kernel(rng_key, states)
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return (states, infos)


def run_nuts(log_prob, key, initial_state, num_samples):
    """
    Given a density, do NUTS inference

    """

    # to do hmc instead:
    # adapt = blackjax.window_adaptation(blackjax.hmc, log_prob, num_integration_steps=60)
    # kernel = blackjax.hmc(log_prob, **parameters).step

    sample_key, warmup_key = jax.random.split(key)
    adapt = blackjax.window_adaptation(blackjax.nuts, log_prob)

    (last_state, parameters), _ = adapt.run(warmup_key, initial_state, num_samples)  # type: ignore

    kernel = blackjax.nuts(log_prob, **parameters).step
    states, infos = nuts_inference_loop(sample_key, kernel, last_state, num_samples)
    return states.position


################################################################################
# Pathfinder
################################################################################


def run_pathfinder(log_prob, key, initial_state, num_samples, elbo_samples=200, **lbfgs_kwargs):
    approx_key, sample_key = jax.random.split(key)
    state, _ = blackjax.vi.pathfinder.approximate(approx_key, log_prob, initial_state, elbo_samples, **lbfgs_kwargs)
    samples, _logq = blackjax.vi.pathfinder.sample(sample_key, state, num_samples)
    return samples


################################################################################
# Generic sampling method
################################################################################

# TODO: Raise error if given discrete latent variable


def sample_flat(
    vars: list[RV],
    given_vars: list[RV],
    given_vals: list,
    *,
    run_inf: Callable,
    bijector_dict: Optional[dict] = default_bijector_dict,
    **inf_args,
) -> list[jnp.ndarray]:
    """
    Given a "flat" specification of an inference problem, do inference using Numpyro. The basic algorithm is:

    1. Use `get_model_flat` to create a Numpyro model and a mapping from RVs to Numpyro variable names.
    2. Use standard Numpyro routines to do inference using MCMC.
    3. Use the name mapping to extract samples for all variables listed in `vars`.

    If given is empty, then example samples will (automatically) be drawn using `ancestor_sample_flat`.

    Parameters
    ----------
    vars
        The RVs you want to sample from
    given
        The RVs you want to condition on
    vals
        The values for the conditioned RVs
    run_inf
        runner for blackjax inference routine
    niter: int, optional
        The number of iterations / samples to draw

    Returns
    -------
    samples: list[jnp.ndarray]
        Samples for each variable in `vars`.

    Examples
    --------
    >>> x = ir.RV(ir.Constant(0.5))
    >>> y = ir.RV(ir.Normal(), x, x)
    >>> z = ir.RV(ir.Normal(), y, x)
    >>> [samps_x, samps_y] = sample_flat([x, y], [z], [3.0], run_inf=run_nuts, num_samples=30)
    >>> samps_x.shape
    (30,)
    >>> samps_y.shape
    (30,)
    >>> np.allclose(samps_x, 0.5)
    True
    >>> np.allclose(samps_y, 0.5)
    False

    """

    if len(given_vars) != len(given_vals):
        raise ValueError("length of given_vars not equal to length of given_vals")

    if any(not v.op.random for v in given_vars):
        nonrandom_ops = [v.op for v in given_vars if not v.op.random]
        raise ValueError(f"Cannot condition on RV with non-random op(s) {nonrandom_ops}")

    # if no given variables, just do ancestor sampling (works but disabled for simplicity)
    # if len(given_vars) == 0:
    #     key = jax.random.PRNGKey(0)
    #     keys = jax.random.split(key, niter)
    #     mysample = lambda key: backend.ancestor_sample_flat(vars, key)
    #     return jax.vmap(mysample)(keys)

    given_vals = [jnp.array(val) for val in given_vals]
    all_vars = dag.upstream_nodes(tuple(vars) + tuple(given_vars))
    latent_vars = [var for var in all_vars if var.op.random and var not in given_vars]

    for v in latent_vars:
        if v.op.discrete:
            raise ValueError(f"Blackjax backend does not support discrete latent/unobserved RV (saw op {v.op})")

    @jax.jit
    def log_prob(latent_vals):
        return jax_backend.ancestor_log_prob_flat(latent_vars + given_vars, latent_vals + given_vals, bijector_dict)

    # key = jax.random.PRNGKey(0)
    seed = np.random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    latent_vals = jax_backend.ancestor_sample_flat(latent_vars, key, bijector_dict=bijector_dict)
    latent_samps = run_inf(log_prob, key, latent_vals, **inf_args)

    if bijector_dict is not None and len(latent_samps) > 0:

        # def constrain(latent_vals):
        #     return jax_backend.ancestor_constrain(latent_vars, latent_vals, bijector_dict)

        def constrain(latent_vals):
            all_unconstrained = jax_backend.ancestor_constrain(
                latent_vars + given_vars, latent_vals + given_vals, bijector_dict
            )
            latent_unconstrained = all_unconstrained[: len(latent_vars)]
            return latent_unconstrained

        latent_samps = jax.vmap(constrain)(latent_samps)

    def fill(latent_vals):
        return jax_backend.fill_in(latent_vars + given_vars, latent_vals + given_vals, vars)

    # include niter in case latent_samps is empty
    # return jax.vmap(fill, axis_size=niter)(latent_samps)

    # TODO: extracting num_samples like this is not elegant
    return jax.vmap(fill, axis_size=inf_args["num_samples"])(latent_samps)


# sample_nuts = functools.partial(sample_flat, run_inf=run_nuts)
# sample_pathfinder = functools.partial(sample_flat, run_inf=run_pathfinder)


default = {"num_samples": 1000}

calc = Calculate(sample_flat, **default)
sample = calc.sample
"""
Default version of `Calculate.sample` that draws 1000 samples.
"""
E = calc.E
"""
Default version of `Calculate.E` that uses 1000 samples.
"""
var = calc.var
"""
Default version of `Calculate.var` that uses 1000 samples.
"""
std = calc.std
"""
Default version of `Calculate.std` that uses 1000 samples.
"""

sample_arviz = calc.sample_arviz
"""
Default version of `Calculate.sample_arviz` that uses 1000 samples.
"""


# def inf_until_match(inf, vars, given, vals, testfun, niter_start=1000, niter_max=100000):
#     from time import time

#     niter = niter_start
#     while niter <= niter_max:
#         t0 = time()
#         out = inf(vars, given, vals, niter=niter)
#         t1 = time()
#         print(f"{niter=} {t1 - t0}")
#         if testfun(out):
#             assert True
#             return
#         else:
#             niter *= 2
#     assert False


# import functools

# sample_until_match = functools.partial(inf_until_match, sample)


# def sample_flat_until_match(vars, given, vals, testfun, niter_start=1000, niter_max=100000):
#     new_testfun = lambda stuff: testfun(stuff[0])
#     return inf_until_match(sample_flat, vars, given, vals, new_testfun, niter_start, niter_max)

import textwrap, inspect


def blackjax_calculate(run_inf, frozen: Iterable[str] = (), **options) -> Calculate:
    """Given a function that calls blackjax, wrap it into a convenient `Calculate` object.

    Parameters
    ----------
    run_inf
        inference routine that calls blackjax. Should have signature ``run_inf(log_prob, key, initial_state, **options) -> samples`` where ``log_prob`` is a jax function that evaluates the log probability, ``key`` is a Jax PRNGKey, and ``initial_state`` is a latent state from which to initialize inference.
    frozen
        parameters that cannot be overriden from `options`
    options
        default options



    Examples
    --------
    >>> nuts = blackjax_calculate(run_nuts)

    """

    calc = Calculate(sample_flat, run_inf=run_inf, frozen=frozenset(frozen) | {"run_inf"}, **options)
    calc.__doc__ = (
        f"Inference engine using {run_inf.__name__}.\n\n"
        f"Options are forwarded to {run_inf.__name__}:\n\n"
        + textwrap.indent(inspect.getdoc(run_inf) or "(undocumented)", "    ")
    )
    return calc


nuts = blackjax_calculate(run_nuts)
"Engine bound to NUTS. Options forwarded to `run_nuts`."
