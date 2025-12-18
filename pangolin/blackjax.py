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
from typing import (
    Any,
    Callable,
    TypeAlias,
    TYPE_CHECKING,
    Type,
    Sequence,
    List,
    Optional,
)
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

__all__ = ["sample", "E", "var", "std", "Calculate", "inf_until_match"]


def inference_loop(rng_key, kernel, initial_states, num_samples):
    @jax.jit
    def one_step(states, rng_key):
        states, infos = kernel(rng_key, states)
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return (states, infos)


def run_nuts(log_prob, key, initial_state, num_samples):
    # to do hmc instead:
    # adapt = blackjax.window_adaptation(blackjax.hmc, log_prob, num_integration_steps=60)
    # kernel = blackjax.hmc(log_prob, **parameters).step

    sample_key, warmup_key = jax.random.split(key)
    adapt = blackjax.window_adaptation(blackjax.nuts, log_prob)

    (last_state, parameters), _ = adapt.run(warmup_key, initial_state, num_samples)  # type: ignore
    kernel = blackjax.nuts(log_prob, **parameters).step
    states, infos = inference_loop(sample_key, kernel, last_state, num_samples)
    return states.position


def sample_flat(
    vars: list[RV],
    given_vars: list[RV],
    given_vals: list,
    *,
    niter: int,
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
    >>> [samps_x, samps_y] = sample_flat([x, y], [z], [3.0], niter=30)
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
        raise ValueError(
            f"Cannot condition on RV with non-random op(s) {nonrandom_ops}"
        )

    # if no given variables, just do ancestor sampling (works but disabled for simplicity)
    # if len(given_vars) == 0:
    #     key = jax.random.PRNGKey(0)
    #     keys = jax.random.split(key, niter)
    #     mysample = lambda key: backend.ancestor_sample_flat(vars, key)
    #     return jax.vmap(mysample)(keys)

    given_vals = [jnp.array(val) for val in given_vals]
    all_vars = dag.upstream_nodes(tuple(vars) + tuple(given_vars))
    latent_vars = [var for var in all_vars if var.op.random and var not in given_vars]

    @jax.jit
    def log_prob(latent_vals):
        return jax_backend.ancestor_log_prob_flat(
            latent_vars + given_vars, latent_vals + given_vals
        )

    key = jax.random.PRNGKey(0)
    latent_vals = jax_backend.ancestor_sample_flat(latent_vars, key)
    latent_samps = run_nuts(log_prob, key, latent_vals, niter)

    def fill(latent_vals):
        return jax_backend.fill_in(
            latent_vars + given_vars, latent_vals + given_vals, vars
        )

    # include niter in case latent_samps is empty
    return jax.vmap(fill, axis_size=niter)(latent_samps)


class Calculate:
    """
    A `Calculate` object just remembers a set of options and then offers inference
    methods.

    Parameters
    ----------
    default
        represents a set of options for the inference engine that can be overriden later
    frozen
        represents a set of options for the inference engine that cannot be overridden

    """

    def __init__(self, default: Optional[dict] = None, frozen: Optional[dict] = None):

        if default is None:
            default = {}
        if frozen is None:
            frozen = {}

        if util.intersects(default, frozen):
            raise ValueError(f"default {default} intersects with frozen {frozen}")

        self.default = default
        self.frozen = frozen  # options for that engine

    def sample(
        self,
        vars: PyTree[RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        reduce_fn: Optional[Callable] = None,
        **options,
    ):
        """
        Draw samples!

        Args:
            vars: A `RV` or list/tuple of `RV` or pytree of `RV` to sample.
            given_vars: A `RV` or list/tuple of `RV` or pytree of `RV` to condition on.
                ``None`` indicates no conditioning variables.
            given_vals: An ``ArrayLike`` or list/tuple of ``ArrayLike`` or pytree of
                ``ArrayLike`` representing observed values. Must match the structure and
                shape of ``given_vars``.
            reduce_fn:  Function to apply to each leaf node in samples before returning.
                This is used to create `E`, `var`, etc. (If ``None``, does nothing.)
            options: extra options to pass to sampler

        Returns:
            Pytree of JAX arrays matching structure and shape of ``vars`` but with one
            extra dimension at the start, containing the samples.

        Examples
        --------
        >>> zero    = ir.RV(ir.Constant(0))
        >>> one     = ir.RV(ir.Constant(1))
        >>> x       = ir.RV(ir.Normal(), zero, one)
        >>> y       = ir.RV(ir.Normal(), x, one)
        >>> calc    = Calculate({'niter': 529})
        >>> x_samps = calc.sample(x,y,2)
        >>> x_samps.shape
        (529,)
        >>> np.mean(x_samps) # something close to 1.0
        Array(...)
        """

        if util.intersects(options, self.frozen):
            raise ValueError(f"options intersects frozen")

        options = self.default | options  # overrides defaults

        (
            flat_vars,
            flat_given_vars,
            flat_given_vals,
            unflatten,
            unflatten_given,
        ) = util.flatten_args(vars, given_vars, given_vals)

        flat_samps = sample_flat(
            flat_vars,
            flat_given_vars,
            flat_given_vals,
            **options,
            **self.frozen,
        )

        if reduce_fn is not None:
            flat_samps = map(reduce_fn, flat_samps)

        return unflatten(flat_samps)

    def E(
        self,
        vars: PyTree[RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        **options,
    ):
        """
        Compute (conditional) expected values. This is just a thin wrapper that calls
        `sample` and then reduces by taking the mean.

        Args:
            vars: A `RV` or list/tuple of `RV` or pytree of `RV` to sample.
            given_vars:  A `RV` or list/tuple of `RV` or pytree of `RV` to condition on.
                ``None`` indicates no conditioning variables.
            given_vals: An ``ArrayLike`` or list/tuple of ``ArrayLike`` or pytree of
                ``ArrayLike`` representing observed values. Must match the structure and
                shape of ``given_vars``.
            reduce_fn:  Function to apply to each leaf node in samples before returning.
                This is used to create `E`, `var`, etc. (If ``None``, does nothing.)
            options: extra options to pass to sampler

        Returns:
            Pytree of JAX arrays matching structure and shape of ``vars``, containing
                the expectations.


        Examples
        --------
        >>> zero    = ir.RV(ir.Constant(0))
        >>> one     = ir.RV(ir.Constant(1))
        >>> x       = ir.RV(ir.Normal(), zero, one)
        >>> y       = ir.RV(ir.Normal(), x, one)
        >>> calc    = Calculate({'niter': 529})
        >>> calc.E(x,y,2) # something close to 1.0
        Array(...)
        """

        return self.sample(
            vars, given_vars, given_vals, lambda x: np.mean(x, axis=0), **options
        )

    def var(
        self,
        vars: PyTree[RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        **options,
    ):
        return self.sample(
            vars, given_vars, given_vals, lambda x: np.var(x, axis=0), **options
        )

    def std(
        self,
        vars: PyTree[RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        **options,
    ):
        return self.sample(
            vars, given_vars, given_vals, lambda x: np.std(x, axis=0), **options
        )

    def sample_arviz(
        self,
        vars: dict[str, RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        **options,
    ):
        """This is an **experimental** function to draw samples in
        `ArviZ <https://www.arviz.org/en/latest/>`__ format.

        Note: ArviZ is not installed with pangolin by default: You must install it
        manually.

        Args:
            vars: dictionary mapping names to individual random variables
                given_vars: A `RV` or list/tuple of `RV` or pytree of `RV` to condition on.
                ``None`` indicates no conditioning variables.
            given_vars: A `RV` or list/tuple of `RV` or pytree of `RV` to condition on.
                given_vals: An ``ArrayLike`` or list/tuple of ``ArrayLike`` or pytree of
                ``ArrayLike`` representing observed values. Must match the structure and
                shape of ``given_vars``.
            reduce_fn:  Function to apply to each leaf node in samples before returning.
                This is used to create `E`, `var`, etc. (If ``None``, does nothing.)
            options: extra options to pass to sampler
        """

        try:
            from arviz import convert_to_inference_data
        except ImportError:
            raise ImportError("To use this method you must install arviz manually")

        samps = self.sample(vars, given_vars, given_vals, **options)
        samps_with_none = {key: samps[key][None, ...] for key in samps}
        dataset = convert_to_inference_data(samps_with_none)
        return dataset


default = {"niter": 1000}

calc = Calculate(default)
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


def inf_until_match(
    inf, vars, given, vals, testfun, niter_start=1000, niter_max=100000
):
    from time import time

    niter = niter_start
    while niter <= niter_max:
        t0 = time()
        out = inf(vars, given, vals, niter=niter)
        t1 = time()
        print(f"{niter=} {t1 - t0}")
        if testfun(out):
            assert True
            return
        else:
            niter *= 2
    assert False


import functools

sample_until_match = functools.partial(inf_until_match, sample)


def sample_flat_until_match(
    vars, given, vals, testfun, niter_start=1000, niter_max=100000
):
    new_testfun = lambda stuff: testfun(stuff[0])
    return inf_until_match(
        sample_flat, vars, given, vals, new_testfun, niter_start, niter_max
    )
