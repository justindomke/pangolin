from jax import numpy as jnp
import jax.tree_util
import numpy as np
from typing import Any, Callable, TypeAlias, TYPE_CHECKING, Type, Sequence, List
from pangolin import ir
from pangolin.ir import Op, RV
from numpy.typing import ArrayLike
from numpyro import distributions as dist
from jax.scipy import special as jspecial
from jax import nn as jnn
from pangolin import dag, util
from pangolin import backend
import blackjax


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
    given_vals: list[ArrayLike],
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
    vars: list[RV]
        The RVs you want to sample from
    given: list[RV]
        The RVs you want to condition on
    vals:
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

    # if no given variables, just do ancestor sampling (works but disabled for efficiency)
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
        return backend.ancestor_log_prob_flat(
            latent_vars + given_vars, latent_vals + given_vals
        )

    key = jax.random.PRNGKey(0)
    latent_vals = backend.ancestor_sample_flat(latent_vars, key)
    latent_samps = run_nuts(log_prob, key, latent_vals, niter)

    def fill(latent_vals):
        return backend.fill_in(latent_vars + given_vars, latent_vals + given_vals, vars)

    # include niter in case latent_samps is empty
    return jax.vmap(fill, axis_size=niter)(latent_samps)


class Calculate:
    def __init__(self, default: None | dict = None, frozen: None | dict = None):
        """
        Create a `Calculate` object.

        Inputs:
        * `**options`: options to "freeze" in
        """

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
        vars,
        given_vars=None,
        given_vals=None,
        reduce_fn=None,
        **options,
    ):
        """
        Draw samples!

        Inputs:
        * `vars`: a pytree of `RV`s to sample. (Can be any pytree)
        * `given_vars`: a pytree of `RV`s to condition on. `None` is no conditioning
        variables. (
        Can be any pytree)
        * `given_vals`: a pytree of observed values. (Pytree must match `given_vars`.)
        * `reduce_fn` (optional) will apply a function to the samples for each `RV` in
        `vars` before returning samples. (This is used to define `E`, `var`, etc.
        below.)

        Outputs:
        * A `pytree` of `RV`s matching `vars` with one extra dimension, containing
        the samples.

        Example:
        ```python
        x = normal(0,1)
        y = normal(x,1)
        sample(x,y,2) # returns something close to 1.0
        ```
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

    def E(self, vars, given_vars=None, given_vals=None, **options):
        return self.sample(
            vars, given_vars, given_vals, lambda x: np.mean(x, axis=0), **options
        )

    def var(self, vars, given_vars=None, given_vals=None, **options):
        return self.sample(
            vars, given_vars, given_vals, lambda x: np.var(x, axis=0), **options
        )

    def std(self, vars, given_vars=None, given_vals=None, **options):
        return self.sample(
            vars, given_vars, given_vals, lambda x: np.std(x, axis=0), **options
        )


default = {"niter": 1000}

calc = Calculate(default)
sample = calc.sample
E = calc.E
var = calc.var
std = calc.std


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
