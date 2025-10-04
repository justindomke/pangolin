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
from pangolin.ir import RV
from pangolin import dag, util, ir

# from pangolin.interface.base import OperatorRV
from numpy.typing import ArrayLike

# from pangolin.interface import RV_or_ArrayLike
from pangolin.inference import inference_util
import numpy as np

RV_or_ArrayLike = RV | ArrayLike

from .model import get_model_flat

from numpyro import enable_validation

enable_validation()  # seems to detect illegal parameters instead of just returning garbage


def generate_seed(size=()):
    import numpy as np  # import here to prevent accidental use elsewhere in a jax shop

    info = np.iinfo(np.int32)
    return np.random.randint(info.min, info.max, size=size)


def generate_key():
    seed = generate_seed()
    return jax.random.PRNGKey(seed)


def ancestor_sample_flat(
    vars: list[RV], *, niter: int | None = None
) -> list[jnp.ndarray]:
    """
    Given a set of RVs, draw exact samples via ancestor sampling. If `niter` is one, then a single sample is drawn.

    Parameters
    ----------
    vars: list[RV]
        variables to sample
    niter: int or None, optional
        number of samples to draw for each variable. by default (None) just a single sample

    Returns
    -------
    samples: list[array]
        list of samples for each variable in `vars`. If `niter=None` then each element will have the same shape as `vars`. If `niter` is int, then each element will have an additional dimension of size `niter`.

    Examples
    --------
    >>> x = ir.RV(ir.Constant(0.5))
    >>> y = ir.RV(ir.Normal(), x, x)
    >>> [x_sample, y_sample] = ancestor_sample_flat([x, y])
    >>> x_sample.shape
    ()
    >>> y_sample.shape
    ()
    >>> [x_samples, y_samples] = ancestor_sample_flat([x, y], niter=17)
    >>> x_samples.shape
    (17,)
    >>> y_samples.shape
    (17,)

    """

    model, names = get_model_flat(vars, [], [])

    def base_sample(seed):
        with numpyro.handlers.seed(rng_seed=seed):
            out = model()
            return [out[names[var]] for var in vars]

    if niter is None:
        my_seed = generate_key()
        return base_sample(my_seed)
    else:
        seeds = generate_seed(niter)
        return jax.vmap(base_sample)(seeds)


def sample_flat(
    vars: list[RV], given: list[RV], vals: list[ArrayLike], *, niter: int = 10000
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

    # assert isinstance(vars, Sequence)
    # assert isinstance(given, Sequence)
    # assert isinstance(vals, Sequence)
    assert len(given) == len(vals)

    # TODO: activate ancestor sampling
    if len(given) == 0:
        # print("ancestor sampling")
        return ancestor_sample_flat(vars, niter=niter)

    # TODO:z
    # raise an exception if no random vars
    # make sure vals is actually an array (not RV!)

    if any(not v.op.random for v in given):
        nonrandom_ops = [v.op for v in given if not v.op.random]
        raise ValueError(
            f"Cannot condition on RV with non-random op(s) {nonrandom_ops}"
        )

    vals = [jnp.array(val) for val in vals]

    model, names = get_model_flat(vars, given, vals)

    def infer(kernel):
        mcmc = numpyro.infer.MCMC(
            kernel,
            num_warmup=niter,
            num_samples=niter,
            progress_bar=False,
        )
        key = generate_key()

        # numpyro gives some annoying future warnings
        import warnings

        with warnings.catch_warnings(action="ignore", category=FutureWarning):  # type: ignore
            mcmc.run(key)

        # numpyro_allocation_error = False
        #
        # try:
        #     mcmc.run(key)
        # except ValueError as e:
        #     if str(e).startswith("Ran out of free dims during allocation"):
        #         numpyro_allocation_error = True
        #     else:
        #         raise e
        #
        # if numpyro_allocation_error:
        #     raise ValueError("NumPyro raised a allocation error.\n"
        #                      "This usually indicates that it wasn't able to integrate out all "
        #                      "discrete latent variables.")

        # wierd trick to get samples for deterministic sites
        latent_samples = mcmc.get_samples()
        if latent_samples == {}:
            predictive = numpyro.infer.Predictive(
                model, num_samples=niter, infer_discrete=True
            )
        else:
            predictive = numpyro.infer.Predictive(
                model, latent_samples, infer_discrete=True
            )
        predictive_samples = predictive(key)
        # print(f"{predictive_samples=}")
        # merge
        samples = {**latent_samples, **predictive_samples}
        return samples

    kernel = numpyro.infer.NUTS(model)
    samples = infer(kernel)

    # try:
    #     kernel = numpyro.infer.DiscreteHMCGibbs(
    #         numpyro.infer.NUTS(model), modified=True
    #     )
    #     # kernel = numpyro.infer.MixedHMC(numpyro.infer.HMC(model), modified=False)
    #     samples = infer(kernel)
    # except AssertionError as e:
    #     # print(f"FALLING BACK TO NUTS: {e}")
    #     if str(e) != "Cannot detect any discrete latent variables in the model.":
    #         raise e
    #     kernel = numpyro.infer.NUTS(model)
    #     # with warnings.simplefilter(action='ignore', category=FutureWarning):
    #     samples = infer(kernel)

    return [samples[names[var]] for var in vars]


# sample = inference_util.get_non_flat_sampler(sample_flat)
calc = inference_util.Calculate(sample_flat)
sample = calc.sample
E = calc.E
var = calc.var
std = calc.std
