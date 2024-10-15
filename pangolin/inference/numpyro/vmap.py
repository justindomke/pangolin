# from . import interface, dag, util, inference
import jax.tree_util
import functools
from jax import numpy as jnp
from numpyro import distributions as numpyro_dist
from typing import Sequence
import numpyro
from jax import lax
from numpyro.distributions import util as dist_util
from jax.scipy import special as jspecial
from jax import nn as jnn
# import numpy as np
from pangolin.ir.rv import RV
from pangolin import dag, util, ir
from pangolin.interface.interface import OperatorRV
from numpy.typing import ArrayLike
from pangolin.interface import RV_or_array
from pangolin.inference import inference_util
import numpy as np

from .model import is_continuous

from .handler_registry import numpyro_handlers, register_handler

@register_handler(ir.VMap)
def handle_vmap(op: ir.VMap, *numpyro_parents, is_observed):
    # if is simple/broadcastable, use broadcasting vmap
    # elif is nonrandom, use nonrandom
    # elif is observed or is non-discrete use class vmap
    # else, can't handle it

    if isinstance(op, ir.VMap) and op.random and not is_continuous(op) and not is_observed:
        raise ValueError("VMap has limited support in numpy backend for discrete and non-observed")

    if op.random:
        return handle_vmap_random(op, *numpyro_parents, is_observed=is_observed)
    else:
        return handle_vmap_nonrandom(op, *numpyro_parents, is_observed=is_observed)

def handle_vmap_nonrandom(op: ir.VMap, *numpyro_parents, is_observed):
    from pangolin.inference.numpyro.handlers import get_numpyro_val

    assert isinstance(op, ir.VMap)
    assert not op.random

    def base_var(*args):
        return get_numpyro_val(op.base_op, *args, is_observed=is_observed)

    in_axes = op.in_axes
    axis_size = op.axis_size
    args = numpyro_parents
    return jax.vmap(base_var, in_axes=in_axes, axis_size=axis_size)(*args)


def handle_vmap_random(op: ir.VMap, *numpyro_parents, is_observed):
    from pangolin.inference.numpyro.handlers import get_numpyro_val

    assert isinstance(op, ir.VMap)
    assert op.random

    class NewDist(numpyro_dist.Distribution):
        # @property
        # def support(self):
        #     # TODO:
        #     # should be a more elegant solution here...
        #     my_op = op
        #     while isinstance(my_op, ir.VMap):
        #         my_op = my_op.base_op
        #     return get_support(my_op)

        def __init__(self, *args, validate_args=False):
            self.args = args

            # TODO: infer correct batch_shape?
            batch_shape = ()
            parents_shapes = [p.shape for p in args]
            event_shape = op.get_shape(*parents_shapes)

            super().__init__(
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

        def sample(self, key, sample_shape=()):
            assert numpyro.util.is_prng_key(key)
            assert sample_shape == () or sample_shape == (1,)

            def base_sample(key, *args):
                var = get_numpyro_val(op.base_op, *args, is_observed=is_observed)
                return var.sample(key)

            keys = jax.random.split(key, self.event_shape[0])
            in_axes = (0,) + op.in_axes
            axis_size = op.axis_size
            args = (keys,) + self.args
            if sample_shape == (1,):
                return jnp.array([jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)])
            elif sample_shape == ():
                return jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)
            else:
                assert False

        @dist_util.validate_sample
        def log_prob(self, value):
            def base_log_prob(val, *args):
                var = get_numpyro_val(op.base_op, *args, is_observed=is_observed)
                return var.log_prob(val)

            in_axes = (0,) + op.in_axes
            axis_size = op.axis_size
            args = (value,) + self.args
            ls = jax.vmap(base_log_prob, in_axes, axis_size=axis_size)(*args)
            return jnp.sum(ls)

    return NewDist(*numpyro_parents)

def vmap_nesting(op):
    n = 0
    while isinstance(op, ir.VMap):
        op = op.base_op
        n += 1
    return n

simple_discrete_classes = {ir.Bernoulli:numpyro_dist.Bernoulli,
                           ir.Categorical:numpyro_dist.Categorical,
                           ir.Binomial: numpyro_dist.Binomial,
                           ir.BetaBinomial: lambda n,a,b: numpyro_dist.BetaBinomial(a,b,n),
                           ir.Multinomial: numpyro_dist.Multinomial,
                           ir.Poisson: numpyro_dist.Poisson,
                           }

# , ir.BernoulliLogit, ir.Beta, ir.BetaBinomial, ir.Binomial,
# ir.Categorical, ir.Cauchy, ir.Exponential, ir.Gamma, ir.LogNormal, ir.Poisson, ir.StudentT,
# ir.Uniform)


def get_plate_size(op, *numpyro_pars):
    parents_shapes = [p.shape for p in numpyro_pars]
    remaining_shapes, axis_size = ir.vmap.get_sliced_shapes(
        parents_shapes, op.in_axes, op.axis_size
    )
    return axis_size

def get_all_plate_sizes(op, *numpyro_pars):
    sizes = []
    parents_shapes = [p.shape for p in numpyro_pars]
    while isinstance(op, ir.VMap):
        remaining_shapes, axis_size = ir.vmap.get_sliced_shapes(
            parents_shapes, op.in_axes, op.axis_size
        )
        sizes.append(axis_size)
        parents_shapes = remaining_shapes
        op = op.base_op
    return tuple(reversed(sizes))

def update_pars(op, *numpyro_pars):
    my_numpyro_pars = []
    for par, in_axis in zip(numpyro_pars, op.in_axes):
        if in_axis is None:
            my_numpyro_par = par
        elif in_axis == 0:
            my_numpyro_par = par
        else:
            my_numpyro_par = jnp.moveaxis(par, in_axis, 0)
        my_numpyro_pars.append(my_numpyro_par)
    return my_numpyro_pars

# def get_numpyro_rv_discrete_latent(op, name, obs, *numpyro_pars):
#     print("GET DISCRETE LATENT TRIGGERED")
#
#
#     # TODO: Check base op is in some limited class
#     num_nested = vmap_nesting(op)
#     # for loop_num in num_nested:
#     #    loop_name = f"i_{loop_num}"
#     #    with numpyro.plate()
#
#     if num_nested == 1:
#         axis_size = get_plate_size(op, *numpyro_pars)
#
#         with numpyro.plate("i",axis_size):
#             base_op_class = type(op.base_op)
#             d = simple_discrete_classes[base_op_class]
#             my_numpyro_pars = update_pars(op, *numpyro_pars)
#
#             #return numpyro.sample(name, d(*numpyro_pars), obs=obs)
#             return numpyro.sample(name, d(*my_numpyro_pars), obs=obs)
#         # TODO
#         # - start here
#         # - either implemented num_nested == 2 or just implement the general version
#         # - should probably be cleverly using vmap to move all the dims around with total reliability
#
#     else:
#         raise ValueError(f"can't handle {num_nested} nesting")


def get_numpyro_rv_discrete_latent(op, name, obs, *numpyro_pars):
    print("GET DISCRETE LATENT TRIGGERED")


    # TODO: Check base op is in some limited class
    num_nested = vmap_nesting(op)
    # for loop_num in num_nested:
    #    loop_name = f"i_{loop_num}"
    #    with numpyro.plate()

    my_numpyro_pars = vmap_numpyro_pars(op, *numpyro_pars)

    print(f"{[p.shape for p in my_numpyro_pars]=}")

    def get(op, plate_sizes, i=0):
        with numpyro.plate(f"i_{i}",plate_sizes[0]):
            op = op.base_op
            #op_class = type(op)
            if isinstance(op, ir.VMap):
                return get(op, plate_sizes[1:], i+1)
            else:
                op_class = type(op)
                d = simple_discrete_classes[op_class]
                return numpyro.sample(name, d(*my_numpyro_pars), obs=obs)

    plate_sizes = get_all_plate_sizes(op, *numpyro_pars)
    return get(op, plate_sizes)


    # if num_nested == 1:
    #     axis_size = get_plate_size(op, *numpyro_pars)
    #
    #     with numpyro.plate("i",axis_size):
    #         base_op_class = type(op.base_op)
    #         d = simple_discrete_classes[base_op_class]
    #         return numpyro.sample(name, d(*my_numpyro_pars), obs=obs)
    # elif num_nested == 2:
    #     plate_sizes = get_all_plate_sizes(op, *numpyro_pars)
    #     with numpyro.plate("i",plate_sizes[0]):
    #         op = op.base_op
    #         with numpyro.plate("j", plate_sizes[1]):
    #             base_op_class = type(op.base_op)
    #             d = simple_discrete_classes[base_op_class]
    #             return numpyro.sample(name, d(*my_numpyro_pars), obs=obs)
    # else:
    #     raise ValueError(f"can't handle {num_nested} nesting")


def get_vmap_dummy(op):
    if not isinstance(op, ir.VMap):
        def dummy(*args):
            return args

        return dummy
    else:
        dummy = get_vmap_dummy(op.base_op)
        return jax.vmap(dummy, op.in_axes, axis_size=op.axis_size)


def vmap_numpyro_pars(op: ir.VMap, *numpyro_pars):
    dummy = get_vmap_dummy(op)
    return dummy(*numpyro_pars)
