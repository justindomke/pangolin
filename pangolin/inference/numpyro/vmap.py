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
from pangolin.interface.base import OperatorRV
from numpy.typing import ArrayLike
from pangolin.interface import RV_or_ArrayLike
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

    #if isinstance(op, ir.VMap) and op.random and not is_continuous(op) and not is_observed:
    #    raise ValueError("VMap has limited support in numpy backend for discrete and non-observed")

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


# def handle_vmap_random(op: ir.VMap, *numpyro_parents, is_observed):
#     from pangolin.inference.numpyro.handlers import get_numpyro_val
#     from .support import get_support
#
#     assert isinstance(op, ir.VMap)
#     assert op.random
#
#     class NewDist(numpyro_dist.Distribution):
#         @property
#         def support(self):
#             # TODO:
#             # should be a more elegant solution here...
#             my_op = op
#             while isinstance(my_op, ir.VMap):
#                 my_op = my_op.base_op
#             return get_support(my_op)
#
#         def __init__(self, *args, validate_args=False):
#             self.args = args
#
#             # TODO: infer correct batch_shape?
#             batch_shape = ()
#             parents_shapes = [p.shape for p in args]
#             event_shape = op.get_shape(*parents_shapes)
#
#             super().__init__(
#                 batch_shape=batch_shape,
#                 event_shape=event_shape,
#                 validate_args=validate_args,
#             )
#
#         def sample(self, key, sample_shape=()):
#             assert numpyro.util.is_prng_key(key)
#             assert sample_shape == () or sample_shape == (1,)
#
#             def base_sample(key, *args):
#                 var = get_numpyro_val(op.base_op, *args, is_observed=is_observed)
#                 return var.sample(key)
#
#             keys = jax.random.split(key, self.event_shape[0])
#             in_axes = (0,) + op.in_axes
#             axis_size = op.axis_size
#             args = (keys,) + self.args
#             if sample_shape == (1,):
#                 return jnp.array([jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)])
#             elif sample_shape == ():
#                 return jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)
#             else:
#                 assert False
#
#         @dist_util.validate_sample
#         def log_prob(self, value):
#             def base_log_prob(val, *args):
#                 var = get_numpyro_val(op.base_op, *args, is_observed=is_observed)
#                 return var.log_prob(val)
#
#             in_axes = (0,) + op.in_axes
#             axis_size = op.axis_size
#             args = (value,) + self.args
#             ls = jax.vmap(base_log_prob, in_axes, axis_size=axis_size)(*args)
#             return jnp.sum(ls)
#
#     return NewDist(*numpyro_parents)

def get_eg_args(op: ir.VMap, *numpyro_parents):
    out = []
    for in_axis, arg in zip(op.in_axes, numpyro_parents, strict=True):
        if in_axis is None:
            out.append(arg)
        else:
            indices = [slice(None)]*in_axis + [0]
            out.append(arg[*indices])
    return out

def handle_vmap_random(op: ir.VMap, *numpyro_parents, is_observed):
    from pangolin.inference.numpyro.handlers import get_numpyro_val
    #from .support import get_support

    assert isinstance(op, ir.VMap)
    assert op.random

    def base_val(*args):
        return get_numpyro_val(op.base_op, *args, is_observed=is_observed)

    def base_support(*args):
        return base_val(*args).support

    _support = jax.vmap(base_support, op.in_axes, axis_size=op.axis_size)(*numpyro_parents)

    eg_args = get_eg_args(op, *numpyro_parents)
    _has_enumerate_support = get_numpyro_val(op.base_op, *eg_args, is_observed=is_observed).has_enumerate_support


    class NewDist(numpyro_dist.Distribution):
        @property
        def support(self):
            return _support

        has_enumerate_support = _has_enumerate_support

        if _has_enumerate_support:
            def enumerate_support(self, expand=True):
                print(f"{expand=}")

                def base_enumerate_support(*args):
                    return base_val(*args).enumerate_support()

                # vmap by default puts batch dims in the FRONT
                # but numpyro wants batch dims in the BACK

                return jax.vmap(
                    base_enumerate_support,
                    op.in_axes,
                    axis_size=op.axis_size,
                    out_axes=-1)(*numpyro_parents)  # out_axes=-1 = DANGER

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
                #var = get_numpyro_val(op.base_op, *args, is_observed=is_observed)
                return base_val(*args).sample(key)
                #return var.sample(key)

            keys = jax.random.split(key, self.event_shape[0])
            in_axes = (0,) + op.in_axes
            axis_size = op.axis_size
            args = (keys,) + self.args
            if sample_shape == (1,):
                return jnp.array([jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)])
            elif sample_shape == ():
                return jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)
            else:
                assert False, "Can't handle this sample shape (Pangolin bug, please report!)"

        @dist_util.validate_sample
        def log_prob(self, value):
            print(f"log prob called {value.shape=}")

            def base_log_prob(val, *args):
                #var = get_numpyro_val(op.base_op, *args, is_observed=is_observed)
                return base_val(*args).log_prob(val)
                #return var.log_prob(val)

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

# def update_pars(op, *numpyro_pars):
#     my_numpyro_pars = []
#     for par, in_axis in zip(numpyro_pars, op.in_axes):
#         if in_axis is None:
#             my_numpyro_par = par
#         elif in_axis == 0:
#             my_numpyro_par = par
#         else:
#             my_numpyro_par = jnp.moveaxis(par, in_axis, 0)
#         my_numpyro_pars.append(my_numpyro_par)
#     return my_numpyro_pars

# plate_compatible_classes = {ir.Bernoulli:numpyro_dist.Bernoulli,
#                            ir.Categorical:numpyro_dist.Categorical,
#                            ir.Binomial: numpyro_dist.Binomial,
#                            ir.BernoulliLogit: numpyro_dist.BernoulliLogits,
#                            ir.BetaBinomial: lambda n,a,b: numpyro_dist.BetaBinomial(a,b,n),
#                            ir.Multinomial: numpyro_dist.Multinomial,
#                            ir.Poisson: numpyro_dist.Poisson,
#                            ir.Normal: numpyro_dist.Normal,
#                            ir.Uniform: numpyro_dist.Uniform,
#                            }

def plate_vmap_compatible(op):
    from .handlers import simple_dists
    plate_compatible_classes = simple_dists

    if isinstance(op, ir.VMap):
        while isinstance(op, ir.VMap):
            op = op.base_op
        op_class = type(op)
        return op_class in plate_compatible_classes
    else:
        return False

def vmap_rv_plate(op, name, obs, *numpyro_pars):
    from .handlers import simple_dists
    plate_compatible_classes = simple_dists

    assert plate_vmap_compatible(op), "op {op} not compatible with vmap_rv_plate (Pangolin bug)"

    my_numpyro_pars = vmap_numpyro_pars(op, *numpyro_pars)

    def get(op, plate_sizes, i=0):
        with numpyro.plate(f"{name}_i_{i}",plate_sizes[0]):
            op = op.base_op
            if isinstance(op, ir.VMap):
                return get(op, plate_sizes[1:], i+1)
            else:
                op_class = type(op)
                if op_class not in plate_compatible_classes:
                    raise ValueError(f"op_class not (yet) supported: {op_class().name}")
                d = plate_compatible_classes[op_class]
                return numpyro.sample(name, d(*my_numpyro_pars), obs=obs)

    plate_sizes = get_all_plate_sizes(op, *numpyro_pars)
    #return get(op, plate_sizes)
    out = get(op, plate_sizes)
    assert out.shape == op.get_shape(*[p.shape for p in numpyro_pars]), "Shape match failed (Pangolin bug)"
    return out

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
