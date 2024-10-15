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
from pangolin.ir.rv import RV
from pangolin import dag, util, ir
from pangolin.interface.interface import OperatorRV
from numpy.typing import ArrayLike
from pangolin.interface import RV_or_array
from pangolin.inference import inference_util
import numpy as np

from .handler_registry import register_handler
from .handlers import get_numpyro_val
#from .model import get_numpyro_rv

@register_handler(ir.Autoregressive)
def handle_autoregressive(op, *numpyro_parents, is_observed):
    if op.random:
        return handle_autoregressive_random(op, *numpyro_parents, is_observed=is_observed)
    else:
        return handle_autoregressive_nonrandom(op, *numpyro_parents, is_observed=is_observed)


def handle_autoregressive_inputs(op: ir.Autoregressive, *numpyro_parents):
    for in_axis in op.in_axes:
        assert in_axis in [0,
                           None], "NumPyro only supports Autoregressive with in_axis of 0 or None"

    mapped_parents = tuple(p for (p, in_axis) in zip(numpyro_parents, op.in_axes) if in_axis == 0)
    unmapped_parents = tuple(p for (p, in_axis) in zip(numpyro_parents, op.in_axes) if
                             in_axis is None)

    def merge_args(mapped_args):
        ret = []
        i_mapped = 0
        i_unmapped = 0
        for in_axis in op.in_axes:
            if in_axis == 0:
                ret.append(mapped_args[i_mapped])
                i_mapped += 1
            elif in_axis is None:
                ret.append(unmapped_parents[i_unmapped])
                i_unmapped += 1
            else:
                assert False
        assert i_mapped == len(mapped_args)
        assert i_unmapped == len(unmapped_parents)
        return tuple(ret)

    return mapped_parents, merge_args


def handle_autoregressive_nonrandom(op: ir.Autoregressive, numpyro_init, *numpyro_parents, is_observed):
    # numpyro.contrib.control_flow.scan exists but seems very buggy/limited
    assert isinstance(op, ir.Autoregressive)
    assert not op.random

    mapped_parents, merge_args = handle_autoregressive_inputs(op, *numpyro_parents)
    assert merge_args(mapped_parents) == numpyro_parents

    print(f"{numpyro_init=}")

    def myfun(carry, x):
        inputs = (carry,) + merge_args(x)
        y = get_numpyro_val(op.base_op,  *inputs, is_observed=False)
        return y, y

    carry, ys = jax.lax.scan(myfun, numpyro_init, mapped_parents, length=op.length)
    return ys

# op_class_to_support = util.WriteOnceDefaultDict(
#     default_factory=lambda key: dist.constraints.real_vector
# )
# op_class_to_support[ir.Exponential] = dist.constraints.positive
# op_class_to_support[ir.Dirichlet] = dist.constraints.simplex
# op_class_to_support[ir.Bernoulli] = dist.constraints.boolean
# op_class_to_support[ir.BernoulliLogit] = dist.constraints.boolean
#
#
# def get_support(op: ir.Op):
#     """
#     Get support. Only used inside by numpyro_vmap_var_random
#
#     """
#     op_class = type(op)
#     return op_class_to_support[type(op)]
#     # if op in op_class_to_support:
#     #    return op_class_to_support[op_class]
#     # else:
#     #    raise Exception("unsupported op class")
#     # elif isinstance(op, ir.Truncated):
#     #     if op.lo is not None and op.hi is not None:
#     #         return dist.constraints.interval(op.lo, op.hi)
#     #     elif op.lo is not None:
#     #         assert op.hi is None
#     #         return dist.constraints.greater_than(op.lo)
#     #     elif op.hi is not None:
#     #         assert op.lo is None
#     #         return dist.constraints.less_than(op.hi)
#     #     else:
#     #         assert False, "should be impossible"




def handle_autoregressive_random(op: ir.Autoregressive, numpyro_init, *numpyro_parents, is_observed):
    # numpyro.contrib.control_flow.scan exists but seems very buggy/limited
    assert isinstance(op, ir.Autoregressive)
    assert op.random

    mapped_parents, merge_args = handle_autoregressive_inputs(op, *numpyro_parents)

    from .model import is_continuous
    if not is_continuous(op) and not is_observed:
        raise ValueError("Can't have non-observed autoregressive over discrete variables")

    class NewDist(dist.Distribution):  # NUMPYRO dist
        # @property
        # def support(self):
        #     return get_support(op.base_op)

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
            # assert sample_shape == (), f"sample shape is {sample_shape} expected ()"
            assert sample_shape in ((), (1,))
            single_samp = (sample_shape == (1,))

            # print(f"{sample_shape=} {single_samp=}")

            def base_sample(carry, key_and_x):
                # print(f"{key_and_x=}")
                key = key_and_x[0]
                x = key_and_x[1:]
                # inputs = (carry,) + x
                inputs = (carry,) + merge_args(x)
                var = get_numpyro_val(op.base_op, *inputs, is_observed=is_observed)
                y = var.sample(key)
                return y, y

            keys = jax.random.split(key, op.length)

            # base_sample(numpyro_init, (keys[0],) + mapped_parents)

            carry, ys = jax.lax.scan(
                base_sample, numpyro_init, (keys,) + mapped_parents, length=op.length
            )
            if single_samp:
                return jnp.array([ys])
            else:
                return ys

        @dist_util.validate_sample
        def log_prob(self, value):
            def base_log_prob(carry, val_and_x):
                val = val_and_x[0]
                x = val_and_x[1:]
                # inputs = (carry,) + x
                inputs = (carry,) + merge_args(x)
                var = get_numpyro_val(op.base_op, *inputs, is_observed=is_observed)
                return val, var.log_prob(val)

            #numpyro_init = 0.0
            print(f"{numpyro_init=}")
            print(f"{value=}")

            carry, ls = jax.lax.scan(
                base_log_prob, numpyro_init, (value,) + mapped_parents, length=op.length
            )
            return jnp.sum(ls)

    return NewDist(numpyro_init, *numpyro_parents)
