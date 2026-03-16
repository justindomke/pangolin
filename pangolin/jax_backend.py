"""
This module compiles pangolin models into plain-old JAX functions.
"""

from __future__ import annotations
from jax import numpy as jnp
import jax.tree_util
import numpy as np
from typing import Callable, Type, Sequence, Optional, Any
from pangolin import ir
from pangolin.ir import Op, RV

# from numpy.typing import ArrayLike
from numpyro import distributions as dist
from jax.scipy import special as jspecial
from jax import nn as jnn
from jax import Array as JaxArray
from pangolin import dag, util
from jaxtyping import PyTree
from jax.typing import ArrayLike

__all__ = ["ancestor_sample", "ancestor_sampler", "ancestor_log_prob"]

################################################################################
# Dict of Ops that correspond to simple functions
################################################################################

simple_funs: dict[Type[Op], Callable] = {
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
    ir.Identity: lambda a: a,
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
    ir.Cholesky: jnp.linalg.cholesky,
    ir.Transpose: jnp.transpose,
    ir.Diag: jnp.diag,
    ir.DiagMatrix: jnp.diag,
    ir.Softmax: jnn.softmax,
    ir.Index: ir.index_orthogonal_no_slices,
}

################################################################################
# Dict of Ops that correspond to simple (Numpyro) distributions
################################################################################


simple_dists: dict[Type[Op], Callable] = {
    ir.Normal: dist.Normal,
    ir.NormalPrec: lambda loc, prec: dist.Normal(loc, 1 / prec**2),
    ir.Bernoulli: dist.Bernoulli,
    ir.BernoulliLogit: dist.BernoulliLogits,
    ir.Beta: dist.Beta,
    ir.BetaBinomial: lambda n, a, b: dist.BetaBinomial(a, b, n),  # numpyro has a different order
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
    ir.Wishart: dist.Wishart,
}

################################################################################
# Handler
################################################################################

from abc import ABC, abstractmethod


class Handler(ABC):
    @abstractmethod
    def sample(self, op: Op, key: jax.Array, parent_values: Sequence[ArrayLike]) -> ArrayLike:
        pass

    @abstractmethod
    def constrained_sample(
        self, op: Op, key: jax.Array, parent_values: Sequence[ArrayLike], bijector_dict: dict
    ) -> tuple[ArrayLike, ArrayLike]:
        pass

    @abstractmethod
    def log_prob(self, op: Op, x: ArrayLike, parent_values: Sequence[ArrayLike]) -> ArrayLike:
        pass

    @abstractmethod
    def constrained_log_prob(
        self, op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict: dict
    ) -> tuple[ArrayLike, ArrayLike]:
        pass

    @abstractmethod
    def unconstrain(self, op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict: dict) -> ArrayLike:
        pass


################################################################################
# Factories that will take an Op class and give log prob and/or sample functions
################################################################################


def make_simple_eval(op_class):
    fun = simple_funs[op_class]

    def simple_eval(op, parent_values):
        return fun(*parent_values)

    return simple_eval


# def make_simple_sample(op_class: Type[Op]):
#     bind = simple_dists[op_class]

#     def simple_sample(op, key, parent_values, bijector_dict=None):
#         bound_dist: dist.Distribution = bind(*parent_values)
#         x = bound_dist.sample(key)
#         if bijector_dict is None or bijector_dict[op_class] is None:
#             return x
#         else:
#             bijector = bijector_dict[op_class](*parent_values)
#             return bijector.forward(x)

#     return simple_sample


# def make_simple_log_prob(
#     op_class: Type[Op],
# ) -> Callable[[Op, ArrayLike, Sequence[ArrayLike], dict | None], ArrayLike]:
#     bind = simple_dists[op_class]

#     # def my_log_prob(op: Op, value: ArrayLike, parent_values: Sequence[ArrayLike]) -> ArrayLike:
#     #     bound_dist: dist.Distribution = bind(*parent_values)
#     #     return bound_dist.log_prob(value)

#     def my_log_prob(op: Op, value: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict=None) -> ArrayLike:
#         # op ignored intentionally (to be homoiconic)
#         bound_dist: dist.Distribution = bind(*parent_values)
#         if bijector_dict is None or bijector_dict[op_class] is None:
#             return bound_dist.log_prob(value)
#         else:
#             bijector = bijector_dict[op_class](*parent_values)
#             x, ldj = bijector.inverse_and_log_det_jac(value)
#             return bound_dist.log_prob(x) - ldj

#     return my_log_prob


# def make_simple_constrain(
#     op_class: Type[Op],
# ) -> Callable[[Op, ArrayLike, Sequence[ArrayLike], dict], ArrayLike]:
#     def simple_constrain(op, value, parent_values, bijector_dict):
#         if bijector_dict[op_class] is None:
#             return value

#         bijector = bijector_dict[op_class](*parent_values)
#         return bijector.forward(value)

#     return simple_constrain


# def make_simple_unconstrain(
#     op_class: Type[Op],
# ) -> Callable[[Op, ArrayLike, Sequence[ArrayLike], dict], ArrayLike]:

#     def simple_unconstrain(op, value, parent_values, bijector_dict):
#         if bijector_dict[op_class] is None:
#             return value

#         bijector = bijector_dict[op_class](*parent_values)
#         return bijector.inverse(value)

#     return simple_unconstrain


class SimpleHandler(Handler):
    def __init__(self, op_class: Type[Op]):
        self.op_class = op_class
        self.bind = simple_dists[self.op_class]

    def sample(self, op: Op, key: jax.Array, parent_values: Sequence[ArrayLike]):
        bound_dist: dist.Distribution = self.bind(*parent_values)
        return bound_dist.sample(key)  # type: ignore[arg-type]

    def constrained_sample(
        self, op: Op, key: jax.Array, parent_values: Sequence[ArrayLike], bijector_dict
    ) -> tuple[ArrayLike, ArrayLike]:
        bound_dist: dist.Distribution = self.bind(*parent_values)
        bijector = bijector_dict[self.op_class](*parent_values)
        x = bound_dist.sample(key)  # type: ignore[arg-type]
        y = bijector.forward(x)
        return y, x

    def log_prob(self, op: Op, x: ArrayLike, parent_values: Sequence[ArrayLike]) -> ArrayLike:
        bound_dist: dist.Distribution = self.bind(*parent_values)
        return bound_dist.log_prob(x)

    def constrained_log_prob(
        self, op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict: dict
    ) -> tuple[ArrayLike, ArrayLike]:
        bound_dist: dist.Distribution = self.bind(*parent_values)
        bijector = bijector_dict[self.op_class](*parent_values)
        x, ldj = bijector.inverse_and_log_det_jac(y)
        l = bound_dist.log_prob(x) - ldj
        return l, x

    def unconstrain(self, op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict: dict) -> ArrayLike:
        bijector = bijector_dict[self.op_class](*parent_values)
        x = bijector.inverse(y)
        return x


################################################################################
# Basic dicts that map Op types to log_prob and/or sample handlers
################################################################################

from typing import Type

# log_prob_handlers: dict[Type[Op], Callable[[Any, ArrayLike, Sequence[ArrayLike], dict | None], Any]] = {}
# sample_handlers: dict[Type[Op], Callable[[Any, ArrayLike, Sequence[ArrayLike], dict | None], Any]] = {}
eval_handlers: dict[Type[Op], Callable[[Any, Sequence[ArrayLike]], Any]] = {}
# unconstrain_handlers: dict[Type[Op], Callable[[Any, ArrayLike, Sequence[ArrayLike], dict], Any]] = {}
handlers: dict[Type[Op], Handler] = {}

for op_class in simple_dists:
    # log_prob_handlers[op_class] = make_simple_log_prob(op_class)
    # sample_handlers[op_class] = make_simple_sample(op_class)
    # unconstrain_handlers[op_class] = make_simple_unconstrain(op_class)
    handlers[op_class] = SimpleHandler(op_class)

for op_class in simple_funs:
    eval_handlers[op_class] = make_simple_eval(op_class)


################################################################################
# Constant handler
################################################################################


def eval_constant(op: ir.Constant, parent_values: Sequence[ArrayLike]):
    return jnp.array(op.value)


eval_handlers[ir.Constant] = eval_constant

################################################################################
# Sum handler
################################################################################


def eval_sum(op: ir.Sum, parent_values: Sequence[ArrayLike]):
    assert len(parent_values) == 1
    parent_value = jnp.array(parent_values[0])
    return jnp.sum(parent_value, axis=op.axis)


eval_handlers[ir.Sum] = eval_sum

################################################################################
# Composite handler
################################################################################


def summarize_composite(op: Op, parent_values: Sequence[ArrayLike]):
    if not (isinstance(op, ir.Composite)):
        raise ValueError("Can only handle Composite")

    assert len(parent_values) == op.num_inputs
    vals = list(parent_values)
    for n, (my_op, my_par_nums) in enumerate(zip(op.ops, op.par_nums, strict=True)):
        my_parent_values = [vals[i] for i in my_par_nums]
        if n == len(op.ops) - 1:
            return my_op, my_parent_values
        new_val = eval_op(my_op, my_parent_values)
        vals.append(new_val)
    assert False, "should be impossible"


def eval_composite(op: ir.Composite, parent_values: Sequence[ArrayLike]):
    assert not op.random
    final_op, final_parent_values = summarize_composite(op, parent_values)
    return eval_op(final_op, final_parent_values)


# def log_prob_composite(op: ir.Composite, value, parent_values: Sequence[ArrayLike], bijector_dict=None):
#     assert op.random
#     final_op, final_parent_values = summarize_composite(op, parent_values)
#     return log_prob_op(final_op, value, final_parent_values, bijector_dict)


# def sample_composite(op: ir.Composite, key, parent_values: Sequence[ArrayLike], bijector_dict=None):
#     assert op.random
#     final_op, final_parent_values = summarize_composite(op, parent_values)
#     return sample_op(final_op, key, final_parent_values, bijector_dict)


# def unconstrain_composite(op: ir.Composite, value, parent_values: Sequence[ArrayLike], bijector_dict):
#     assert op.random
#     final_op, final_parent_values = summarize_composite(op, parent_values)
#     return unconstrain_op(final_op, value, final_parent_values, bijector_dict)


class CompositeHandler(Handler):
    def __init__(self):
        pass

    def sample(self, op: Op, key: jax.Array, parent_values: Sequence[ArrayLike]):
        final_op, final_parent_values = summarize_composite(op, parent_values)
        x = sample_op(final_op, key, final_parent_values)
        return x

    def constrained_sample(
        self, op: Op, key: jax.Array, parent_values: Sequence[ArrayLike], bijector_dict
    ) -> tuple[ArrayLike, ArrayLike]:
        final_op, final_parent_values = summarize_composite(op, parent_values)
        return constrained_sample_op(final_op, key, final_parent_values, bijector_dict)

    def log_prob(self, op: Op, x: ArrayLike, parent_values: Sequence[ArrayLike]) -> ArrayLike:
        final_op, final_parent_values = summarize_composite(op, parent_values)
        return log_prob_op(final_op, x, final_parent_values)

    def constrained_log_prob(
        self, op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict: dict
    ) -> tuple[ArrayLike, ArrayLike]:
        final_op, final_parent_values = summarize_composite(op, parent_values)
        return constrained_log_prob_op(final_op, y, final_parent_values, bijector_dict)

    def unconstrain(self, op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict: dict) -> ArrayLike:
        final_op, final_parent_values = summarize_composite(op, parent_values)
        return unconstrain_op(final_op, y, final_parent_values, bijector_dict)


eval_handlers[ir.Composite] = eval_composite
# log_prob_handlers[ir.Composite] = log_prob_composite
# sample_handlers[ir.Composite] = sample_composite
# unconstrain_handlers[ir.Composite] = unconstrain_composite
handlers[ir.Composite] = CompositeHandler()

################################################################################
# Autoregressive
################################################################################


def handle_scan_inputs(op: ir.Scan, *numpyro_parents):
    for in_axis in op.in_axes:
        assert in_axis in [
            0,
            None,
        ], "Jax backend only supports Scan with in_axis of 0 or None"

    mapped_parents = tuple(p for (p, in_axis) in zip(numpyro_parents, op.in_axes) if in_axis == 0)
    unmapped_parents = tuple(p for (p, in_axis) in zip(numpyro_parents, op.in_axes) if in_axis is None)

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


def summarize_scan(op: Op, parent_values):
    assert isinstance(op, ir.Scan)

    init = parent_values[0]
    rest = parent_values[1:]

    mapped_rest, merge_args = handle_scan_inputs(op, *rest)
    assert merge_args(mapped_rest) == tuple(rest)

    return init, merge_args, mapped_rest


def eval_scan(op: ir.Scan, parent_values: Sequence[ArrayLike]):
    init, merge_args, mapped_rest = summarize_scan(op, parent_values)

    def myfun(carry, x):
        inputs = (carry,) + merge_args(x)
        y = eval_op(op.base_op, inputs)
        return y, y

    carry, ys = jax.lax.scan(myfun, init, mapped_rest, length=op.length)
    return ys


eval_handlers[ir.Scan] = eval_scan


# # TODO: This should be parallel!
# def log_prob_scan(op: ir.Scan, value: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict=None):
#     if bijector_dict:
#         raise NotImplementedError()

#     assert isinstance(op, ir.Scan)
#     assert op.random

#     init = parent_values[0]
#     rest = parent_values[1:]

#     mapped_rest, merge_args = handle_scan_inputs(op, *rest)
#     assert merge_args(mapped_rest) == tuple(rest)

#     def myfun(carry, value_x):
#         value = value_x[0]
#         x = value_x[1:]
#         inputs = (carry,) + merge_args(x)
#         l = log_prob_op(op.base_op, value, inputs, bijector_dict)

#         return value, l  # pass value to next iteration

#     carry, ls = jax.lax.scan(myfun, init, (value,) + mapped_rest, length=op.length)
#     return jnp.sum(ls)
#
# log_prob_handlers[ir.Scan] = log_prob_scan


# def sample_scan(op: ir.Scan, key, parent_values: Sequence[ArrayLike], bijector_dict=None):
#     assert isinstance(op, ir.Scan)
#     assert op.random

#     init = parent_values[0]
#     rest = parent_values[1:]

#     mapped_rest, merge_args = handle_scan_inputs(op, *rest)
#     assert merge_args(mapped_rest) == tuple(rest)

#     def myfun(carry, key_x):
#         key = key_x[0]
#         x = key_x[1:]
#         inputs = (carry,) + merge_args(x)
#         y = sample_op(op.base_op, key, inputs, bijector_dict)
#         return y, y

#     subkey = jax.random.split(key, op.length)

#     carry, ys = jax.lax.scan(myfun, init, (subkey,) + mapped_rest, length=op.length)
#     return ys


# sample_handlers[ir.Scan] = sample_scan


# def unconstrain_scan(op: ir.Scan, constrained_value: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict):
#     # TODO: should be parallel
#     assert isinstance(op, ir.Scan)
#     assert op.random

#     init = parent_values[0]
#     rest = parent_values[1:]

#     mapped_rest, merge_args = handle_scan_inputs(op, *rest)
#     assert merge_args(mapped_rest) == tuple(rest)

#     def myfun(carry, value_x):
#         value = value_x[0]
#         x = value_x[1:]
#         inputs = (carry,) + merge_args(x)
#         unconstrained_value = unconstrain_op(op.base_op, value, inputs, bijector_dict)
#         return value, unconstrained_value

#     _, unconstrained_values = jax.lax.scan(myfun, init, (constrained_value,) + mapped_rest, length=op.length)
#     return unconstrained_values


# unconstrain_handlers[ir.Scan] = unconstrain_scan


class ScanHandler(Handler):
    # TODO: Not confident this is correct!

    def sample(self, op: Op, key: jax.Array, parent_values: Sequence[ArrayLike]):
        assert isinstance(op, ir.Scan)
        assert op.random
        init, merge_args, mapped_rest = summarize_scan(op, parent_values)

        def myfun(carry, key_input):
            key = key_input[0]
            input = key_input[1:]
            inputs = (carry,) + merge_args(input)
            x = sample_op(op.base_op, key, inputs)
            return x, x

        subkey = jax.random.split(key, op.length)

        carry, xs = jax.lax.scan(myfun, init, (subkey,) + mapped_rest, length=op.length)
        return xs

    def constrained_sample(
        self, op: Op, key: jax.Array, parent_values: Sequence[ArrayLike], bijector_dict
    ) -> tuple[ArrayLike, ArrayLike]:
        assert isinstance(op, ir.Scan)
        assert op.random
        init, merge_args, mapped_rest = summarize_scan(op, parent_values)

        def myfun(carry, key_input):
            key = key_input[0]
            input = key_input[1:]
            inputs = (carry,) + merge_args(input)
            y, x = constrained_sample_op(op.base_op, key, inputs, bijector_dict)
            return x, (y, x)

        subkey = jax.random.split(key, op.length)

        carry, (ys, xs) = jax.lax.scan(myfun, init, (subkey,) + mapped_rest, length=op.length)
        return (ys, xs)

    def log_prob(self, op: Op, x: ArrayLike, parent_values: Sequence[ArrayLike]) -> ArrayLike:
        assert isinstance(op, ir.Scan)
        assert op.random
        init, merge_args, mapped_rest = summarize_scan(op, parent_values)

        def myfun(carry, value_input):
            value = value_input[0]
            input = value_input[1:]
            inputs = (carry,) + merge_args(input)
            l = log_prob_op(op.base_op, value, inputs)

            return value, l  # pass value to next iteration

        carry, ls = jax.lax.scan(myfun, init, (x,) + mapped_rest, length=op.length)
        return jnp.sum(ls)

    def constrained_log_prob(
        self, op: Op, x: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict
    ) -> tuple[ArrayLike, ArrayLike]:
        assert isinstance(op, ir.Scan)
        assert op.random
        init, merge_args, mapped_rest = summarize_scan(op, parent_values)

        def myfun(carry, value_input):
            value = value_input[0]
            input = value_input[1:]
            inputs = (carry,) + merge_args(input)
            l, x = constrained_log_prob_op(op.base_op, value, inputs, bijector_dict)

            return value, (l, x)  # pass value to next iteration

        carry, (ls, xs) = jax.lax.scan(myfun, init, (x,) + mapped_rest, length=op.length)
        return jnp.sum(ls), xs

    def unconstrain(self, op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict: dict) -> ArrayLike:
        assert isinstance(op, ir.Scan)
        assert op.random
        init, merge_args, mapped_rest = summarize_scan(op, parent_values)

        def myfun(carry, value_input):
            value = value_input[0]
            input = value_input[1:]
            inputs = (carry,) + merge_args(input)
            x = unconstrain_op(op.base_op, value, inputs, bijector_dict)
            return x, x

        _, xs = jax.lax.scan(myfun, init, (y,) + mapped_rest, length=op.length)
        return xs


handlers[ir.Scan] = ScanHandler()

################################################################################
# VMap
################################################################################


def eval_vmap(op: ir.VMap, parent_values: Sequence[ArrayLike]):
    assert isinstance(op, ir.VMap)
    assert not op.random

    def base_eval(*args):
        return eval_op(op.base_op, args)

    in_axes = op.in_axes
    axis_size = op.axis_size
    return jax.vmap(base_eval, in_axes=in_axes, axis_size=axis_size)(*parent_values)


eval_handlers[ir.VMap] = eval_vmap


class VMapHandler(Handler):
    def sample(self, op: Op, key, parent_values: Sequence[ArrayLike]):
        assert isinstance(op, ir.VMap)
        assert op.random

        out_shape = op.get_shape(*[shape(p) for p in parent_values])
        out_axis_size = out_shape[0]

        subkey = jax.random.split(key, out_axis_size)

        def base_sample(key, *args):
            return sample_op(op.base_op, key, args)

        in_axes = (0,) + op.in_axes
        axis_size = op.axis_size
        return jax.vmap(base_sample, in_axes=in_axes, axis_size=axis_size)(subkey, *parent_values)

    def constrained_sample(self, op: Op, key, parent_values: Sequence[ArrayLike], bijector_dict):
        assert isinstance(op, ir.VMap)
        assert op.random

        out_shape = op.get_shape(*[shape(p) for p in parent_values])
        out_axis_size = out_shape[0]

        subkey = jax.random.split(key, out_axis_size)

        def base_sample(key, *args):
            return constrained_sample_op(op.base_op, key, args, bijector_dict)

        in_axes = (0,) + op.in_axes
        axis_size = op.axis_size
        return jax.vmap(base_sample, in_axes=in_axes, axis_size=axis_size)(subkey, *parent_values)

    def log_prob(self, op: Op, x: ArrayLike, parent_values: Sequence[ArrayLike]):
        assert isinstance(op, ir.VMap)
        assert op.random

        def base_log_prob(value, *args):
            return log_prob_op(op.base_op, value, args)

        in_axes = (0,) + op.in_axes
        axis_size = op.axis_size
        return jnp.sum(jax.vmap(base_log_prob, in_axes=in_axes, axis_size=axis_size)(x, *parent_values))

    def constrained_log_prob(self, op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict):
        assert isinstance(op, ir.VMap)
        assert op.random

        def base_log_prob(value, *args):
            return constrained_log_prob_op(op.base_op, value, args, bijector_dict)

        in_axes = (0,) + op.in_axes
        axis_size = op.axis_size
        ls, x = jax.vmap(base_log_prob, in_axes=in_axes, axis_size=axis_size)(y, *parent_values)
        return jnp.sum(ls), x

    def unconstrain(self, op: ir.Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict):
        assert isinstance(op, ir.VMap)
        assert op.random

        def base_unconstrain(value, *parent_values):
            return unconstrain_op(op.base_op, value, parent_values, bijector_dict)

        in_axes = (0,) + op.in_axes
        axis_size = op.axis_size
        return jax.vmap(base_unconstrain, in_axes=in_axes, axis_size=axis_size)(y, *parent_values)


handlers[ir.VMap] = VMapHandler()

################################################################################
# Transformed
################################################################################

# Do we really even want to allow Transformed nodes? I guess so

# def sample_transformed[O: Op, B: ir.Bijector](
#     op: ir.Transformed[O, B], key, parent_values: Sequence[ArrayLike], bijector_dict=None
# ):
#     assert isinstance(op, ir.Transformed)
#     assert op.random
#     assert op.base_op.random

#     bijector_args = tuple(parent_values[: op.bijector.n_biject_params])
#     dist_args = parent_values[op.bijector.n_biject_params :]

#     x = sample_op(op.base_op, key, dist_args)
#     y = eval_op(op.bijector.forward, (x,) + bijector_args)
#     return y


# sample_handlers[ir.Transformed] = sample_transformed


# def log_prob_transformed[O: Op, B: ir.Bijector](
#     op: ir.Transformed[O, B], y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict=None
# ):
#     assert isinstance(op, ir.Transformed)
#     assert op.random
#     assert op.base_op.random

#     bijector_args = tuple(parent_values[: op.bijector.n_biject_params])
#     dist_args = parent_values[op.bijector.n_biject_params :]

#     x = eval_op(op.bijector.inverse, (y,) + bijector_args)
#     log_px = log_prob_op(op.base_op, x, dist_args)
#     log_jac_det = eval_op(op.bijector.log_det_jac, (x, y) + bijector_args)
#     return log_px - log_jac_det


# log_prob_handlers[ir.Transformed] = log_prob_transformed

# TODO
# unconstrain_handlers[ir.Transformed] = ...

################################################################################
# Functions to do sample and/or log prob on a single node
################################################################################


def shape(value: ArrayLike):
    if isinstance(value, jnp.ndarray):
        return value.shape
    else:
        return jnp.array(value).shape


def eval_op(op: Op, parent_values: Sequence[ArrayLike]):
    """
    Given a single `Op` and parent values, draw a sample.
    """
    if op.random:
        raise ValueError("Cannot evaluate eval_op for random op")

    op_class = type(op)
    handler = eval_handlers[op_class]
    out = handler(op, parent_values)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(out) == expected_shape, "Error: shape was not as expected"
    return out


def sample_op(op: Op, key, parent_values: Sequence[ArrayLike]):
    """
    Given a single `Op` and parent values, draw a sample.
    """
    if not op.random:
        raise ValueError("Cannot evaluate sample_op for non-random op")

    # op_class = type(op)
    # handler = sample_handlers[op_class]
    # out = handler(op, key, parent_values, bijector_dict)
    # TODO: check bijected shapes
    # expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    # assert shape(out) == expected_shape, "Error: shape was not as expected"
    # return out

    op_class = type(op)
    handler = handlers[op_class]
    x = handler.sample(op, key, parent_values)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(x) == expected_shape, "Error: shape was not as expected"
    return x


def constrained_sample_op(op: Op, key, parent_values: Sequence[ArrayLike], bijector_dict):
    """
    Given a single `Op` and parent values, draw a constrained sample. Also return unconstrained value.
    """
    if not op.random:
        raise ValueError("Cannot evaluate sample_op for non-random op")

    # op_class = type(op)
    # handler = sample_handlers[op_class]
    # out = handler(op, key, parent_values, bijector_dict)
    # TODO: check bijected shapes
    # expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    # assert shape(out) == expected_shape, "Error: shape was not as expected"
    # return out

    op_class = type(op)
    handler = handlers[op_class]
    y, x = handler.constrained_sample(op, key, parent_values, bijector_dict)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(x) == expected_shape, "Error: shape was not as expected"
    return y, x


def log_prob_op(op: Op, x: ArrayLike, parent_values: Sequence[ArrayLike]):
    """
    Given a single `Op`, evaluate log_prob.
    """
    if not op.random:
        raise ValueError("Cannot evaluate log_prob_op for non-random op")
    # op_class = type(op)
    # TODO: check bijected shapes
    # expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    # if shape(value) != expected_shape:
    #    raise ValueError(f"shape(value) {shape(value)} not {expected_shape} as expected")
    # return log_prob_handlers[op_class](op, value, parent_values, bijector_dict)

    op_class = type(op)
    handler = handlers[op_class]
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(x) == expected_shape, "Error: shape was not as expected"
    l = handler.log_prob(op, x, parent_values)
    return l


def constrained_log_prob_op(op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict):
    """
    Given a single `Op`, evaluate log_prob and return unconstrained value.
    """
    if not op.random:
        raise ValueError("Cannot evaluate log_prob_op for non-random op")

    op_class = type(op)
    handler = handlers[op_class]
    l, x = handler.constrained_log_prob(op, y, parent_values, bijector_dict)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(x) == expected_shape, "Error: shape was not as expected"
    return l, x


def unconstrain_op(op: Op, constrained_value: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict: dict):
    if not op.random:
        raise ValueError("Cannot unconstrain non-random op")

    op_class = type(op)
    handler = handlers[op_class]
    value = handler.unconstrain(op, constrained_value, parent_values, bijector_dict)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    if shape(value) != expected_shape:
        raise ValueError(f"{shape(value)=} does not match {expected_shape=}")
    return value


################################################################################
# Functions to do sample and/or log prob on a "flat" graph
################################################################################

# TODO: Add conditioning?


def ancestor_sample_flat_single(vars: list[RV], key):
    all_vars = dag.upstream_nodes(vars)
    all_values = {}
    for var in all_vars:
        parent_values = [all_values[p] for p in var.parents]
        if var.op.random:
            key, subkey = jax.random.split(key)
            all_values[var] = sample_op(var.op, subkey, parent_values)
        else:
            all_values[var] = eval_op(var.op, parent_values)
    return [all_values[var] for var in vars]


def ancestor_sample_flat(vars: list[RV], key: Optional[JaxArray] = None, size: Optional[int] = None):
    if key is None:
        # Generate random seed from numpy
        seed = np.random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)

    if size is None:
        return ancestor_sample_flat_single(vars, key)
    else:
        mysample = lambda key: ancestor_sample_flat_single(vars, key)
        keys = jax.random.split(key, size)
        return jax.vmap(mysample)(keys)


def ancestor_log_prob_flat(vars: Sequence[RV], values: Sequence[ArrayLike]):
    all_vars = dag.upstream_nodes(vars)
    all_values = {var: val for var, val in zip(vars, values, strict=True)}
    l = 0.0
    for var in all_vars:
        parent_values = [all_values[p] for p in var.parents]
        if var.op.random:
            value = all_values[var]
            l += log_prob_op(var.op, value, parent_values)
        else:
            if var in vars:
                raise ValueError("Can't provide value for non-random variable in ancestor_log_prob_flat")
            out = eval_op(var.op, parent_values)
            all_values[var] = out
    return l


def ancestor_unconstrain(vars: Sequence[RV], values: Sequence[ArrayLike], bijector_dict):
    all_vars = dag.upstream_nodes(vars)
    all_values = {var: val for var, val in zip(vars, values, strict=True)}
    all_unconstrained_values = {}
    for var in all_vars:
        parent_values = [all_values[p] for p in var.parents]
        if var.op.random:
            value = all_values[var]
            all_unconstrained_values[var] = unconstrain_op(var.op, value, parent_values, bijector_dict)
        else:
            if var in vars:
                raise ValueError("Can't provide value for non-random variable in ancestor_unconstrain")
            out = eval_op(var.op, parent_values)
            all_values[var] = out
    return [all_unconstrained_values[var] for var in vars]


def fill_in(
    random_vars: Sequence[RV],
    random_values: Sequence[ArrayLike],
    desired_vars: Sequence[RV],
):
    # TODO: assert random / nonrandom / length / etc

    for var in random_vars:
        if not var.op.random:
            raise ValueError("var not random as expected")

    for var in desired_vars:
        if var.op.random and var not in random_vars:
            raise ValueError("var unexpectedly random")

    all_vars = dag.upstream_nodes(list(random_vars) + list(desired_vars))
    all_values = {}
    for var in all_vars:
        if var in random_vars:
            value = random_values[random_vars.index(var)]
        else:
            parent_values = [all_values[p] for p in var.parents]
            value = eval_op(var.op, parent_values)
        all_values[var] = value
    return [all_values[var] for var in desired_vars]


################################################################################
# Functions to do sample and/or log prob on any graph
################################################################################


def ancestor_sample(vars: PyTree[RV], key: Optional[JaxArray] = None, size: Optional[int] = None):
    """
    Draw exact samples!

    Parameters
    ----------
    vars
        a Pytree of `RV` to sample
    key
        a JAX ``PRNGKey`` or ``None`` (default)
    size
        number of samples to draw (default of ``None`` is just a single sample)

    Returns
    -------
    out
        Pytree matching structure of ``vars``, but with ``jax.ndarray`` arrays in place
        of `RV`. If ``size`` is ``None``, then each array will have the same shape as
        the corresponding `RV`. Otherwise, each array will have an extra dimension of
        size ``size`` appended at the beginning.

    Examples
    --------

    Sample a constant RV.

    >>> x = RV(ir.Constant(1.5))
    >>> ancestor_sample(x)
    Array(1.5, dtype=...)

    Sample a PyTree with the RV inside it.

    >>> ancestor_sample({'sup': [[x]]})
    {'sup': [[Array(1.5, dtype=...)]]}

    Draw several samples.

    >>> ancestor_sample(x, size=3)
    Array([1.5, 1.5, 1.5], dtype=...)

    Sample several samples from a PyTree with an RV inside it.

    >>> ancestor_sample({'sup': x}, size=3)
    {'sup': Array([1.5, 1.5, 1.5], dtype=...)}

    Sample from several random variables at once

    >>> y = RV(ir.Add(), x, x)
    >>> z = RV(ir.Mul(), x, y)
    >>> print(ancestor_sample({'cat': x, 'dog': [y, z]}))
    {'cat': Array(1.5, dtype=...), 'dog': [Array(3., dtype=...), Array(4.5, dtype=...)]}
    """

    if key is None:
        # Generate random seed from numpy
        seed = np.random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)

    (
        flat_vars,
        _,
        _,
        unflatten,
        _,
    ) = util.flatten_args(vars, [], [])

    flat_samps = ancestor_sample_flat(flat_vars, key, size)

    return unflatten(flat_samps)


def ancestor_sampler(vars: PyTree[RV]) -> Callable[[JaxArray], Any]:
    """
    Compiles a pytree of RVs into a plain-old JAX function that takes a PNGKey and returns a pytree with the same structure containing a joint sample from the distribution of those RVs.

    Parameters
    ----------
    vars
        Pytree of `RV` to sample

    Returns
    -------
    out
        function mapping a JAX ``PRNGKey`` to a sample in the form of a pytree of
        ``jax.ndarray`` matching the structure and shape of ``vars``


    Examples
    --------
    >>> x = RV(ir.Constant(1.5))
    >>> y = RV(ir.Add(), x, x)
    >>> fun = ancestor_sampler([{'cat': x}, y])

    You now have a plain-old JAX function that's completely independent of pangolin.

    >>> key = jax.random.PRNGKey(0)
    >>> fun(key)
    [{'cat': Array(1.5, dtype=float32)}, Array(3., dtype=float32)]

    You can do normal JAX stuff with it, e.g. vmap it.

    >>> print(jax.vmap(fun)(jax.random.split(key, 3)))
    [{'cat': Array([1.5, 1.5, 1.5], dtype=float32)}, Array([3., 3., 3.], dtype=float32)]

    """

    (
        flat_vars,
        _,
        _,
        unflatten,
        _,
    ) = util.flatten_args(vars, [], [])

    def sampler(key):
        flat_samps = ancestor_sample_flat(flat_vars, key)

        return unflatten(flat_samps)

    return sampler


def ancestor_log_prob(*vars: PyTree[RV], **kwvars: PyTree[RV]) -> Callable:
    """
    Given a pytree of vars, create a plain-old JAX function to compute log probabilities

    Parameters
    ----------
    vars
        pytrees of `RV`
    kwargs
        more pytrees of `RV`, given as keyword arguments

    Returns
    -------
    out
        log-prob function that expects ``jax.ndarray`` arguments matching ``vars`` and
        ``kwargs`` and returning a scalar.


    Examples
    --------
    >>> loc = ir.RV(ir.Constant(0.0))
    >>> scale = ir.RV(ir.Constant(1.0))
    >>> x = RV(ir.Normal(),loc,scale)
    >>> fun = ancestor_log_prob(x)

    You now have a plain JAX function that's completely independent of pangolin. You can evaluate it.

    >>> fun(0.0)
    Array(-0.9189385, dtype=...)

    Or you can vmap it.

    >>> jax.vmap(fun)(jnp.array([0.0, 0.5]))
    Array([-0.9189385, -1.0439385], dtype=float32)

    Here's a more complex example:

    >>> op = ir.VMap(ir.Normal(), [None,None], 3)
    >>> y = RV(op,loc,scale)
    >>> fun = ancestor_log_prob({'x':x, 'y':y})
    >>> fun({'x':0.0, 'y':[0.0, 0.5, 0.1]})
    Array(-3.8057542, dtype=...)

    You can also create a function that uses positional and/or keyword arguments:

    >>> fun = ancestor_log_prob(x, cat=y)
    >>> fun(0.0, cat=[0.0, 0.5, 0.1])
    Array(-3.8057542, dtype=...)
    """

    all_vars = (vars, kwvars)

    def myfun(*vals, **kwvals):
        all_vals = (vals, kwvals)

        all_vals = util.assimilate_vals(all_vars, all_vals)  # casts lists and such to ndarray

        flat_vars, vars_treedef = jax.tree_util.tree_flatten(all_vars)
        flat_vals, vals_treedef = jax.tree_util.tree_flatten(all_vals)
        if vars_treedef != vals_treedef:
            raise ValueError("vars_treedef does not match vals_treedef")

        return ancestor_log_prob_flat(flat_vars, flat_vals)

    return myfun


################################################################################
# Bijections
################################################################################

# class JaxBijector(ABC):
#     """
#     the idea is that if ``P(X)`` is some density and ``Y=T(X)`` is a diffeomorphism, then ``P(Y=y) = P(X=T⁻¹(y)) × |det ∇T⁻¹(y)|``
#     """

#     @abstractmethod
#     def forward(self, x):
#         pass

#     @abstractmethod
#     def inverse(self, y):
#         pass

#     @abstractmethod
#     def log_jac_det(self, x, y):
#         pass


# class JaxBijector:
#     """
#     the idea is that if ``P(X)`` is some density and ``Y=T(X)`` is a diffeomorphism, then ``P(Y=y) = P(X=T⁻¹(y)) × |det ∇T⁻¹(y)|``
#     """

#     def __init__(self, forward, inverse, log_det_jac, n_biject_params=0):
#         self._forward = forward
#         self._inverse = inverse
#         self._log_det_jac = log_det_jac
#         self.n_biject_params = n_biject_params

#     def forward(self, x, *params):
#         if len(params) != self.n_biject_params:
#             raise ValueError(f"{len(params)=} not equal to {self.n_biject_params=}")
#         return self._forward(x, *params)

#     def inverse(self, y, *params):
#         if len(params) != self.n_biject_params:
#             raise ValueError(f"{len(params)=} not equal to {self.n_biject_params=}")
#         return self._inverse(y, *params)

#     def log_det_jac(self, x, y, *params):
#         if len(params) != self.n_biject_params:
#             raise ValueError(f"{len(params)=} not equal to {self.n_biject_params=}")
#         return self._log_det_jac(x, y, *params)

#     def forward_and_log_det_jac(self, x, *params):
#         if len(params) != self.n_biject_params:
#             raise ValueError(f"{len(params)=} not equal to {self.n_biject_params=}")
#         y = self.forward(x, *params)
#         ldj = self.log_det_jac(x, y, *params)
#         return y, ldj

#     def inverse_and_log_det_jac(self, y, *params):
#         if len(params) != self.n_biject_params:
#             raise ValueError(f"{len(params)=} not equal to {self.n_biject_params=}")
#         x = self.inverse(y, *params)
#         ldj = self.log_det_jac(x, y, *params)
#         return x, ldj

#     @property
#     def reverse(self):
#         return JaxBijector(
#             self.inverse, self.forward, lambda y, x, *params: -self.log_det_jac(x, y, *params), self.n_biject_params
#         )


class JaxBijector:
    """
    the idea is that if ``P(X)`` is some density and ``Y=T(X)`` is a diffeomorphism, then ``P(Y=y) = P(X=T⁻¹(y)) × |det ∇T⁻¹(y)|``
    """

    def __init__(self, forward, inverse, log_det_jac):
        self._forward = forward
        self._inverse = inverse
        self._log_det_jac = log_det_jac

    def forward(self, x):
        return self._forward(x)

    def inverse(self, y):
        return self._inverse(y)

    def log_det_jac(self, x, y):
        return self._log_det_jac(x, y)

    def forward_and_log_det_jac(self, x):
        y = self.forward(x)
        ldj = self.log_det_jac(x, y)
        return y, ldj

    def inverse_and_log_det_jac(self, y):
        x = self.inverse(y)
        ldj = self.log_det_jac(x, y)
        return x, ldj

    @property
    def reverse(self):
        return JaxBijector(self.inverse, self.forward, lambda y, x: -self.log_det_jac(x, y))


def compose_jax_bijectors(bijectors: Sequence[JaxBijector], log_det_direction: str = "forward") -> JaxBijector:
    bijectors = tuple(bijectors)

    def composed_forward(x):
        current_x = x
        for b in bijectors:
            current_x = b.forward(current_x)
        return current_x

    def composed_inverse(y):
        current_y = y
        for b in reversed(bijectors):
            current_y = b.inverse(current_y)
        return current_y

    def _log_det_forward(x, y):
        log_det_sum = 0.0
        current_x = x
        for b in bijectors:
            next_x = b.forward(current_x)
            log_det_sum += b.log_det_jac(current_x, next_x)
            current_x = next_x

        return log_det_sum

    def _log_det_inverse(x, y):
        log_det_sum = 0.0
        current_y = y
        for b in reversed(bijectors):
            previous_x = b.inverse(current_y)
            log_det_sum += b.log_det_jac(previous_x, current_y)
            current_y = previous_x
        return log_det_sum

    if log_det_direction == "forward":
        composed_log_det_jac = _log_det_forward
    elif log_det_direction == "inverse":
        composed_log_det_jac = _log_det_inverse
    else:
        raise ValueError("log_det_direction must be 'forward' or 'inverse'")

    return JaxBijector(composed_forward, composed_inverse, composed_log_det_jac)


########################################################################################
# Library of specific transforms
########################################################################################


def _cholesky_log_det_jac(X, Y):
    """Logarithm of the absolute determinant of the Cholesky mapping Jacobian.

    The Cholesky decomposition maps a symmetric positive-definite matrix $X$
    to a lower triangular matrix $Y$ with positive diagonal elements such that
    $X = YY^T$. This function computes the log-determinant of the Jacobian
    of the forward transformation (from SPD matrix to Cholesky factor).

    The formula implemented is:
    $$ \log |\det J| = -k \log(2) - \sum_{i=1}^{k} (k-i+1) \log(Y_{ii}) $$

    Args:
        X: Original symmetric positive-definite matrix of shape $(k, k)$ (unused
           in computation but kept for API consistency).
        Y: Cholesky factor, a lower triangular matrix of shape $(k, k)$ with
           positive diagonal elements.

    Returns:
        Scalar array containing $\log |\det J|$ where $J$ is the Jacobian of
        the Cholesky mapping.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[4., 2.], [2., 2.]])
        >>> Y = jnp.linalg.cholesky(X)
        >>> _cholesky_log_det_jac(X, Y)  # doctest: +ELLIPSIS
        Array(-2.7725887..., dtype=float32)

        >>> X = jnp.array([[1., 0., 0.],
        ...                [0., 1., 0.],
        ...                [0., 0., 1.]])
        >>> Y = jnp.linalg.cholesky(X)
        >>> _cholesky_log_det_jac(X, Y)
        Array(-2.0794415, dtype=float32)
    """
    k = Y.shape[0]
    # Match the dtype of Y to prevent implicit type upcasting
    powers = jnp.arange(k, 0, -1, dtype=Y.dtype)
    return -k * jnp.log(2.0) - powers @ jnp.log(jnp.diag(Y))


def _fill_tril(x: jax.Array) -> jax.Array:
    """Fill a lower triangular matrix from a packed 1D vector.

    This is the inverse operation of `_extract_tril`. Reconstructs a
    lower triangular matrix from its packed representation.

    Args:
        x: 1D array of packed lower triangular elements.

    Returns:
        Lower triangular matrix of shape $(n, n)$ reconstructed from the
        packed vector, where $n$ is determined from the length of x.

    Example:
        >>> x = jnp.array([1., 2., 3., 4., 5., 6.])
        >>> _fill_tril(x)
        Array([[1., 0., 0.],
               [2., 3., 0.],
               [4., 5., 6.]], dtype=float32)

        >>> _fill_tril(jnp.array([1., 3., 4.]))
        Array([[1., 0.],
               [3., 4.]], dtype=float32)
    """
    x = jnp.asarray(x)

    if x.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape {x.shape}")

    m = x.shape[0]

    # Use standard Python math for static shape calculations to prevent JIT Concretization errors
    n = int(((8 * m + 1) ** 0.5 - 1) / 2)

    if n * (n + 1) // 2 != m:
        raise ValueError(f"Length {m} is not a triangular number n*(n+1)/2. Valid lengths: 1, 3, 6, 10, 15, ...")

    out = jnp.zeros((n, n), dtype=x.dtype)
    return out.at[jnp.tril_indices(n)].set(x)


def _extract_tril(X: jax.Array) -> jax.Array:
    """Extract lower triangular elements into a packed 1D vector.

    Packs the elements of the lower triangle of a square matrix into a
    1D array using row-major ordering (C-style).

    Args:
        X: Square matrix of shape $(n, n)$.

    Returns:
        1D array containing the packed lower triangular elements, including
        the diagonal. Length is $n(n+1)/2$.

    Example:
        >>> X = jnp.array([[1., 0., 0.],
        ...                [2., 3., 0.],
        ...                [4., 5., 6.]])
        >>> _extract_tril(X)
        Array([1., 2., 3., 4., 5., 6.], dtype=float32)

        >>> _extract_tril(jnp.array([[1., 2.], [3., 4.]]))
        Array([1., 3., 4.], dtype=float32)
    """
    X = jnp.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {X.shape}")

    if X.shape[0] != X.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {X.shape}")

    return X[jnp.tril_indices(X.shape[0])]


def _exp_diagonal(X: jax.Array):
    """
    Exponentiate diagonal elements: Y_ii = exp(X_ii), Y_ij = X_ij for i≠j.

    This is a bijection on square matrices that leaves off-diagonal
    elements unchanged and maps the diagonal through exp().

    Args:
        X: Square matrix of shape (n, n)

    Returns:
        Matrix of same shape with exponentiated diagonal

    Raises:
        ValueError: If X is not a 2D square matrix.

    Example:
        >>> X = jnp.array([[1., 2.], [3., 4.]])
        >>> _exp_diagonal(X) # doctest: +ELLIPSIS
        Array([[ 2.718...,  2.       ],
               [ 3.       , 54.598... ]], dtype=float32)
    """
    X = jnp.asarray(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {X.shape}")

    x = jnp.diag(X)
    # X + diag(exp(x) - x) preserves off-diagonal and sets diagonal to exp(x)
    return X + jnp.diag(jnp.exp(x) - x)


def _log_diagonal(X: jax.Array):
    """
    Log-transform diagonal elements: Y_ii = log(X_ii), Y_ij = X_ij for i≠j.

    Inverse of _exp_diagonal. Requires strictly positive diagonal entries
    (otherwise returns NaN/Inf).

    Args:
        X: Square matrix with positive diagonal

    Returns:
        Matrix of same shape with log-transformed diagonal

    Raises:
        ValueError: If X is not a 2D square matrix.

    Example:
        >>> X = jnp.array([[1., 2.], [3., 4.]])
        >>> _log_diagonal(X) # doctest: +ELLIPSIS
        Array([[0.       , 2.       ],
               [3.       , 1.386...]], dtype=float32)
    """
    X = jnp.asarray(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {X.shape}")

    x = jnp.diag(X)
    return X + jnp.diag(jnp.log(x) - x)


def _exp_diagonal_log_det_jac(X: jax.Array, Y: jax.Array):
    """
    Log determinant of Jacobian for _exp_diagonal.

    For Y = _exp_diagonal(X), the Jacobian is diagonal with:
    - dY_ii/dX_ii = exp(X_ii)  (n terms)
    - dY_ij/dX_ij = 1        for i≠j (n²-n terms)

    det(J) = ∏ exp(X_ii) = exp(∑ X_ii)
    log det(J) = ∑ X_ii

    Args:
        X: Input square matrix (n, n)
        Y: Output matrix (unused, kept for JaxBijector API consistency)

    Returns:
        Scalar log-determinant (sum of diagonal of X)

    Example:
        >>> X = jnp.array([[1., 2.], [3., 4.]])
        >>> Y = _exp_diagonal(X)
        >>> _exp_diagonal_log_det_jac(X, Y)  # 1 + 4 = 5
        Array(5., dtype=float32)

        >>> # Verify: log(det) = log(exp(1)*exp(4)) = 5
        >>> float(_exp_diagonal_log_det_jac(X, Y)) == 5.0
        True
    """
    X = jnp.asarray(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {X.shape}")

    return jnp.trace(X)


def exp_bijector():
    """
    Creates a `JaxBijector` instance that applies the exponential function.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array(0.0)
        >>> exp_bijector().forward(x)  # doctest: +ELLIPSIS
        Array(1., dtype=float32...)
    """
    # f(x) = exp(x)  <==>  df/dx = exp(x) = y  <==>  log df/dx = log(y) = x
    return JaxBijector(jnp.exp, jnp.log, lambda x, y: x)


def log_bijector():
    """
    Creates a `JaxBijector` instance that applies the natural logarithm.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array(1.0)
        >>> log_bijector().forward(x)  # doctest: +ELLIPSIS
        Array(0., dtype=float32...)
    """

    # f(x) = log(x) <==> df/dx = 1/x <==> log df/dx = -log(x) = -y
    return JaxBijector(jnp.log, jnp.exp, lambda x, y: -y)


def logit_bijector():
    """
    Creates a `JaxBijector` instance that applies the logit bijector ``y = logit(x)``. Commonly used to transform from [0,1] to reals.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array(0.5)
        >>> logit_bijector().forward(x)  # doctest: +ELLIPSIS
        Array(0., dtype=float32...)
    """
    return JaxBijector(jax.scipy.special.logit, jax.scipy.special.expit, lambda x, y: -jnp.log(x) - jnp.log1p(-x))


def inv_logit_bijector():
    """
    Creates a `JaxBijector` instance that applies the inverse logit (expit/sigmoid).

    Example:
        >>> import jax.numpy as jnp
        >>> y = jnp.array(0.0)
        >>> inv_logit_bijector().forward(y)  # doctest: +ELLIPSIS
        Array(0.5, dtype=float32...)
    """

    return logit_bijector().reverse


def scaled_logit_bijector(a, b):
    """
    Creates a `JaxBijector` instance that applies the scaled logit ``y = logit((x-a)/(b-a))``. Commonly used to transform from [a,b] to reals.
    """
    return JaxBijector(
        lambda x: jax.scipy.special.logit((x - a) / (b - a)),
        lambda y: a + (b - a) * jax.scipy.special.expit(y),
        lambda x, y: jnp.log(b - a) - jnp.log(x - a) - jnp.log(b - x),
    )


def cholesky_bijector():
    """
    Creates a `JaxBijector` instance that applies a Cholesky decomposition. Commonly used to transform from symmetric positive definite matrices into triangular matrices.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[1., 0.], [0., 1.]])
        >>> cholesky_bijector().forward(X)
        Array([[1., 0.],
               [0., 1.]], dtype=float32)
    """
    return JaxBijector(
        lambda X: jnp.linalg.cholesky(X),
        lambda Y: Y @ Y.T,
        _cholesky_log_det_jac,
    )


def fill_tril_bijector():
    """
    A `JaxBijector` instance that fills a lower-triangular matrix from a vector. Used to transform from real vectors to lower-triangular matrices.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1., 2., 3.])
        >>> fill_tril_bijector().forward(x)
        Array([[1., 0.],
               [2., 3.]], dtype=float32)
    """
    return JaxBijector(_fill_tril, _extract_tril, lambda x, y: jnp.array(0.0))


def extract_tril_bijector():
    """
    A `JaxBijector` instance that extracts the lower-triangular part of a matrix. Commonly used to transform from triangular lower-triangular matrices to real vectors.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[1., 0.], [2., 3.]])
        >>> extract_tril_bijector().forward(X)
        Array([1., 2., 3.], dtype=float32)
    """
    # return fill_tril_bijector().reverse
    return JaxBijector(_extract_tril, _fill_tril, lambda x, y: jnp.array(0.0))


def exp_diagonal_bijector():
    """
    A `JaxBijector` instance that exponentiates the diagonal of a matrix. Commonly used to transform real lower-triangular matrices into Cholesky factors.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[0., 0.], [2., 0.]])
        >>> exp_diagonal_bijector().forward(X)
        Array([[1., 0.],
               [2., 1.]], dtype=float32)
    """
    return JaxBijector(_exp_diagonal, _log_diagonal, _exp_diagonal_log_det_jac)


def log_diagonal_bijector():
    """
    A `JaxBijector` instance that takes the logarithm of the diagonal of a matrix. Commonly used to transform real lower-triangular matrices into Cholesky factors.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[1., 0.], [2., 1.]])
        >>> log_diagonal_bijector().forward(X)
        Array([[0., 0.],
               [2., 0.]], dtype=float32)
    """
    return exp_diagonal_bijector().reverse


def unconstrain_spd_bijector():
    """
    Creates a `JaxBijector` instance that transforms a symmetric positive definite into the space of unconstrained reals. Accomplished by (1) taking a Cholesky decomposition (2) taking the logarithm of the diagonal (3) extracting the lower-triangular entries.

    Example:
        >>> import jax.numpy as jnp
        >>> # Identity matrix is symmetric positive definite
        >>> X = jnp.array([[1., 0.], [0., 1.]])
        >>> unconstrain_spd_bijector().forward(X)
        Array([0., 0., 0.], dtype=float32)

        >>> # Transform back to SPD matrix
        >>> unconstrained_vec = jnp.array([0., 2., 0.])
        >>> unconstrain_spd_bijector().inverse(unconstrained_vec)
        Array([[1., 2.],
               [2., 5.]], dtype=float32)
    """
    return compose_jax_bijectors([cholesky_bijector(), log_diagonal_bijector(), extract_tril_bijector()])


class bijectors:
    """
    A namespace containing a bunch of pre-baked `JaxBijector` instances.
    """

    # f(x) = exp(x)  <==>  df/dx = exp(x) = y  <==>  log df/dx = log(y) = x
    exp = JaxBijector(jnp.exp, jnp.log, lambda x, y: x)
    """
    A `JaxBijector` instance that applies the exponential function.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array(0.0)
        >>> bijectors.exp.forward(x)  # doctest: +ELLIPSIS
        Array(1., dtype=float32...)
    """

    # f(x) = log(x) <==> df/dx = 1/x <==> log df/dx = -log(x) = -y
    log = JaxBijector(jnp.log, jnp.exp, lambda x, y: -y)
    """
    A `JaxBijector` instance that applies the natural logarithm.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array(1.0)
        >>> bijectors.log.forward(x)  # doctest: +ELLIPSIS
        Array(0., dtype=float32...)
    """

    # logit = JaxBijector(jax.scipy.special.logit, jax.scipy.special.expit, lambda x, y: -jnp.log(x) - jnp.log1p(-x))
    # """
    # A `Transform` instance that applies the logit bijector ``y = logit(x)``. Commonly used to transform from [0,1] to reals.

    # Example:
    #     >>> import jax.numpy as jnp
    #     >>> x = jnp.array(0.5)
    #     >>> bijectors.logit.forward(x)  # doctest: +ELLIPSIS
    #     Array(0., dtype=float32...)
    # """

    # inv_logit = logit.reverse
    # """
    # A `JaxBijector` instance that applies the inverse logit (expit/sigmoid).

    # Example:
    #     >>> import jax.numpy as jnp
    #     >>> y = jnp.array(0.0)
    #     >>> bijectors.inv_logit.forward(y)  # doctest: +ELLIPSIS
    #     Array(0.5, dtype=float32...)
    # """

    # scaled_logit = JaxBijector(
    #     lambda x, a, b: jax.scipy.special.logit((x - a) / (b - a)),
    #     lambda y, a, b: a + (b - a) * jax.scipy.special.expit(y),
    #     lambda x, y, a, b: jnp.log(b - a) - jnp.log(x - a) - jnp.log(b - x),
    #     n_biject_params=2,
    # )
    # """
    # A `JaxBijector` instance that applies the scaled logit ``y = logit((x-a)/(b-a))``. Commonly used to transform from [a,b] to reals.

    # Example:
    #     >>> import jax.numpy as jnp
    #     >>> x = jnp.array(5.0)
    #     >>> # Transform 5.0 from the bounds [0.0, 10.0] to unconstrained space
    #     >>> bijectors.scaled_logit.forward(x, 0.0, 10.0)  # doctest: +ELLIPSIS
    #     Array(0., dtype=float32...)
    #     >>> bijectors.scaled_logit.inverse(jnp.array(0.0), 0.0, 10.0)  # doctest: +ELLIPSIS
    #     Array(5., dtype=float32...)
    # """

    # cholesky = JaxBijector(
    #     lambda X: jnp.linalg.cholesky(X),
    #     lambda Y: Y @ Y.T,
    #     _cholesky_log_det_jac,
    # )
    # """
    # A `JaxBijector` instance that applies a Cholesky decomposition. Commonly used to transform from symmetric positive definite matrices into triangular matrices.

    # Example:
    #     >>> import jax.numpy as jnp
    #     >>> X = jnp.array([[1., 0.], [0., 1.]])
    #     >>> bijectors.cholesky.forward(X)
    #     Array([[1., 0.],
    #            [0., 1.]], dtype=float32)
    # """

    # fill_tril = JaxBijector(_fill_tril, _extract_tril, lambda x, y: 0.0)
    # """
    # A `JaxBijector` instance that fills a lower-triangular matrix from a vector. Used to transform from real vectors to lower-triangular matrices.

    # Example:
    #     >>> import jax.numpy as jnp
    #     >>> x = jnp.array([1., 2., 3.])
    #     >>> bijectors.fill_tril.forward(x)
    #     Array([[1., 0.],
    #            [2., 3.]], dtype=float32)
    # """

    # extract_tril = fill_tril.reverse
    # """
    # A `JaxBijector` instance that extracts the lower-triangular part of a matrix. Commonly used to transform from triangular lower-triangular matrices to real vectors.

    # Example:
    #     >>> import jax.numpy as jnp
    #     >>> X = jnp.array([[1., 0.], [2., 3.]])
    #     >>> bijectors.extract_tril.forward(X)
    #     Array([1., 2., 3.], dtype=float32)
    # """

    # exp_diagonal = JaxBijector(_exp_diagonal, _log_diagonal, _exp_diagonal_log_det_jac)
    # """
    # A `JaxBijector` instance that exponentiates the diagonal of a matrix. Commonly used to transform real lower-triangular matrices into Cholesky factors.

    # Example:
    #     >>> import jax.numpy as jnp
    #     >>> X = jnp.array([[0., 0.], [2., 0.]])
    #     >>> bijectors.exp_diagonal.forward(X)
    #     Array([[1., 0.],
    #            [2., 1.]], dtype=float32)
    # """

    # log_diagonal = exp_diagonal.reverse
    # """
    # A `JaxBijector` instance that takes the logarithm of the diagonal of a matrix. Commonly used to transform real lower-triangular matrices into Cholesky factors.

    # Example:
    #     >>> import jax.numpy as jnp
    #     >>> X = jnp.array([[1., 0.], [2., 1.]])
    #     >>> bijectors.log_diagonal.forward(X)
    #     Array([[0., 0.],
    #            [2., 0.]], dtype=float32)
    # """

    # unconstrain_spd = compose_jax_bijectors([cholesky, log_diagonal, extract_tril])
    # """
    # A `JaxBijector` instance that transforms a symmetric positive definite into the space of unconstrained reals. Accomplished by (1) taking a Cholesky decomposition (2) taking the logarithm of the diagonal (3) extracting the lower-triangular entries.

    # Example:
    #     >>> import jax.numpy as jnp
    #     >>> # Identity matrix is symmetric positive definite
    #     >>> X = jnp.array([[1., 0.], [0., 1.]])
    #     >>> bijectors.unconstrain_spd.forward(X)
    #     Array([0., 0., 0.], dtype=float32)

    #     >>> # Transform back to SPD matrix
    #     >>> unconstrained_vec = jnp.array([0., 2., 0.])
    #     >>> bijectors.unconstrain_spd.inverse(unconstrained_vec)
    #     Array([[1., 2.],
    #            [2., 5.]], dtype=float32)
    # """

    # def __init__(self):
    #     raise TypeError("Use bijectors as a static namespace, do not instantiate.")


class BijectorFactory:
    def __init__(self):
        self.bijector_dict = {
            ir.Normal: None,
            ir.NormalPrec: None,
            ir.Cauchy: None,
            ir.Exponential: lambda a: log_bijector(),
            ir.Gamma: lambda a: log_bijector(),
            ir.StudentT: None,
            ir.MultiNormal: None,
            # ir.Dirichlet: lambda a, b: raise NotImplementedError(),
            ir.Lognormal: lambda a, b: log_bijector(),
            ir.Uniform: lambda a, b: scaled_logit_bijector(a, b),
            ir.Wishart: lambda a, b: unconstrain_spd_bijector(),
            ir.Beta: lambda a, b: logit_bijector(),
        }

    def get_bijector(self, op, *parent_values):

        if not op.random:
            return None

        op_class = type(op)
        parent_values = tuple(parent_values)

        if op_class in self.bijector_dict:
            return self.bijector_dict[op_class](*parent_values)
