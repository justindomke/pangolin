"""
This module compiles pangolin models into plain-old JAX functions.

These methods allow an optional ``biject`` argument. If ``True``, then constrained continous distributions will be automatically transformed to an unconstrained space. If ``False``, then this will not be done. One can also provide a dictionary to precisely control how distributions are transformed. See the ``bijectors`` submodule for more information.
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
from . import bijectors

__all__ = ["ancestor_sample", "ancestor_sampler", "ancestor_log_prob", "bijectors"]

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
    def sample(self, op: Op, key: JaxArray, parent_values: Sequence[JaxArray]) -> JaxArray:
        pass

    @abstractmethod
    def unconstrained_sample(
        self, op: Op, key: JaxArray, parent_values: Sequence[JaxArray], bijector_dict: dict
    ) -> tuple[JaxArray, JaxArray]:
        pass

    @abstractmethod
    def log_prob(self, op: Op, x: JaxArray, parent_values: Sequence[JaxArray]) -> JaxArray:
        pass

    @abstractmethod
    def unconstrained_log_prob(
        self, op: Op, y: JaxArray, parent_values: Sequence[JaxArray], bijector_dict: dict
    ) -> tuple[JaxArray, JaxArray]:
        pass

    @abstractmethod
    def constrain(self, op: Op, y: JaxArray, parent_values: Sequence[JaxArray], bijector_dict: dict) -> JaxArray:
        pass


################################################################################
# Factories that will take an Op class and give log prob and/or sample functions
################################################################################


def make_simple_eval(op_class):
    fun = simple_funs[op_class]

    def simple_eval(op, parent_values):
        return fun(*parent_values)

    return simple_eval


class SimpleHandler(Handler):
    def __init__(self, op_class: Type[Op]):
        self.op_class = op_class
        self.bind = simple_dists[self.op_class]

    def sample(self, op, key, parent_values):
        bound_dist: dist.Distribution = self.bind(*parent_values)
        x: JaxArray = bound_dist.sample(key)  # type: ignore[arg-type]
        return x

    def unconstrained_sample(self, op, key, parent_values, bijector_dict):
        bound_dist: dist.Distribution = self.bind(*parent_values)
        x: JaxArray = bound_dist.sample(key)  # type: ignore[arg-type]
        if op.discrete or bijector_dict[self.op_class] is None:
            y = x
        else:
            bijector = bijector_dict[self.op_class](*parent_values)
            y = bijector.forward(x)
        return y, x

    def log_prob(self, op, x, parent_values):
        bound_dist: dist.Distribution = self.bind(*parent_values)
        l: JaxArray = bound_dist.log_prob(x)  # type: ignore
        return l

    def unconstrained_log_prob(self, op, y, parent_values, bijector_dict):
        bound_dist: dist.Distribution = self.bind(*parent_values)
        if op.discrete or bijector_dict[self.op_class] is None:
            x = y
            l: JaxArray = bound_dist.log_prob(x)  # type: ignore
        else:
            bijector = bijector_dict[self.op_class](*parent_values)
            x, ldj = bijector.inverse_and_log_det_jac(y)
            l = bound_dist.log_prob(x) - ldj
        return l, x

    def constrain(self, op, y, parent_values, bijector_dict):
        if bijector_dict[self.op_class] is None:
            x = y
        else:
            bijector = bijector_dict[self.op_class](*parent_values)
            x = bijector.inverse(y)
        return x


################################################################################
# Basic dicts that map Op types to log_prob and/or sample handlers
################################################################################

from typing import Type

eval_handlers: dict[Type[Op], Callable[[Any, Sequence[JaxArray]], Any]] = {}
handlers: dict[Type[Op], Handler] = {}

for op_class in simple_dists:
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


def summarize_composite(op: Op, parent_values: Sequence[JaxArray]):
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


def eval_composite(op: ir.Composite, parent_values: Sequence[JaxArray]):
    assert not op.random
    final_op, final_parent_values = summarize_composite(op, parent_values)
    return eval_op(final_op, final_parent_values)


class CompositeHandler(Handler):
    def __init__(self):
        pass

    def sample(self, op, key, parent_values):
        final_op, final_parent_values = summarize_composite(op, parent_values)
        x = sample_op(final_op, key, final_parent_values)
        return x

    def unconstrained_sample(self, op, key, parent_values, bijector_dict):
        final_op, final_parent_values = summarize_composite(op, parent_values)
        return unconstrained_sample_op(final_op, key, final_parent_values, bijector_dict)

    def log_prob(self, op, x, parent_values):
        final_op, final_parent_values = summarize_composite(op, parent_values)
        return log_prob_op(final_op, x, final_parent_values)

    def unconstrained_log_prob(self, op, y, parent_values, bijector_dict):
        final_op, final_parent_values = summarize_composite(op, parent_values)
        return unconstrained_log_prob_op(final_op, y, final_parent_values, bijector_dict)

    def constrain(self, op, y, parent_values, bijector_dict):
        final_op, final_parent_values = summarize_composite(op, parent_values)
        return constrain_op(final_op, y, final_parent_values, bijector_dict)


eval_handlers[ir.Composite] = eval_composite
handlers[ir.Composite] = CompositeHandler()

################################################################################
# Scan
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

    def unconstrained_sample(self, op, key, parent_values, bijector_dict):
        assert isinstance(op, ir.Scan)
        assert op.random
        init, merge_args, mapped_rest = summarize_scan(op, parent_values)

        def myfun(carry, key_input):
            key = key_input[0]
            input = key_input[1:]
            inputs = (carry,) + merge_args(input)
            y, x = unconstrained_sample_op(op.base_op, key, inputs, bijector_dict)
            return x, (y, x)

        subkey = jax.random.split(key, op.length)

        carry, (ys, xs) = jax.lax.scan(myfun, init, (subkey,) + mapped_rest, length=op.length)
        return (ys, xs)

    # TODO: Should be parallel!
    def log_prob(self, op, x, parent_values):
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

    # TODO: Should be parallel!
    def unconstrained_log_prob(self, op, y, parent_values, bijector_dict):
        assert isinstance(op, ir.Scan)
        assert op.random
        init, merge_args, mapped_rest = summarize_scan(op, parent_values)

        def myfun(carry, value_input):
            value = value_input[0]
            input = value_input[1:]
            inputs = (carry,) + merge_args(input)
            l, x = unconstrained_log_prob_op(op.base_op, value, inputs, bijector_dict)

            return value, (l, x)  # pass value to next iteration

        carry, (ls, xs) = jax.lax.scan(myfun, init, (y,) + mapped_rest, length=op.length)
        return jnp.sum(ls), xs

    def constrain(self, op, y, parent_values, bijector_dict):
        assert isinstance(op, ir.Scan)
        assert op.random
        init, merge_args, mapped_rest = summarize_scan(op, parent_values)

        def myfun(carry, value_input):
            value = value_input[0]
            input = value_input[1:]
            inputs = (carry,) + merge_args(input)
            x = constrain_op(op.base_op, value, inputs, bijector_dict)
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
    def sample(self, op, key, parent_values):
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

    def unconstrained_sample(self, op, key, parent_values, bijector_dict):
        assert isinstance(op, ir.VMap)
        assert op.random

        out_shape = op.get_shape(*[shape(p) for p in parent_values])
        out_axis_size = out_shape[0]

        subkey = jax.random.split(key, out_axis_size)

        def base_sample(key, *args):
            return unconstrained_sample_op(op.base_op, key, args, bijector_dict)

        in_axes = (0,) + op.in_axes
        axis_size = op.axis_size
        return jax.vmap(base_sample, in_axes=in_axes, axis_size=axis_size)(subkey, *parent_values)

    def log_prob(self, op, x, parent_values):
        assert isinstance(op, ir.VMap)
        assert op.random

        def base_log_prob(value, *args):
            return log_prob_op(op.base_op, value, args)

        in_axes = (0,) + op.in_axes
        axis_size = op.axis_size
        return jnp.sum(jax.vmap(base_log_prob, in_axes=in_axes, axis_size=axis_size)(x, *parent_values))

    def unconstrained_log_prob(self, op, y, parent_values, bijector_dict):
        assert isinstance(op, ir.VMap)
        assert op.random

        def base_log_prob(value, *args):
            return unconstrained_log_prob_op(op.base_op, value, args, bijector_dict)

        in_axes = (0,) + op.in_axes
        axis_size = op.axis_size
        ls, x = jax.vmap(base_log_prob, in_axes=in_axes, axis_size=axis_size)(y, *parent_values)
        return jnp.sum(ls), x

    def constrain(self, op, y, parent_values, bijector_dict):
        assert isinstance(op, ir.VMap)
        assert op.random

        def base_unconstrain(value, *parent_values):
            return constrain_op(op.base_op, value, parent_values, bijector_dict)

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

    parent_values = [jnp.array(v) for v in parent_values]

    op_class = type(op)
    handler = eval_handlers[op_class]
    out = handler(op, parent_values)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(out) == expected_shape, "Error: shape was not as expected"
    return out


def sample_op(op: Op, key: JaxArray, parent_values: Sequence[ArrayLike]):
    """
    Given a single `Op` and parent values, draw a sample.
    """
    if not op.random:
        raise ValueError("Cannot evaluate sample_op for non-random op")

    parent_values = [jnp.array(v) for v in parent_values]

    op_class = type(op)
    handler = handlers[op_class]
    x = handler.sample(op, key, parent_values)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(x) == expected_shape, "Error: shape was not as expected"
    return x


def unconstrained_sample_op(op: Op, key: JaxArray, parent_values: Sequence[ArrayLike], bijector_dict):
    """
    Given a single `Op` and parent values, draw a constrained sample. Also return unconstrained value.
    """
    if not op.random:
        raise ValueError("Cannot evaluate sample_op for non-random op")

    parent_values = [jnp.array(v) for v in parent_values]

    op_class = type(op)
    handler = handlers[op_class]
    y, x = handler.unconstrained_sample(op, key, parent_values, bijector_dict)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(x) == expected_shape, "Error: shape was not as expected"
    return y, x


def log_prob_op(op: Op, x: ArrayLike, parent_values: Sequence[ArrayLike]):
    """
    Given a single `Op`, evaluate log_prob.
    """
    if not op.random:
        raise ValueError("Cannot evaluate log_prob_op for non-random op")

    x = jnp.array(x)
    parent_values = [jnp.array(v) for v in parent_values]

    op_class = type(op)
    handler = handlers[op_class]
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(x) == expected_shape, "Error: shape was not as expected"
    l = handler.log_prob(op, x, parent_values)
    return l


def unconstrained_log_prob_op(op: Op, y: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict):
    """
    Given a single `Op`, evaluate log_prob and return unconstrained value.
    """
    if not op.random:
        raise ValueError("Cannot evaluate log_prob_op for non-random op")

    parent_values = [jnp.array(v) for v in parent_values]
    y = jnp.array(y)

    op_class = type(op)
    handler = handlers[op_class]
    l, x = handler.unconstrained_log_prob(op, y, parent_values, bijector_dict)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(x) == expected_shape, "Error: shape was not as expected"
    return l, x


def constrain_op(op: Op, constrained_value: ArrayLike, parent_values: Sequence[ArrayLike], bijector_dict: dict):
    if not op.random:
        raise ValueError("Cannot unconstrain non-random op")

    parent_values = [jnp.array(v) for v in parent_values]
    constrained_value = jnp.array(constrained_value)

    op_class = type(op)
    handler = handlers[op_class]
    value = handler.constrain(op, constrained_value, parent_values, bijector_dict)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    if shape(value) != expected_shape:
        raise ValueError(f"{shape(value)=} does not match {expected_shape=}")
    return value


################################################################################
# Functions to do sample and/or log prob on a "flat" graph
################################################################################

# TODO: Add conditioning?


def ancestor_sample_flat_single(vars: list[RV], key: JaxArray, bijector_dict: Optional[dict] = None):
    all_vars = dag.upstream_nodes(vars)
    constrained_values: dict[RV, JaxArray] = {}
    unconstrained_values: dict[RV, JaxArray] = {}
    for var in all_vars:
        parent_values = [constrained_values[p] for p in var.parents]
        if var.op.random:
            key, subkey = jax.random.split(key)
            if bijector_dict is None:
                x = y = sample_op(var.op, subkey, parent_values)
            else:
                y, x = unconstrained_sample_op(var.op, subkey, parent_values, bijector_dict)
        else:
            x = y = eval_op(var.op, parent_values)
        unconstrained_values[var] = y
        constrained_values[var] = x
    return [unconstrained_values[var] for var in vars]


def ancestor_sample_flat(
    vars: list[RV],
    key: Optional[JaxArray] = None,
    size: Optional[int] = None,
    bijector_dict: Optional[dict] = None,
):

    if key is None:
        # Generate random seed from numpy
        seed = np.random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)

    if size is None:
        return ancestor_sample_flat_single(vars, key, bijector_dict)
    else:
        mysample = lambda key: ancestor_sample_flat_single(vars, key, bijector_dict)
        keys = jax.random.split(key, size)
        return jax.vmap(mysample)(keys)


def ancestor_log_prob_flat(vars: Sequence[RV], values: Sequence[JaxArray], bijector_dict: Optional[dict] = None):
    all_vars = dag.upstream_nodes(vars)
    unconstrained_values = {var: val for var, val in zip(vars, values, strict=True)}
    constrained_values = {}
    l = 0.0
    for var in all_vars:
        parent_values = [constrained_values[p] for p in var.parents]
        if var.op.random:
            y = unconstrained_values[var]
            if bijector_dict is None:
                my_l = log_prob_op(var.op, y, parent_values)
                x = y
            else:
                my_l, x = unconstrained_log_prob_op(var.op, y, parent_values, bijector_dict)
            l += my_l
            constrained_values[var] = x
        else:
            if var in vars:
                raise ValueError("Can't provide value for non-random variable in ancestor_log_prob_flat")
            out = eval_op(var.op, parent_values)
            constrained_values[var] = out
    return l


def ancestor_constrain(vars: Sequence[RV], values: Sequence[JaxArray], bijector_dict: dict):
    all_vars = dag.upstream_nodes(vars)
    unconstrained_values = {var: val for var, val in zip(vars, values, strict=True)}
    constrained_values = {}
    for var in all_vars:
        parent_values = [constrained_values[p] for p in var.parents]
        if var.op.random:
            y = unconstrained_values[var]
            x = constrain_op(var.op, y, parent_values, bijector_dict)
            constrained_values[var] = x
        else:
            if var in vars:
                raise ValueError("Can't provide value for non-random variable in ancestor_unconstrain")
            out = eval_op(var.op, parent_values)
            constrained_values[var] = out
    return [constrained_values[var] for var in vars]


def fill_in(
    random_vars: Sequence[RV],
    random_values: Sequence[ArrayLike],
    desired_vars: Sequence[RV],
):
    # TODO: assert random / nonrandom / length / etc

    random_values = [jnp.array(v) for v in random_values]

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


def lookup_bijector_dict(biject: bool | dict = False):
    if isinstance(biject, dict):
        biject_dict = biject
    elif isinstance(biject, bool):
        if biject:
            biject_dict = bijectors.default_bijector_dict
        else:
            biject_dict = None
    return biject_dict


def ancestor_sample(
    vars: PyTree[RV], key: Optional[JaxArray] = None, size: Optional[int] = None, biject: bool | dict = False
):
    """
    Draw exact samples.

    Parameters
    ----------
    vars
        a Pytree of `RV` to sample
    key
        a JAX ``PRNGKey`` or ``None`` (default)
    size
        number of samples to draw (default of ``None`` is just a single sample)
    biject
        A boolean determining if constrained continuous distributions should be transformed to an unconstrained space. Or a dict giving a non-default set of bijectors. (Default: False)


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

    biject_dict = lookup_bijector_dict(biject)

    flat_samps = ancestor_sample_flat(flat_vars, key, size, biject_dict)

    return unflatten(flat_samps)


def ancestor_sampler(vars: PyTree[RV], biject: bool | dict = False) -> Callable[[JaxArray], Any]:
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
    biject
        A boolean determining if constrained continuous distributions should be transformed to an unconstrained space. Or a dict giving a non-default set of bijectors. (Default: False)


    Examples
    --------
    >>> x = RV(ir.Constant(1.5))
    >>> y = RV(ir.Add(), x, x)
    >>> fun = ancestor_sampler([{'cat': x}, y])

    You now have a plain-old JAX function that's completely independent of pangolin.

    >>> key = jax.random.PRNGKey(0)
    >>> fun(key)
    [{'cat': Array(1.5, dtype=float32)}, Array(3., dtype=...)]

    You can do normal JAX stuff with it, e.g. vmap it.

    >>> print(jax.vmap(fun)(jax.random.split(key, 3)))
    [{'cat': Array([1.5, 1.5, 1.5], dtype=float32)}, Array([3., 3., 3.], dtype=...)]

    """

    (
        flat_vars,
        _,
        _,
        unflatten,
        _,
    ) = util.flatten_args(vars, [], [])

    bijector_dict = lookup_bijector_dict(biject)

    def sampler(key):
        flat_samps = ancestor_sample_flat(flat_vars, key, size=None, bijector_dict=bijector_dict)

        return unflatten(flat_samps)

    return sampler


def ancestor_log_prob(*vars: PyTree[RV], biject: bool | dict = False, **kwvars: PyTree[RV]) -> Callable:
    """
    Given a pytree of vars, create a plain-old JAX function to compute log probabilities

    Parameters
    ----------
    vars
        pytrees of `RV`
    biject
        A boolean determining if constrained continuous distributions should be transformed to an unconstrained space. Or a dict giving a non-default set of bijectors. Only available as a keyword argument. (Default: False)
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
    Array([-0.9189385, -1.0439385], dtype=...)

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

    bijector_dict = lookup_bijector_dict(biject)

    def myfun(*vals, **kwvals):
        all_vals = (vals, kwvals)

        all_vals = util.assimilate_vals(all_vars, all_vals)  # casts lists and such to ndarray

        flat_vars, vars_treedef = jax.tree_util.tree_flatten(all_vars)
        flat_vals, vals_treedef = jax.tree_util.tree_flatten(all_vals)
        if vars_treedef != vals_treedef:
            raise ValueError("vars_treedef does not match vals_treedef")

        return ancestor_log_prob_flat(flat_vars, flat_vals, bijector_dict)

    return myfun
