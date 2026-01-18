"""
This is an **experimental** sub-module to compile pangolin models into plain-old pytorch functions.

**Note**: Because pytorch is large and sometimes annoying to install, and many users
will not use this functionality, pangolin does not install pytorch as a requirement by
default. This might lead you to get this error:

``ImportError: Using torch backend requires torch to be installed``

To fix this, either install pangolin with the pytorch requirements
(e.g. with ``uv sync --extra torch``) or manually install pytorch yourself
(e.g. with ``pip install torch`` or ``uv pip install torch``).

**Also note**: This backend has some limitations in its support for certain distributions, as shown in the following table:

============================ ======== ================ ========= =================
Op                           sampling vmapped sampling log probs vmapped log probs
============================ ======== ================ ========= =================
``Beta``                     ✔        ❌               ✔         ✔
``StudentT``                 ✔        ❌               ✔         ✔
``Dirichlet``                ✔        ❌               ✔         ✔
``Multinomial``              ✔        ❌               ✔         ✔
``Wishart``                  ✔        ❌               ✔         ✔
``MultiNormal``              ✔        ❌               ✔         ✔
``BetaBinomial``             ❌       ❌               ❌        ❌
``Multinomial``              ❌       ❌               ❌        ❌
============================ ======== ================ ========= =================

Everything else is fully supported.

For ``Multinomial`` and ``BetaBinomial``, this is due to torch lacking these distributions. For the others, this is due to a basic ~~bug~~ limitations in PyTorch. Namely, in PyTorch, this works fine:

``torch.vmap(lambda dummy: torch.distributions.Normal(0,1).sample(), randomness='different')(torch.zeros(2))``

And this *should* work fine:

``torch.vmap(lambda dummy: torch.distributions.Exponential(2.0).rsample(), randomness='different')(torch.zeros(2))``

But the latter raises the error ``RuntimeError: vmap: Cannot ask for different inplace randomness on an unbatched tensor. This will appear like same randomness. If this is necessary for your usage, please file an issue with functorch.`` (Said issue is `here <https://github.com/pytorch/functorch/issues/996>`_ .) Curiously, ``Gamma`` does not have this issue, even though ``Exponential`` does and ``Gamma`` is a generalization of ``Exponential``. So this backend just creates a ``Gamma`` when an ``Exponential`` is needed.

"""

from __future__ import annotations
import numpy as np
from typing import (
    Callable,
    Type,
    Sequence,
    Optional,
)
from pangolin import ir
from pangolin.ir import Op, RV
from numpy.typing import ArrayLike
from pangolin import dag, util
from jaxtyping import PyTree

try:
    import torch
    import torch.distributions as dist
except ImportError:
    raise ImportError("Using torch backend requires torch to be installed")

from typing import Type, Callable

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
    ir.Abs: torch.abs,
    ir.Arccos: torch.acos,
    ir.Arccosh: torch.acosh,
    ir.Arcsin: torch.asin,
    ir.Arcsinh: torch.asinh,
    ir.Arctan: torch.atan,
    ir.Arctanh: torch.atanh,
    ir.Cos: torch.cos,
    ir.Cosh: torch.cosh,
    ir.Exp: torch.exp,
    ir.Identity: lambda a: a,
    ir.InvLogit: torch.sigmoid,
    ir.Log: torch.log,
    ir.Loggamma: torch.lgamma,
    ir.Logit: torch.logit,
    ir.Sin: torch.sin,
    ir.Sinh: torch.sinh,
    ir.Step: lambda x: torch.heaviside(x, values=torch.tensor(0.5)),
    ir.Tan: torch.tan,
    ir.Tanh: torch.tanh,
    ir.Matmul: torch.matmul,
    ir.Inv: torch.linalg.inv,
    ir.Cholesky: torch.linalg.cholesky,
    ir.Transpose: torch.t,
    ir.Diag: torch.diagonal,
    ir.DiagMatrix: torch.diag_embed,
    ir.Softmax: lambda x, axis=-1: torch.nn.Softmax(dim=axis)(x),
    ir.SimpleIndex: ir.index_orthogonal_no_slices,
}


################################################################################
# Dict of Ops that correspond to simple (Numpyro) distributions
################################################################################


def wrap(torch_dist_class: Type[torch.distributions.Distribution]):
    """Torch validate_args causes problems with vmap. So we need to turn it off.
    Unfortunately there's no way to do this by default.
    """
    # TODO: Maybe someday selectively enable validation outside of vmap

    def fun(*args, **kwargs):
        return torch_dist_class(*args, **kwargs, validate_args=False)

    return fun


def not_implemented(*args):
    raise NotImplementedError()


simple_dists: dict[Type[Op], Callable] = {
    ir.Normal: wrap(dist.Normal),
    ir.NormalPrec: lambda loc, prec: wrap(dist.Normal)(loc, 1 / prec**2),
    ir.Bernoulli: wrap(dist.Bernoulli),
    ir.BernoulliLogit: lambda logits: wrap(dist.Bernoulli)(logits=logits),
    ir.Beta: wrap(dist.Beta),
    ir.BetaBinomial: not_implemented,
    ir.Binomial: wrap(dist.Binomial),
    ir.Categorical: wrap(dist.Categorical),
    ir.Cauchy: wrap(dist.Cauchy),
    ir.Exponential: lambda scale: wrap(dist.Gamma)(1, scale),  # dist.exponential doesn't support vmap
    ir.Dirichlet: wrap(dist.Dirichlet),
    ir.Gamma: wrap(dist.Gamma),
    ir.Lognormal: wrap(dist.LogNormal),
    ir.Multinomial: wrap(dist.Multinomial),
    ir.MultiNormal: wrap(dist.MultivariateNormal),
    ir.Poisson: wrap(dist.Poisson),
    ir.StudentT: wrap(dist.StudentT),
    ir.Uniform: wrap(dist.Uniform),
    ir.Wishart: wrap(dist.Wishart),
}

################################################################################
# Factories that will take an Op class and give log prob and/or sample functions
################################################################################


def maketensor(a):
    if isinstance(a, torch.Tensor):
        return a
    else:
        return torch.tensor(a)


def make_simple_log_prob(op_class: Type[Op]):
    bind = simple_dists[op_class]

    def my_log_prob(op, value, parent_values):
        value = maketensor(value)
        parent_values = [maketensor(p) for p in parent_values]
        bound_dist: dist.Distribution = bind(*parent_values)
        return bound_dist.log_prob(value)

    return my_log_prob


def make_simple_sample(op_class: Type[Op]):
    bind = simple_dists[op_class]

    def my_sample(op, parent_values):
        parent_values = [maketensor(p) for p in parent_values]
        bound_dist: dist.Distribution = bind(*parent_values)
        return bound_dist.sample()

    return my_sample


def make_simple_eval(op_class):
    fun = simple_funs[op_class]

    def simple_eval(op, parent_values):
        parent_values = [maketensor(p) for p in parent_values]
        return fun(*parent_values)

    return simple_eval


################################################################################
# Basic dicts that map Op types to log_prob and/or sample handlers
################################################################################


log_prob_handlers: dict[Type[Op], Callable] = {}
sample_handlers: dict[Type[Op], Callable] = {}
eval_handlers: dict[Type[Op], Callable] = {}

for op_class in simple_dists:
    log_prob_handlers[op_class] = make_simple_log_prob(op_class)
    sample_handlers[op_class] = make_simple_sample(op_class)


for op_class in simple_funs:
    eval_handlers[op_class] = make_simple_eval(op_class)

################################################################################
# Constant handler
################################################################################


def eval_constant(op: ir.Constant, parent_values):
    return torch.tensor(op.value, dtype=torch.get_default_dtype())


eval_handlers[ir.Constant] = eval_constant

################################################################################
# Sum handler
################################################################################


def eval_sum(op: ir.Sum, parent_values: Sequence[ArrayLike]):
    assert len(parent_values) == 1
    parent_value = maketensor(parent_values[0])
    return torch.sum(parent_value, dim=op.axis)


eval_handlers[ir.Sum] = eval_sum

################################################################################
# Composite handler
################################################################################


def summarize_composite(op: ir.Composite, parent_values: Sequence[ArrayLike]):
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


def log_prob_composite(op: ir.Composite, value, parent_values: Sequence[ArrayLike]):
    assert op.random
    final_op, final_parent_values = summarize_composite(op, parent_values)
    return log_prob_op(final_op, value, final_parent_values)


def sample_composite(op: ir.Composite, parent_values: Sequence[ArrayLike]):
    assert op.random
    final_op, final_parent_values = summarize_composite(op, parent_values)
    return sample_op(final_op, final_parent_values)


eval_handlers[ir.Composite] = eval_composite
log_prob_handlers[ir.Composite] = log_prob_composite
sample_handlers[ir.Composite] = sample_composite

################################################################################
# Autoregressive
################################################################################


import torch
from typing import Callable, Any, Optional, Tuple, List
import torch.utils._pytree as pytree

import torch
from typing import Callable, Optional, Tuple, Union, Any


def scan(
    f: Callable,
    init: torch.Tensor,
    xs: Tuple[torch.Tensor, ...],
    length: Optional[int] = None,
) -> Tuple[Any, Any]:
    """
    Simple PyTorch scan for tensors/tuples only

    Parameters
    ----------
    f
        Function with signature (carry, x) -> (carry, y)
    init
        Initial carry value
    xs
        Input to scan over (tuple of tensors, can be empty)
    length
        Number of iterations (required if xs is empty tuple)

    Returns
    -------
    final_carry
        final carry value
    ys
        stacked outputs

    Examples
    --------
    >>> # Empty tuple with length - counter example
    >>> def counter(carry, _):
    ...     return carry + 1, carry
    >>> init = torch.tensor(0)
    >>> final, counts = scan(counter, init, xs=(), length=5)
    >>> final
    tensor(5)
    >>> counts
    tensor([0, 1, 2, 3, 4])

    >>> # Cumulative sum with single tensor in tuple
    >>> def cumsum_fn(carry, x):
    ...     new_carry = carry + x[0]
    ...     return new_carry, new_carry
    >>> init = torch.tensor(0.0)
    >>> xs = (torch.tensor([1.0, 2.0, 3.0, 4.0]),)
    >>> final, sums = scan(cumsum_fn, init, xs)
    >>> final
    tensor(10.)
    >>> sums
    tensor([ 1.,  3.,  6., 10.])

    >>> # Fibonacci sequence using empty tuple
    >>> def fib_step(carry, _):
    ...     a, b = carry
    ...     return (b, a + b), a
    >>> init = (torch.tensor(0), torch.tensor(1))
    >>> _, fibs = scan(fib_step, init, xs=(), length=7)
    >>> fibs
    tensor([0, 1, 1, 2, 3, 5, 8])

    >>> # Working with tuples of tensors
    >>> def dual_accumulate(carry, x):
    ...     c1, c2 = carry
    ...     x1, x2 = x
    ...     return (c1 + x1, c2 * x2), (c1 + x1, c2 * x2)
    >>> init = (torch.tensor(0.0), torch.tensor(1.0))
    >>> xs = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 3.0, 4.0]))
    >>> final, ys = scan(dual_accumulate, init, xs)
    >>> final[0]  # sum: 0+1+2+3 = 6
    tensor(6.)
    >>> final[1]  # product: 1*2*3*4 = 24
    tensor(24.)
    >>> ys[0]  # cumulative sums
    tensor([1., 3., 6.])
    >>> ys[1]  # cumulative products
    tensor([ 2.,  6., 24.])
    """
    # Determine iteration count
    if len(xs) == 0:
        # Empty tuple case - must use length
        if length is None:
            raise ValueError("length must be provided when xs is empty tuple")
        scan_length = length
    else:
        scan_length = xs[0].shape[0]
        if length is not None:
            assert scan_length == length

    assert isinstance(scan_length, int)

    # Prepare inputs
    if len(xs) == 0:
        # Empty tuple - create list of None
        xs_list = [()] * scan_length
    else:
        # Multiple tensors - extract tuple of elements at each index
        xs_list = [tuple(x[i] for x in xs) for i in range(scan_length)]

    # Run scan
    carry = init
    ys_list = []

    for x in xs_list:
        carry, y = f(carry, x)
        ys_list.append(y)

    # Stack outputs
    if ys_list and ys_list[0] is not None:
        if isinstance(ys_list[0], torch.Tensor):
            ys = torch.stack(ys_list, dim=0)
        elif isinstance(ys_list[0], tuple):
            ys = tuple(torch.stack([y[i] for y in ys_list], dim=0) for i in range(len(ys_list[0])))
        else:
            ys = ys_list
    else:
        ys = None

    return carry, ys


def handle_autoregressive_inputs(op: ir.Autoregressive, *numpyro_parents) -> tuple[tuple[torch.Tensor, ...], Callable]:
    for in_axis in op.in_axes:
        assert in_axis in [
            0,
            None,
        ], "Torch only supports Autoregressive with in_axis of 0 or None"

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


def eval_autoregressive(op: ir.Autoregressive, parent_values: Sequence[ArrayLike]):
    assert isinstance(op, ir.Autoregressive)
    assert not op.random

    init = maketensor(parent_values[0])
    rest = tuple(maketensor(p) for p in parent_values[1:])

    mapped_rest, merge_args = handle_autoregressive_inputs(op, *rest)
    assert merge_args(mapped_rest) == tuple(rest)

    def myfun(carry, x):
        inputs = (carry,) + merge_args(x)
        y = eval_op(op.base_op, inputs)
        return y, y

    carry, ys = scan(myfun, init, mapped_rest, length=op.length)
    return ys


eval_handlers[ir.Autoregressive] = eval_autoregressive


# TODO: This should be parallel!
def log_prob_autoregressive(op: ir.Autoregressive, value: ArrayLike, parent_values: Sequence[ArrayLike]):
    assert isinstance(op, ir.Autoregressive)
    assert op.random

    value = maketensor(value)

    init = maketensor(parent_values[0])
    rest = tuple(maketensor(p) for p in parent_values[1:])

    mapped_rest, merge_args = handle_autoregressive_inputs(op, *rest)
    assert merge_args(mapped_rest) == tuple(rest)

    def myfun(carry, value_x):
        value = value_x[0]
        x = value_x[1:]
        inputs = (carry,) + merge_args(x)
        l = log_prob_op(op.base_op, value, inputs)
        return value, l  # pass value to next iteration

    carry, ls = scan(myfun, init, (value,) + mapped_rest, length=op.length)
    return torch.sum(ls)


log_prob_handlers[ir.Autoregressive] = log_prob_autoregressive


def sample_autoregressive(op: ir.Autoregressive, parent_values: Sequence[ArrayLike]):
    assert isinstance(op, ir.Autoregressive)
    assert op.random

    # init = parent_values[0]
    # rest = parent_values[1:]
    init = maketensor(parent_values[0])
    rest = tuple(maketensor(p) for p in parent_values[1:])

    mapped_rest, merge_args = handle_autoregressive_inputs(op, *rest)
    assert merge_args(mapped_rest) == tuple(rest)

    def myfun(carry, x):
        inputs = (carry,) + merge_args(x)
        y = sample_op(op.base_op, inputs)
        return y, y

    carry, ys = scan(myfun, init, mapped_rest, length=op.length)
    return ys


sample_handlers[ir.Autoregressive] = sample_autoregressive


################################################################################
# VMap
################################################################################


def my_vmap(fun, in_dims, axis_size):
    """A version of vmap that can handle an axis_size argument"""
    # TODO: check axis_size even when in_dims not all None

    if any(in_dim is not None for in_dim in in_dims):
        return torch.vmap(fun, in_dims, randomness="different")
    else:

        def dummy_fun(dummy, *rest):
            return fun(*rest)

        dummy = torch.zeros(axis_size)
        my_in_dims = (0,) + in_dims

        def evaluator(*args):
            return torch.vmap(dummy_fun, my_in_dims, randomness="different")(dummy, *args)

        return evaluator


def eval_vmap(op: ir.VMap, parent_values: Sequence[ArrayLike]):
    assert isinstance(op, ir.VMap)
    assert not op.random

    parent_values = tuple(maketensor(p) for p in parent_values)

    def base_eval(*args):
        return eval_op(op.base_op, args)

    in_axes = op.in_axes
    axis_size = op.axis_size  # axis_size=axis_size

    # return torch.vmap(base_eval, in_dims=in_axes)(*parent_values)
    return my_vmap(base_eval, in_dims=in_axes, axis_size=axis_size)(*parent_values)


eval_handlers[ir.VMap] = eval_vmap


def sample_vmap(op: ir.VMap, parent_values: Sequence[ArrayLike]):
    assert isinstance(op, ir.VMap)
    assert op.random

    parent_values = tuple(maketensor(p) for p in parent_values)

    out_shape = op.get_shape(*[shape(p) for p in parent_values])
    out_axis_size = out_shape[0]

    def base_sample(*args):
        return sample_op(op.base_op, args)

    in_axes = op.in_axes
    axis_size = op.axis_size
    return my_vmap(base_sample, in_dims=in_axes, axis_size=axis_size)(*parent_values)


sample_handlers[ir.VMap] = sample_vmap


def log_prob_vmap(op: ir.VMap, value: ArrayLike, parent_values: Sequence[ArrayLike]):
    assert isinstance(op, ir.VMap)
    assert op.random

    # def base_log_prob(value, *args):
    #    return log_prob_op(op.base_op, value, args)
    base_log_prob = log_prob_op_fun(op.base_op)

    in_axes = (0,) + op.in_axes
    axis_size = op.axis_size

    return torch.sum(my_vmap(base_log_prob, in_dims=in_axes, axis_size=axis_size)(value, *parent_values))


log_prob_handlers[ir.VMap] = log_prob_vmap

################################################################################
# Transformed
################################################################################


def sample_transformed[O: Op, B: ir.Bijector](op: ir.Transformed[O, B], parent_values: Sequence[ArrayLike]):
    assert isinstance(op, ir.Transformed)
    assert op.random
    assert op.base_op.random

    bijector_args = tuple(parent_values[: op.n_biject_args])
    dist_args = parent_values[op.n_biject_args :]

    x = sample_op(op.base_op, dist_args)
    y = eval_op(op.bijector.forward, (x,) + bijector_args)
    return y


sample_handlers[ir.Transformed] = sample_transformed


def log_prob_transformed[O: Op, B: ir.Bijector](
    op: ir.Transformed[O, B], y: ArrayLike, parent_values: Sequence[ArrayLike]
):
    assert isinstance(op, ir.Transformed)
    assert op.random
    assert op.base_op.random

    bijector_args = tuple(parent_values[: op.n_biject_args])
    dist_args = parent_values[op.n_biject_args :]

    x = eval_op(op.bijector.inverse, (y,) + bijector_args)
    log_px = log_prob_op(op.base_op, x, dist_args)
    log_jac_det = eval_op(op.bijector.log_det_jac, (x, y) + bijector_args)
    return log_px - log_jac_det


log_prob_handlers[ir.Transformed] = log_prob_transformed

################################################################################
# Functions to do sample and/or log prob on a single node
################################################################################


def shape(value: ArrayLike):
    if isinstance(value, torch.Tensor):
        return value.shape
    else:
        return torch.tensor(value).shape


def sample_op(op: Op, parent_values: Sequence[ArrayLike]):
    """
    Given a single `Op` and parent values, draw a sample.
    """
    if not op.random:
        raise ValueError("Cannot evaluate sample_op for non-random op")

    op_class = type(op)
    handler = sample_handlers[op_class]
    out = handler(op, parent_values)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    assert shape(out) == expected_shape, "Error: shape was not as expected"
    return out


def log_prob_op_fun(op):
    if not op.random:
        raise ValueError("Cannot evaluate log_prob_op for non-random op")
    op_class = type(op)
    handler = log_prob_handlers[op_class]

    def fun(value, *parent_values):
        # expected_shape = op.get_shape(*[shape(v) for v in parent_values])
        # if shape(value) != expected_shape:
        #     raise ValueError(
        #         f"shape(value) {shape(value)} not {expected_shape} as expected"
        #     )
        return handler(op, value, parent_values)

    return fun


def log_prob_op(
    op: Op,
    value: ArrayLike,
    parent_values: Sequence[ArrayLike],
):
    """
    Given a single `Op`, evaluate log_prob.
    """
    if not op.random:
        raise ValueError("Cannot evaluate log_prob_op for non-random op")
    op_class = type(op)
    expected_shape = op.get_shape(*[shape(v) for v in parent_values])
    if shape(value) != expected_shape:
        raise ValueError(f"shape(value) {shape(value)} not {expected_shape} as expected")
    return log_prob_handlers[op_class](op, value, parent_values)


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


################################################################################
# Functions to do sample and/or log prob on a "flat" graph
################################################################################

# TODO: Add conditioning?


def ancestor_sample_flat_single(vars: list[RV]):
    all_vars = dag.upstream_nodes(vars)
    all_values = {}
    for var in all_vars:
        parent_values = [all_values[p] for p in var.parents]
        if var.op.random:
            all_values[var] = sample_op(var.op, parent_values)
        else:
            all_values[var] = eval_op(var.op, parent_values)
    return [all_values[var] for var in vars]


def ancestor_sample_flat(vars: list[RV], size: Optional[int] = None):
    if size is None:
        return ancestor_sample_flat_single(vars)
    else:
        mysample = lambda dummy: ancestor_sample_flat_single(vars)
        dummy = torch.zeros(size)
        # return torch.vmap(mysample, randomness="different")(dummy)
        return my_vmap(mysample, in_dims=(0,), axis_size=size)(dummy)


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
# Torch tree utils
################################################################################

from typing import Any, Callable, List
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten


def assimilate_vals_pytorch(vars, vals):
    """
    Converts `vals` to a pytree of tensors with the same structure and shape as `vars`.

    Args:
        vars: A pytree of tensors that serves as a template for structure, shape,
              dtype, and device.
        vals: A pytree of values (e.g., lists, scalars) that has the same
              structure as `vars` and whose elements can be converted to tensors
              with shapes matching the corresponding tensors in `vars`.

    Returns:
        A new pytree with the same structure as `vars`, where each leaf is a
        torch.Tensor created from the corresponding value in `vals`.

    Raises:
        ValueError: If the pytree structures of `vars` and `vals` do not match,
                    or if the shapes of the resulting tensors do not match the
                    template tensors in `vars`.
        TypeError: If the leaves of `vars` are not tensor-like objects.
    """
    flat_vars, vars_spec = tree_flatten(vars)
    flat_vals, vals_spec = tree_flatten(vals)

    if vars_spec != vals_spec:
        raise ValueError(
            f"Pytree structure mismatch for vars and vals.\n"
            f"Vars structure: {vars_spec}.\n"
            f"Vals structure: {vals_spec}."
        )

    new_flat_tensors = []
    for var_template, val in zip(flat_vars, flat_vals):
        # Ensure the template is a tensor to get its properties (shape, dtype, device)
        if not isinstance(var_template, torch.Tensor):
            raise TypeError(
                f"The template pytree `vars` must contain only tensors, but found "
                f"an object of type {type(var_template)}."
            )

        val_tensor = maketensor(val)
        # torch.as_tensor(
        #     val, dtype=var_template.dtype, device=var_template.device
        # )

        if var_template.shape != val_tensor.shape:
            raise ValueError(
                f"Shape mismatch: The template tensor has shape {var_template.shape}, "
                f"but the corresponding value creates a tensor with shape {val_tensor.shape}."
            )
        new_flat_tensors.append(val_tensor)

    new_vals_pytree = tree_unflatten(new_flat_tensors, vars_spec)

    return new_vals_pytree


def flatten_args_torch(
    vars_pytree: Any, given_vars_pytree: Any = None, given_vals_pytree: Any = None
) -> tuple[List, List, List, Callable, Callable]:
    """
    Flattens pytrees and provides functions to unflatten them.

    This function takes up to three pytrees (`vars`, `given_vars`, `given_vals`),
    flattens each into a list of leaves, and returns these lists along with
    helper functions to reconstruct the original `vars` and `given_vars` structures.

    It asserts that `given_vars_pytree` and `given_vals_pytree` have the exact
    same pytree structure.

    Args:
        vars_pytree: The primary pytree of variables to flatten.
        given_vars_pytree: An optional pytree of observed variable specifications.
        given_vals_pytree: An optional pytree of observed variable values.

    Returns:
        A tuple containing:
        - flat_vars (List): The flattened leaves of `vars_pytree`.
        - flat_given_vars (List): The flattened leaves of `given_vars_pytree`.
        - flat_given_vals (List): The flattened leaves of `given_vals_pytree`.
        - unflatten_vars (Callable): A function to reconstruct the `vars_pytree` structure.
        - unflatten_given (Callable): A function to reconstruct the `given_vars_pytree` structure.
    """

    given_vals_pytree = assimilate_vals_pytorch(given_vars_pytree, given_vals_pytree)  # casts lists and such to tensor

    flat_vars, vars_treespec = tree_flatten(vars_pytree)
    flat_given_vars, given_vars_treespec = tree_flatten(given_vars_pytree)
    flat_given_vals, given_vals_treespec = tree_flatten(given_vals_pytree)

    # The TreeSpec objects returned by PyTorch's tree_flatten can be
    # compared directly, just like JAX's TreeDefs.
    assert given_vars_treespec == given_vals_treespec, (
        "The pytree structure of `given_vars` and `given_vals` must be identical. "
        f"Got {given_vars_treespec} vs {given_vals_treespec}"
    )

    # Create simple lambda functions to capture the treespecs for unflattening.
    unflatten_vars = lambda flat_leaves: tree_unflatten(flat_leaves, vars_treespec)
    unflatten_given = lambda flat_leaves: tree_unflatten(flat_leaves, given_vars_treespec)

    return flat_vars, flat_given_vars, flat_given_vals, unflatten_vars, unflatten_given


################################################################################
# Functions to do sample and/or log prob on any graph
################################################################################


def ancestor_sample(vars: PyTree[RV], size: Optional[int] = None):
    """
    Draw exact samples!

    Parameters
    ----------
    vars
        a PyTree of `RV` s to sample
    size
        number of samples to draw (default of ``None`` is just a single sample)

    Returns
    -------
    out
        PyTree matching structure of ``vars``, but with ``torch.tensor`` in place
        of `RV`. If ``size`` is ``None``, then each tensor will have the same shape as
        the corresponding ``RV``. Otherwise, each tensor will have an extra dimension
        of size ``size`` appended at the beginning.

    Examples
    --------

    Sample a constant RV.

    >>> x = RV(ir.Constant(1.5))
    >>> ancestor_sample(x)
    tensor(1.5000)

    Sample a PyTree with the RV inside it.

    >>> ancestor_sample({'sup': [[x]]})
    {'sup': [[tensor(1.5000)]]}

    Draw several samples.

    >>> ancestor_sample(x, size=3)
    tensor([1.5000, 1.5000, 1.5000])

    Sample several samples from a PyTree with an RV inside it.

    >>> ancestor_sample({'sup': x}, size=3)
    {'sup': tensor([1.5000, 1.5000, 1.5000])}

    Sample from several random variables at once

    >>> y = RV(ir.Add(), x, x)
    >>> z = RV(ir.Mul(), x, y)
    >>> print(ancestor_sample({'cat': x, 'dog': [y, z]}))
    {'cat': tensor(1.5000), 'dog': [tensor(3.), tensor(4.5000)]}

    See Also
    --------
    pangolin.jax_backend.ancestor_sample
    """

    (
        flat_vars,
        _,
        _,
        unflatten,
        _,
    ) = flatten_args_torch(vars, [], [])

    flat_samps = ancestor_sample_flat(flat_vars, size)

    return unflatten(flat_samps)


def ancestor_sampler(vars):
    """
    Compiles a pytree of RVs into a plain-old torch function that returns a pytree with the same structure containing a joint sample from the distribution of those RVs.

    Parameters
    ----------
    vars
        a pytree of `RV` to sample


    Returns
    -------
    out
        callable to create a sample matching the structure and shape of ``vars``


    Examples
    --------
    >>> x = RV(ir.Constant(1.5))
    >>> y = RV(ir.Add(), x, x)
    >>> fun = ancestor_sampler([{'cat': x}, y])

    You now have a plain-old torch function that's completely independent of pangolin.

    >>> fun()
    [{'cat': tensor(1.5000)}, tensor(3.)]

    You can do normal torch stuff with it, e.g. vmap. But note that limitations
    in pytorch mean that you must pass some kind of dummy argument and pass
    ``randomness='different'`` to get independent samples.

    >>> print(torch.vmap(lambda dummy: fun(), randomness='different')(torch.ones(3)))
    [{'cat': tensor([1.5000, 1.5000, 1.5000])}, tensor([3., 3., 3.])]

    See Also
    --------
    pangolin.jax_backend.ancestor_sampler
    """

    (
        flat_vars,
        _,
        _,
        unflatten,
        _,
    ) = flatten_args_torch(vars, [], [])

    def sampler():
        # flat_samps = ancestor_sample_flat(flat_vars, key)
        flat_samps = ancestor_sample_flat(flat_vars)

        return unflatten(flat_samps)

    return sampler


def ancestor_log_prob(*vars, **kwvars):
    """
    Given a pytree of vars, create a plain-old torch function to compute log_probabilities.

    Parameters
    ----------
    vars
        pytrees of `RV`
    kwargs
        more pytrees of `RV`

    Returns
    -------
    out
        log_prob function


    Examples
    --------
    >>> loc = ir.RV(ir.Constant(0.0))
    >>> scale = ir.RV(ir.Constant(1.0))
    >>> x = RV(ir.Normal(),loc,scale)
    >>> fun = ancestor_log_prob(x)

    You now have a plain torch function that's completely independent of pangolin. You can evaluate it.

    >>> fun(torch.tensor(0.0))
    tensor(-0.9189)

    Or you can vmap it.

    >>> torch.vmap(fun)(torch.tensor([0.0, 0.5]))
    tensor([-0.9189, -1.0439])

    Here's a more complex example:

    >>> op = ir.VMap(ir.Normal(), [None,None], 3)
    >>> y = RV(op,loc,scale)
    >>> fun = ancestor_log_prob({'x':x, 'y':y})
    >>> fun({'x':torch.tensor(0.0), 'y':torch.tensor([0.0, 0.5, 0.1])})
    tensor(-3.8058)

    You can also create a function that uses positional and/or keyword arguments:

    >>> fun = ancestor_log_prob(x, cat=y)
    >>> fun(torch.tensor(0.0), cat=torch.tensor([0.0, 0.5, 0.1]))
    tensor(-3.8058)
    """

    all_vars = (vars, kwvars)

    def myfun(*vals, **kwvals):
        all_vals = (vals, kwvals)

        flat_vars, vars_treedef = tree_flatten(all_vars)
        flat_vals, vals_treedef = tree_flatten(all_vals)

        if vars_treedef != vals_treedef:
            raise ValueError("vars_treedef does not match vals_treedef")

        if not all(isinstance(val, torch.Tensor) for val in flat_vals):
            raise TypeError("All passed vals must be torch tensor")

        if not all(isinstance(var, ir.RV) for var in flat_vars):
            raise TypeError("All passed vars must be RV")

        return ancestor_log_prob_flat(flat_vars, flat_vals)

    return myfun
