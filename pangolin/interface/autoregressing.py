from pangolin.ir import Composite, RV
from pangolin.ir import Autoregressive
from pangolin.interface import (
    InfixRV,
    makerv,
    constant,
    create_rv,
    exp,
    add,
    normal,
    exponential,
)
from pangolin.interface.compositing import composite_flat, make_composite
from .vmapping import generated_nodes, AbstractOp, get_dummy_args
from pangolin import util
import jax.tree_util
from pangolin.interface.base import RVLike

# from pangolin.simple_interface.vmapping import get_flat_vmap_args_and_axes
from typing import Callable, Sequence, Any


# would like to insist that the function takes RV args but type system not up to the task
SingleOutputFun = Callable[..., RV]


def _get_autoregressive_length(length: int | None, my_in_axes: Sequence[int | None], args: Sequence[RV]) -> int:
    "If length is given, checks that it's compatible with all args. Otherwise, infers from args and chcecks they are compatible."

    my_length = length

    for ax, arg in zip(my_in_axes, args, strict=True):
        if ax is not None:
            if my_length:
                assert my_length == arg.shape[ax]
            else:
                my_length = arg.shape[ax]

    if my_length is None:
        raise ValueError("Can't create autoregressive with length=None and no mapped axis")

    return my_length


def autoregressive_flat(flat_fun: SingleOutputFun, length: int, in_axes: tuple[int | None, ...]) -> Callable:
    """
    Given a "flat" function, create a function to generate an RV with an `Autoregressive` Op.

    Doing

    ```python
    z = autoregressive_flat(fun, length=5, in_axes=[0,None])(start, A, B)
    ```

    is semantically similar to

    ```python
    carry = start
    values = []
    for i in range(5):
        carry = fun(carry, A[i], B)
        values.append(carry)
    z = concatenate(values)
    ```

    Parameters
    ----------
    flat_fun
        function for base of autoregressive
    length: int | None
        length of autoregressive. Can be None if any inputs are mapped along some axis.
    in_axes: int | None | Sequence[int | None]

    Examples
    --------
    >>> x = autoregressive_flat(exp, 5, None)(makerv(7.7))
    >>> x.op
    Autoregressive(Composite(1, (Exp(),), ((0,),)), 5, (), 0)
    >>> x.op.base_op
    Composite(1, (Exp(),), ((0,),))
    >>> x.op.length
    5
    >>> x.op.in_axes
    ()
    >>> x.op.where_self
    0
    >>> x.parents
    (InfixRV(Constant(7.7)),)

    >>> a = makerv(7.7)
    >>> b = makerv([1,2,3,4,5])
    >>> x = autoregressive_flat(add, 5, None)(a, b)
    >>> x.op
    Autoregressive(Composite(2, (Add(),), ((0, 1),)), 5, (0,), 0)
    >>> x.op.base_op
    Composite(2, (Add(),), ((0, 1),))
    >>> x.op.length
    5
    >>> x.op.in_axes
    (0,)
    >>> x.op.where_self
    0
    >>> x.parents
    (InfixRV(Constant(7.7)), InfixRV(Constant([1,2,3,4,5])))

    """

    # next = flat_fun(prev, *args)
    # from pangolin.interface.base import rv_factory

    def myfun(init: RVLike, *args0: RVLike):
        init = makerv(init)
        args = tuple(makerv(a) for a in args0)

        # my_length = _get_autoregressive_length(length, in_axes, args)

        # first, get composite op
        init_shape = init.shape
        args_shapes = tuple(a.shape for a in args)
        base_args_shapes = tuple(s[1:] for s in args_shapes)  # assume all mapped along 1st axis
        base_op, constants = make_composite(flat_fun, init_shape, *base_args_shapes)
        where_self = 0
        # where_self = len(constants) # constants always first in composite
        op = Autoregressive(
            base_op,
            length,
            in_axes=[None] * len(constants) + list(in_axes),
            where_self=where_self,
        )
        return create_rv(op, init, *constants, *args)

    return myfun


def autoregressive(fun: Callable, length: None | int = None, in_axes: Any = 0) -> Callable[..., RV[Autoregressive]]:
    """
    Given a function, create a function to generate an RV with an `Autoregressive` Op. Doing

    .. code-block:: python

        auto_fun = autoregressive(fun, 5, [0, None])
        z = autoregressive_fun(start, A, B)

    is semantically like

    .. code-block:: python

        carry = start
        values = []
        # length = 5
        for i in range(5):
            # in_axes = [0, None] <-> A sliced on 0th axis, B unsliced
            carry = fun(carry, A[i], B)
            values.append(carry)
        z = concatenate(values)

    Parameters
    ----------
    fun
        Function to call repeatedly to define the distribution.
        Must take ``carry`` as the first input.
        Must return a single `RV`.
        Can only create a single *random* `RV`, which must be the final output.
        But can create an arbitrary number of *non*-random `RV`.
        Can optionally take extra inputs that will be mapped.
    length
        Length of autoregressive. Can be ``None`` if any inputs are mapped along some axis.
    in_axes
        What axis to map each input other than ``carry`` over (or ``None`` if
        non-mapped).
        As with `vmap`, can be a pytree of `RV` corresponding to the structure of all
        inputs other than ``carry``.

    Returns
    -------
    auto_fun
        Function that takes some number of pytrees of `RV` with mapped axes and produces a single ``RV[Autoregressive]``

    Examples
    --------
    Distribution where ``z[i] ~ normal(exp(z[i-1]), 1)``.

    >>> x = constant(3.3)
    >>> def fun(carry):
    ...     return normal(exp(carry), 1)
    >>> z = autoregressive(fun, 5)(x)
    >>> z.parents == (x,)
    True
    >>> print(z.op.name)
    Autoregressive
    >>> z.op.length
    5
    >>> print(z.op.base_op.name)
    Composite
    >>> z.op.base_op.num_inputs
    1
    >>> z.op.base_op.ops
    (Exp(), Constant(1), Normal())
    >>> z.op.base_op.par_nums
    ((0,), (), (1, 2))
    >>> z.op.in_axes
    ()

    Distribution where ``z[i] ~ exponential(z[i-1]*y[i])``

    >>> x = constant(3.3)
    >>> y = constant([1,2,3])
    >>> def fun(carry, yi):
    ...     return exponential(carry * yi)
    >>> z = autoregressive(fun)(x,y)
    >>> print(z.op.name)
    Autoregressive
    >>> z.op.length
    3
    >>> z.parents == (x, y)
    True
    >>> print(z.op.base_op.name)
    Composite
    >>> z.op.base_op.num_inputs
    2
    >>> z.op.base_op.ops
    (Mul(), Exponential())
    >>> z.op.base_op.par_nums
    ((0, 1), (2,))
    >>> z.op.in_axes
    (0,)

    See Also
    --------
    pangolin.ir.Autoregressive
    autoregress
    """

    # TODO: Document pytree inputs

    def myfun(init: RVLike, *args):
        if len(args) == 1:
            # handles vmap(f, 0)(x) instead vmap(f,(0,))(x)
            my_in_axes = (in_axes,)
        elif isinstance(in_axes, list):
            # handles vmap(f, [0,1])(x,y) instead of vmap(f,(0,1))(x,y)
            my_in_axes = tuple(in_axes)
        else:
            my_in_axes = in_axes

        init = makerv(init)
        args = jax.tree_util.tree_map(makerv, args)

        # abstract args without extra dims
        dummy_args = get_dummy_args(my_in_axes, args)
        flat_in_axes, flat_args = util.dual_flatten(my_in_axes, args)

        flat_fun, flatten_inputs, unflatten_output = util.flatten_fun(
            fun, init, *dummy_args, is_leaf=util._is_leaf_with_none
        )
        new_flat_fun = lambda *args: flat_fun(*args)[0]  # don't return list

        my_length = _get_autoregressive_length(length, flat_in_axes, flat_args)

        return autoregressive_flat(new_flat_fun, my_length, flat_in_axes)(init, *flat_args)

    return myfun


def autoregress(length: int | None = None, in_axes: Any = 0) -> Callable:
    """
    Simple decorator to create functions to create autoregressive RVs. The idea is that

    .. code-block:: python

        autoregress(length, in_axes)(fun)

    is exactly the same as

    .. code-block:: python

        autoregressive(fun, length, in_axes)

    This can be very convenient as a decorator.

    Parameters
    ----------
    length
        the number of repetitions
    in_axes
        axis to map arguments other than ``carry`` over

    Examples
    --------

    Here are two equivalent examples. Here's `autoregressive`:

    >>> x = constant(3.3)
    >>> def fun(carry):
    ...     return normal(exp(carry), 1)
    >>> z1 = autoregressive(fun,5)(x)

    And here's `autoregress`:

    >>> @autoregress(5)
    ... def fun(carry):
    ...     return normal(exp(carry), 1)
    >>> z2 = fun(x)

    >>> z1.op == z2.op
    True
    >>> z1.parents == z2.parents == (x,)
    True

    See Also
    --------
    autoregressive
    pangolin.ir.Autoregressive
    """

    return lambda fun: autoregressive(fun, length, in_axes)
