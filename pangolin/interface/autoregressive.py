from pangolin.ir import Composite, RV
from pangolin.ir import Autoregressive
from pangolin.interface import OperatorRV, makerv
from pangolin.interface.composite import composite_flat, make_composite
from .vmapping import generated_nodes, AbstractOp
from pangolin import util
import jax.tree_util
from pangolin.interface.base import RV_or_ArrayLike
from pangolin.interface.vmapping import get_flat_vmap_args_and_axes
from typing import Callable, Sequence


def _get_autoregressive_length(
    length: int | None, my_in_axes: Sequence[int | None], args: Sequence[RV]
) -> int:
    "If length is given, checks that it's compatible with all args. Otherwise, infers from args and chcecks they are compatible."

    my_length = length

    for ax, arg in zip(my_in_axes, args, strict=True):
        if ax is not None:
            if my_length:
                assert my_length == arg.shape[ax]
            else:
                my_length = arg.shape[ax]

    if my_length is None:
        raise ValueError(
            "Can't create autoregressive with length=None and no mapped axis"
        )

    return my_length


def autoregressive_flat(
    flat_fun, length: int | None = None, in_axes0: None | Sequence[int | None] = None
) -> Callable:
    """
    next = flat_fun(prev,*args)
    """
    from pangolin.interface.base import rv_factory

    def myfun(init: RV_or_ArrayLike, *args0: RV_or_ArrayLike):
        init = makerv(init)
        args = tuple(makerv(a) for a in args0)

        # if no axes given assume all 0
        if in_axes0:
            in_axes = in_axes0
        else:
            in_axes = [0 for _ in args]

        my_length = _get_autoregressive_length(length, in_axes, args)

        # first, get composite op
        init_shape = init.shape
        args_shapes = tuple(a.shape for a in args)
        base_args_shapes = tuple(
            s[1:] for s in args_shapes
        )  # assume all mapped along 1st axis
        base_op, constants = make_composite(flat_fun, init_shape, *base_args_shapes)
        where_self = 0
        # where_self = len(constants) # constants always first in composite
        op = Autoregressive(
            base_op,
            my_length,
            in_axes=[None] * len(constants) + list(in_axes),
            where_self=where_self,
        )
        return rv_factory(op, init, *constants, *args)

    return myfun


def autoregressive(fun: Callable, length: None | int = None, in_axes=0):
    """
    next = flat_fun(prev,*args)
    """

    def myfun(init: RV_or_ArrayLike, *args):
        init = makerv(init)
        args = jax.tree_util.tree_map(makerv, args)

        # TODO: Find a good name and push this functionality back to vmap
        dummy_args, new_in_axes, flat_args, flat_in_axes = get_flat_vmap_args_and_axes(
            in_axes, args
        )

        flat_fun, flatten_inputs, unflatten_output = util.flatten_fun(
            fun, init, *dummy_args, is_leaf=util.is_leaf_with_none
        )
        new_flat_fun = lambda *args: flat_fun(*args)[0]  # don't return list
        return autoregressive_flat(new_flat_fun, length, flat_in_axes)(init, *flat_args)

    return myfun


def repeat(length=None, in_axes=0):
    """
    Simple decorator to create functions to create autoregressive RVs

    Examples
    --------
    @auto(length=10)
    def fun(last):
        return normal(last,1)
    x = fun(0)
    """

    return lambda fun: autoregressive(fun, length, in_axes)
