from pangolin.ir import Composite
from pangolin.ir.autoregressive import Autoregressive
from pangolin.interface import OperatorRV, makerv
from pangolin.interface.composite import composite_flat, make_composite
from .vmap import generated_nodes, AbstractOp
from pangolin import util
import jax.tree_util
from pangolin.interface.interface import RV_or_array
from pangolin.interface.vmap import get_flat_vmap_args_and_axes
from typing import Callable

# def autoregressive_flat(length, flat_fun):
#     """
#     next = flat_fun(prev,*args)
#     """
#     from pangolin.interface.interface import rv_factory
#     def myfun(init, *args):
#         # first, get composite op
#         init_shape = init.shape
#         args_shapes = tuple(a.shape for a in args)
#         print(f"{init_shape=}")
#         print(f"{args_shapes=}")
#         base_op, constants = make_composite(flat_fun, init_shape, *args_shapes)
#         print(f"{base_op=}")
#         # now make autoregressive
#         in_axes = tuple(None for a in args)
#         op = Autoregressive(base_op, length, in_axes, 0)
#         return rv_factory(op,init,*args)
#     return myfun


def autoregressive_flat(flat_fun, length=None, in_axes=None):
    """
    next = flat_fun(prev,*args)
    """
    from pangolin.interface.interface import rv_factory

    def myfun(init: RV_or_array, *args: RV_or_array):
        init = makerv(init)
        args = tuple(makerv(a) for a in args)

        # if no axes given assume all 0
        my_in_axes = in_axes
        if my_in_axes is None:
            my_in_axes = [0 for a in args]

        # if no length given, get from axes
        my_length = length
        if my_length is None:
            if all(ax is None for ax in my_in_axes):
                raise ValueError("Can't create autoregressive with length=None and no mapped axes")
            for ax, arg in zip(my_in_axes, args, strict=True):
                if ax is not None:
                    my_length = arg.shape[ax]

        # first, get composite op
        init_shape = init.shape
        args_shapes = tuple(a.shape for a in args)
        base_args_shapes = tuple(s[1:] for s in args_shapes)  # assume all mapped along 1st axis
        base_op, constants = make_composite(flat_fun, init_shape, *base_args_shapes)
        where_self = 0
        #where_self = len(constants) # constants always first in composite
        op = Autoregressive(
            base_op, my_length, in_axes=[None] * len(constants) + my_in_axes, where_self=where_self
        )
        return rv_factory(op, init, *constants, *args)

    return myfun


# def get_flat_vmap_args_and_axes(in_axes, args):
#     from pangolin.interface.interface import rv_factory
#
#     def get_dummy(i, x):
#         if i is None:
#             new_shape = x.shape
#         else:
#             lo, mid, hi = (x.shape[:i], x.shape[i], x.shape[i + 1 :])
#             new_shape = lo + hi
#
#         # In old code tried to preserve x.op when isinstance(x.op, VMap)
#         op = AbstractOp(new_shape, x.op.random)
#         return rv_factory(op)
#
#     dummy_args = util.tree_map_recurse_at_leaf(
#         get_dummy, in_axes, args, is_leaf=util.is_leaf_with_none
#     )
#     new_in_axes = util.tree_map_recurse_at_leaf(
#         lambda i, x: i, in_axes, dummy_args, is_leaf=util.is_leaf_with_none
#     )
#     check_tree_consistency(args, dummy_args, new_in_axes)
#
#     flat_args, args_treedef = jax.tree_util.tree_flatten(args, is_leaf=util.is_leaf_with_none)
#     flat_in_axes, axes_treedef = jax.tree_util.tree_flatten(
#         new_in_axes, is_leaf=util.is_leaf_with_none
#     )
#     return dummy_args, new_in_axes, flat_args, flat_in_axes
#
#
# def check_tree_consistency(*args):
#     trees = [jax.tree_util.tree_structure(args, is_leaf=util.is_leaf_with_none)]
#     for t in trees:
#         assert t == trees[0]


def autoregressive(fun: Callable, length:None | int = None, in_axes=0):
    """
    next = flat_fun(prev,*args)
    """
    from pangolin.interface.interface import rv_factory


    def myfun(init: RV_or_array, *args):
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



# def autoregressive(fun, length):
#     """
#     next = flat_fun(prev,*args)
#     """
#
#     def myfun(init: RV_or_array, *args):
#         init = makerv(init)
#
#         # flat_args, args_treedef = jax.tree_util.tree_flatten(
#         #    args, is_leaf=util.is_leaf_with_none
#         # )
#
#         flat_fun, flatten_inputs, unflatten_output = util.flatten_fun(fun, init, *args)
#         flat_init_plus_args = flatten_inputs(init, *args)
#         assert flat_init_plus_args[0] is init
#         flat_args = flat_init_plus_args[1:]
#         new_flat_fun = lambda *args: flat_fun(*args)[0]  # don't return list
#         return autoregressive_flat(new_flat_fun, length)(init, *flat_args)
#
#     return myfun


# def autoregressive(fun,length=None):
#     """
#     Convenience function for creating Autoregressive(Composite) RVs
#     fun - function where 1st argument is recursive variable and other arguments are whatever.
#           it is OK if fun implicitly uses existing random variables as a closure.
#     length - length to be mapped over (optional unless there are no argments)
#     """
#     def myfun(init, *args):
#         if length is None:
#             if args == ():
#                 raise ValueError("autoregressive needs length if there are no mapped args")
#             else:
#                 my_length = args[0].shape[0]
#                 for a in args:
#                     assert a.shape[0] == my_length, "all args must have matching first dim"
#         else:
#             for a in args:
#                 assert a.shape[0] == length, "all args must have first dim matching length"
#             my_length = length
#
#         args_shapes = tuple(a.shape[1:] for a in args)
#         input_shapes = (init.shape,) + args_shapes
#         base_op, consts = make_composite(fun, *input_shapes)
#         print(f"{consts=}")
#         num_constants = len(consts)
#         op = Autoregressive(base_op, length=my_length, num_constants=num_constants)
#         return op(init,*consts,*args)
#     return myfun
