import collections

from . import OperatorRV
from cleanpangolin.ir import Op, RV, VMap, Constant
from cleanpangolin import dag, ir, util
from collections.abc import Callable
from .interface import makerv, get_rv_class
from typing import Sequence, Type
import jax.tree_util

def vmap(f: Callable, in_axes: int | None | Sequence=0, axis_size: int | None=None):
    """@public
    vmap a function. See also the documentation for
    [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) which has exactly the same
    interface (with some extra arguments not supported here).

    Parameters
    ----------
    f: Callable
        The function to vmap. Should take a pytree of `RV`s as inputs and return a pytree of `RV`s
        as outputs.
    in_axes:
        An integer, None, or sequence of values specifying what input axes to map over. If
        the positional arguments to `f` are container (pytree) types then `in_axes` must be a
        sequence with length equal to the number of positional arguments to `f` and each element
        of `in_axes` must be a container tree prefix of the corresponding positional argument.
    axis_size: int | None
        An integer indicating the size of the axis to be mapped. This is optional unless all
        leaves of `in_axes` are `None`.

    Returns
    -------
    vec_f
        batched/vectorized version of `f`
    """

    def call(*args):
        # no greedy casting because this leads to ambiguity
        # if the user sends [(1,2),(3,4)] is that a list of two
        # arrays?
        args = jax.tree_util.tree_map(makerv, args)

        rv_class = get_rv_class(*jax.tree_util.tree_leaves(args))

        # if isinstance(d, VMapDist) and i == 0:
        #     my_dummy = AbstractRVWithDist(d.base_cond_dist, new_shape)
        # else:
        #     my_dummy = AbstractRV(new_shape)

        def get_dummy(i, x):
            if i is None:
                new_shape = x.shape
            else:
                lo, mid, hi = (x.shape[:i], x.shape[i], x.shape[i + 1:])
                new_shape = lo + hi

            # In old code tried to preserve x.op when isinstance(x.op, VMap)
            op = AbstractOp(new_shape, x.op.random)
            return rv_class(op)

        dummy_args = util.tree_map_recurse_at_leaf(
            get_dummy, in_axes, args, is_leaf=util.is_leaf_with_none
        )
        new_in_axes = util.tree_map_recurse_at_leaf(
            lambda i, x: i, in_axes, dummy_args, is_leaf=util.is_leaf_with_none
        )

        tree1 = jax.tree_util.tree_structure(args, is_leaf=util.is_leaf_with_none)
        tree2 = jax.tree_util.tree_structure(dummy_args, is_leaf=util.is_leaf_with_none)
        tree3 = jax.tree_util.tree_structure(new_in_axes, is_leaf=util.is_leaf_with_none)
        assert tree1 == tree2
        assert tree1 == tree3

        flat_in_axes, axes_treedef = jax.tree_util.tree_flatten(
            new_in_axes, is_leaf=util.is_leaf_with_none
        )
        flat_f, flatten_inputs, unflatten_output = util.flatten_fun(
            f, *dummy_args, is_leaf=util.is_leaf_with_none
        )
        flat_args = flatten_inputs(*args)
        flat_output = vmap_eval(flat_f, flat_in_axes, axis_size, *flat_args)
        output = unflatten_output(flat_output)
        return output
    return call

def convert_args(rv_type: Type[RV], *args: RV):
    """
    Given some set of (interdependent) RVs, get a new set where all are converted to a new type
    but all inter-RV parent links are preserved.

    Parameters
    ----------
    rv_type
        Some subclass of `RV`
    args:RV
        arguments, all of type RV

    Returns
    -------
    new_args:tuple[rv_type]
        converted args
    """
    abstract_args = {}
    for a in args:
        new_parents = [abstract_args[p] if p in abstract_args else p for p in a.parents]
        abstract_a = rv_type(a.op, *new_parents)
        abstract_args[a] = abstract_a
    return tuple(abstract_args[a] for a in args)


def generated_nodes(fun: Callable[[tuple[RV]], list[RV]], *args: RV) -> tuple[list[RV], list[RV]]:
    """
    Given a "flat" function and some number of RV arguments, get all the nodes that the function
    generates that are downstream of those arguments.

    Parameters
    ----------
    fun: Callable[[Tuple[RV]], List[RV]]
        A function that takes some number of `RV` arguments and returns a list of `RV`s
    *args: RV
        arguments to call the function on.
    Returns
    -------
    all_vars: List[RV]
        All `RV`s that are generated by this function and downstream of `args`
    out: List[RV]
        The outputs of the original function (typically overlaps with `all_vars`)
    """
    for a in args:
        assert isinstance(a, RV), "arguments must be RVs"

    class TracerRV(OperatorRV):
        """
        New RV type, just for tracing out this function
        """

        pass

    abstract_args = convert_args(TracerRV, *args)
    # some outputs might be non-abstract if independent of abstract_args
    abstract_out = fun(*abstract_args)

    if not isinstance(abstract_out, list):
        raise ValueError("generated_nodes must take a function that returns a list")
    if not all(isinstance(a, TracerRV) for a in abstract_out):
        raise ValueError(
            f"all outputs of fun passed to generated_nodes must depend on at least one input. ("
            f"Got {abstract_out})"
        )
    if any(a in abstract_args for a in abstract_out):
        raise ValueError("fun passed to generated_nodes cannot return input values")

    all_abstract_vars = dag.upstream_nodes(
        abstract_out, block_condition=lambda var: not isinstance(var, TracerRV)
    )
    assert all(isinstance(v, TracerRV) for v in all_abstract_vars)

    # convert abstract nodes to concrete
    abstract_to_concrete = {}
    for abstract_var in all_abstract_vars:
        if abstract_var in abstract_args:
            where_var = abstract_args.index(abstract_var)
            concrete_var = args[where_var]
        else:
            new_parents = tuple(
                abstract_to_concrete[p] if isinstance(p, TracerRV) else p
                for p in abstract_var.parents
            )
            # concrete_var = OperatorRV(abstract_var.op, *new_parents)
            # get most specific class
            rv_class = get_rv_class(*new_parents)
            concrete_var = rv_class(abstract_var.op, *new_parents)
        abstract_to_concrete[abstract_var] = concrete_var

    all_vars = [abstract_to_concrete[v] for v in all_abstract_vars if v not in abstract_args]
    out = [abstract_to_concrete[v] if v in abstract_to_concrete else v for v in abstract_out]

    return all_vars, out


class AbstractOp(Op):
    """
    An `AbstractOp` doesn't actually do anything and expects no parents. It just has a fixed shape.
    """

    def __init__(self, shape: tuple[int], random: bool):
        """
        Create an abstract Op.

        Parameters
        ----------
        shape: tuple[int]
            the shape for the output
        random: bool
            is the op random
        """
        self.shape = shape
        super().__init__(name="AbstractOp", random=random)

    def _get_shape(self, *parents_shapes):
        return self.shape

    def __eq__(self, other):
        """
        Equality for abstract ops is very restricted
        """
        return id(self) == id(other)


def vmap_dummy_args(in_axes: tuple[int|None], axis_size: int|None, *args: RV):
    """
    Given a "full" arguments, get a list of dummy/sliced arguments
    Parameters
    ----------
    in_axes: tuple[int|None]
        What axis to map each argument over. Should have same same length as `args`
    """

    if not util.all_unique(args):
        raise ValueError("vmap_dummy_args requires all unique arguments")

    rv_class = get_rv_class(*args)

    dummy_args = []
    for i, a in zip(in_axes, args, strict=True):
        new_shape, new_axis_size = ir.vmap.split_shape(a.shape, i)

        if isinstance(a.op, VMap) and i == 0:
            new_op = a.op.base_op  # why do we care?
        else:
            new_op = AbstractOp(new_shape, a.op.random)
        my_dummy = rv_class(new_op)  # no parents!

        dummy_args.append(my_dummy)
        if axis_size is None:
            axis_size = new_axis_size
        elif new_axis_size is not None:
            assert axis_size == new_axis_size, "incoherent axis size"
    return tuple(dummy_args), axis_size


def vmap_eval(f, in_axes, axis_size, *args):
    """
    actually evaluate a vmap.
    This function (but not vmap itself) works on "flat" function f, meaning that each
    argument of the function is just a RV. And the function must return
    a list of arguments which again are each just a RV.
    """

    # make sure inputs are RVs
    args = list(makerv(a) for a in args)
    # slice arguments as appropriate
    dummy_args, axis_size = vmap_dummy_args(in_axes, axis_size, *args)
    # run function, get all generated nodes
    dummy_nodes, dummy_outputs = generated_nodes(f, *dummy_args)

    rv_class = get_rv_class(*args)

    dummy_to_real = util.WriteOnceDefaultDict(lambda p: p)
    dummy_mapped_axis = util.WriteOnceDefaultDict(lambda p: None)
    for dummy_arg, i, arg in zip(dummy_args, in_axes, args, strict=True):
        dummy_to_real[dummy_arg] = arg
        dummy_mapped_axis[dummy_arg] = i

    for dummy_node in dummy_nodes:
        dummy_parents = dummy_node.parents
        parents = [dummy_to_real[p] for p in dummy_parents]
        my_in_axes = [dummy_mapped_axis[p] for p in dummy_parents]

        no_mapped_axes = all(axis is None for axis in my_in_axes)
        if no_mapped_axes and not dummy_node.op.random and dummy_node not in dummy_outputs:
            new_op = dummy_node.op
            new_axis = None
        else:
            new_op = VMap(dummy_node.op, in_axes=my_in_axes, axis_size=axis_size)
            new_axis = 0

        #dummy_to_real[dummy_node] = OperatorRV(new_op, *parents)
        dummy_to_real[dummy_node] = rv_class(new_op, *parents)
        dummy_mapped_axis[dummy_node] = new_axis


    output = [dummy_to_real[dummy] for dummy in dummy_outputs]
    return output


