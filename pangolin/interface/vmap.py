import collections

from . import OperatorRV
from pangolin.ir import Op, RV, VMap, Constant
from pangolin import dag, ir, util
from collections.abc import Callable
from .interface import makerv, rv_factory
from typing import Sequence, Type
import jax.tree_util
from . import interface
from .interface import api

from typing import Protocol
from numpy.typing import ArrayLike

# class FlatCallable(Protocol):
#    def __call__(self, *args) -> list[RV]:
#        ...

FlatCallable = Callable[..., list[RV]]

def check_tree_consistency(*args):
    trees = [jax.tree_util.tree_structure(args, is_leaf=util.is_leaf_with_none)]
    for t in trees:
        assert t == trees[0]

def get_flat_vmap_args_and_axes(in_axes, args):
    from pangolin.interface.interface import rv_factory

    def get_dummy(i, x):
        if i is None:
            new_shape = x.shape
        else:
            lo, mid, hi = (x.shape[:i], x.shape[i], x.shape[i + 1 :])
            new_shape = lo + hi

        # In old code tried to preserve x.op when isinstance(x.op, VMap)
        op = AbstractOp(new_shape, x.op.random)
        return rv_factory(op)

    dummy_args = util.tree_map_recurse_at_leaf(
        get_dummy, in_axes, args, is_leaf=util.is_leaf_with_none
    )
    new_in_axes = util.tree_map_recurse_at_leaf(
        lambda i, x: i, in_axes, dummy_args, is_leaf=util.is_leaf_with_none
    )
    check_tree_consistency(args, dummy_args, new_in_axes)

    flat_args, args_treedef = jax.tree_util.tree_flatten(args, is_leaf=util.is_leaf_with_none)
    flat_in_axes, axes_treedef = jax.tree_util.tree_flatten(
        new_in_axes, is_leaf=util.is_leaf_with_none
    )
    return dummy_args, new_in_axes, flat_args, flat_in_axes

@api
def vmap(f: Callable, in_axes: int | None | Sequence = 0, axis_size: int | None = None):
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

    if isinstance(in_axes,Sequence):
        in_axes = tuple(in_axes)

    def call(*args):
        # no greedy casting because this leads to ambiguity
        # if the user sends [(1,2),(3,4)] is that a list of two
        # arrays?
        args = jax.tree_util.tree_map(makerv, args)

        # if isinstance(d, VMapDist) and i == 0:
        #     my_dummy = AbstractRVWithDist(d.base_cond_dist, new_shape)
        # else:
        #     my_dummy = AbstractRV(new_shape)

        # def get_dummy(i, x):
        #     if i is None:
        #         new_shape = x.shape
        #     else:
        #         lo, mid, hi = (x.shape[:i], x.shape[i], x.shape[i + 1 :])
        #         new_shape = lo + hi
        #
        #     # In old code tried to preserve x.op when isinstance(x.op, VMap)
        #     op = AbstractOp(new_shape, x.op.random)
        #     return rv_factory(op)
        #
        # dummy_args = util.tree_map_recurse_at_leaf(
        #     get_dummy, in_axes, args, is_leaf=util.is_leaf_with_none
        # )
        # new_in_axes = util.tree_map_recurse_at_leaf(
        #     lambda i, x: i, in_axes, dummy_args, is_leaf=util.is_leaf_with_none
        # )
        #
        # tree1 = jax.tree_util.tree_structure(args, is_leaf=util.is_leaf_with_none)
        # tree2 = jax.tree_util.tree_structure(dummy_args, is_leaf=util.is_leaf_with_none)
        # tree3 = jax.tree_util.tree_structure(new_in_axes, is_leaf=util.is_leaf_with_none)
        # assert tree1 == tree2
        # assert tree1 == tree3

        dummy_args, new_in_axes, flat_args, flat_in_axes = get_flat_vmap_args_and_axes(in_axes,
                                                                                       args)

        #flat_in_axes, axes_treedef = jax.tree_util.tree_flatten(
        #    new_in_axes, is_leaf=util.is_leaf_with_none
        #)
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


def generated_nodes(fun: FlatCallable, *args: RV) -> tuple[list[RV], list[RV]]:
    """
    Given a "flat" function and some number of RV arguments, get all the nodes that the function
    creates. This *includes* nodes that do not depend on the inputs.

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

    # all generated nodes must have higher n
    n_before_call = OperatorRV.n
    def is_abstract(rv):
        return rv.n >= n_before_call

    abstract_out = fun(*args)

    if not isinstance(abstract_out, list):
        raise ValueError("generated_nodes must take a function that returns a list")
    if any(a in args for a in abstract_out):
        raise ValueError("fun passed to generated_nodes cannot return input values")
    for a in abstract_out:
        if a in args:
            raise ValueError("fun passed to generated_nodes cannot returned inputs.")
        if not isinstance(a, RV):
            raise ValueError(f"fun passed to generated_nodes returned non-RV output (got {type(a)}")

    all_abstract_vars = dag.upstream_nodes(
        abstract_out, block_condition=lambda var: not is_abstract(var)
    )

    all_abstract_vars = sorted(all_abstract_vars, key=lambda node:node.n)

    # convert abstract nodes to concrete
    abstract_to_concrete = {}
    for abstract_var in all_abstract_vars:
        if abstract_var in args:
            where_var = args.index(abstract_var)
            concrete_var = args[where_var]
        else:
            new_parents = tuple(
                abstract_to_concrete[p] if is_abstract(p) else p
                for p in abstract_var.parents
            )
            concrete_var = rv_factory(abstract_var.op, *new_parents)
        abstract_to_concrete[abstract_var] = concrete_var

    all_vars = [abstract_to_concrete[v] for v in all_abstract_vars if v not in args]
    out = [abstract_to_concrete[v] if v in abstract_to_concrete else v for v in abstract_out]

    return all_vars, out


def generated_nodes_old(fun: FlatCallable, *args: RV) -> tuple[list[RV], list[RV]]:
    """
    OLD VERSION, using tracing instead of node numbers

    Given a "flat" function and some number of RV arguments, get all the nodes that the function
    creates. This *includes* nodes that do not depend on the inputs.

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

    # some outputs might be non-abstract if independent of abstract_args
    with interface.SetRVFactory(TracerRV):
        abstract_out = fun(*args)

    if not isinstance(abstract_out, list):
        raise ValueError("generated_nodes must take a function that returns a list")
    if any(a in args for a in abstract_out):
        raise ValueError("fun passed to generated_nodes cannot return input values")
    for a in abstract_out:
        if a in args:
            raise ValueError("fun passed to generated_nodes cannot returned inputs.")
        if not isinstance(a, RV):
            raise ValueError(f"fun passed to generated_nodes returned non-RV output (got {type(a)}")
        if not isinstance(a, TracerRV):
            raise ValueError(
                f"fun passed to generated_nodes returned non-TracedRV type (got"
                f" {type(a)}. (Should be using only pangolin.interface functions.)"
            )

    all_abstract_vars = dag.upstream_nodes(
        abstract_out, block_condition=lambda var: not isinstance(var, TracerRV)
    )
    assert all(isinstance(v, TracerRV) for v in all_abstract_vars)

    # convert abstract nodes to concrete
    abstract_to_concrete = {}
    for abstract_var in all_abstract_vars:
        if abstract_var in args:
            where_var = args.index(abstract_var)
            concrete_var = args[where_var]
        else:
            new_parents = tuple(
                abstract_to_concrete[p] if isinstance(p, TracerRV) else p
                for p in abstract_var.parents
            )
            concrete_var = rv_factory(abstract_var.op, *new_parents)
        abstract_to_concrete[abstract_var] = concrete_var

    all_vars = [abstract_to_concrete[v] for v in all_abstract_vars if v not in args]
    out = [abstract_to_concrete[v] if v in abstract_to_concrete else v for v in abstract_out]

    return all_vars, out


class AbstractOp(Op):
    """
    An `AbstractOp` doesn't actually do anything and expects no parents. It just has a fixed shape.
    """

    def __init__(self, shape: tuple[int, ...], random: bool):
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


def vmap_dummy_args(in_axes: Sequence[int | None], axis_size: int | None, *args: RV):
    """
    Given a "full" arguments, get a list of dummy/sliced arguments
    Parameters
    ----------
    in_axes: tuple[int|None]
        What axis to map each argument over. Should have same length as `args`
    """

    if not util.all_unique(args):
        raise ValueError("vmap_dummy_args requires all unique arguments")

    dummy_args = []
    for i, a in zip(in_axes, args, strict=True):
        new_shape, new_axis_size = ir.vmap.split_shape(a.shape, i)

        # once upon a time we did this—but don't remember the point of it
        # if isinstance(a.op, VMap) and i == 0:
        #     new_op = a.op.base_op  # why do we care?
        # else:
        #     new_op = AbstractOp(new_shape, a.op.random)

        new_op = AbstractOp(new_shape, a.op.random)
        my_dummy = rv_factory(new_op)  # no parents!

        dummy_args.append(my_dummy)
        if axis_size is None:
            axis_size = new_axis_size
        elif new_axis_size is not None:
            assert axis_size == new_axis_size, "incoherent axis size"
    return tuple(dummy_args), axis_size


def vmap_subgraph(roots, dummy_roots, roots_axes, axis_size, dummy_nodes, dummy_outputs):
    """
    Parameters
    ----------
    roots
        "real" root RVs
    dummy_roots
        dummy root RVs (correspond to roots sliced according to `dummy_axes` and `axis_size`)
    roots_axes
        the axes along which dummy_roots have been sliced
    axis_size
        the axis size for all mapped nodes
    dummy_nodes
        sequence of RVs which may depend on each other
    dummy_outputs
        the desired outputs

    Returns
    -------
    real_outputs
        real nodes corresponding to `dummy_outputs`
    """

    if any(a in dummy_roots for a in dummy_nodes):
        raise ValueError("dummy_roots cannot be included in dummy_nodes")

    dummy_to_real = util.WriteOnceDefaultDict(lambda p: p)
    dummy_mapped_axis = util.WriteOnceDefaultDict(lambda p: None)
    for dummy_arg, i, arg in zip(dummy_roots, roots_axes, roots, strict=True):
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

        dummy_to_real[dummy_node] = rv_factory(new_op, *parents)
        dummy_mapped_axis[dummy_node] = new_axis
    real_nodes = [dummy_to_real[dummy_node] for dummy_node in dummy_outputs]
    return real_nodes


def vmap_eval(f, in_axes, axis_size, *args):
    """
    actually evaluate a vmap.
    This function (but not vmap itself) works on "flat" function f, meaning that each
    argument of the function is just a RV. And the function must return
    a list of arguments which again are each just a RV.
    """

    # make sure inputs are RVs
    args = list(makerv(a) for a in args)
    dummy_args, axis_size = vmap_dummy_args(in_axes, axis_size, *args)
    dummy_nodes, dummy_outputs = generated_nodes(f, *dummy_args)

    return vmap_subgraph(args, dummy_args, in_axes, axis_size, dummy_nodes, dummy_outputs)


def vmap_flat(f: FlatCallable, in_axes: tuple[int | None, ...], axis_size: int | None):
    """
    vmap a flat function (one that takes some number of RV arguments and returns a list of RV
    arguments)
    """

    def vec_f(*args):
        args = list(makerv(a) for a in args)
        dummy_args, my_axis_size = vmap_dummy_args(in_axes, axis_size, *args)
        dummy_nodes, dummy_outputs = generated_nodes(f, *dummy_args)
        return vmap_subgraph(args, dummy_args, in_axes, my_axis_size, dummy_nodes, dummy_outputs)

    return vec_f


def plate(*args, size: int | None = None, in_axes=0):
    """
    Plate is a simple shortcut notation to vmap. For example

    ```python
    z = normal(0,1)
    x = plate(size=10)(lambda:
        normal(z,1)
    )
    ```

    Is equivalent to
    ```python
    z = normal(0,1)
    x = vmap(lambda: normal(z,1),axis_size=10)()
    ```

    And

    ```python
    z = multi_normal(np.ones(3),np.eye(3))
    x = plate(z)(lambda z_i:
        normal(z_i,1)
    )
    ```

    Is equivalent to
    ```python
    z = multi_normal(np.ones(3),np.eye(3))
    x = vmap(lambda z_i: normal(z_i,1))(z)
    ```

    And

    ```python
    z = multi_normal(np.ones(3),np.eye(3))
    x = plate(z,size=3,in_axes=0)(lambda z_i:
        normal(z_i,1)
    )
    ```

    Is equivalent to
    ```python
    z = multi_normal(np.ones(3),np.eye(3))
    x = vmap(lambda z_i: normal(z_i,1),in_axes=0,axis_size=3)(z)
    ```
    """

    # args, in_axes = util.unzip(args_and_in_axes,strict=True)

    print(f"PLATE CALLED {in_axes=} {args=} {size=}")

    def get_mapped(fun: Callable):
        print(f"{fun=}")
        print(f"{in_axes=}")
        print(f"{args=}")
        print(f"{size=}")
        return vmap(fun, in_axes, axis_size=size)(*args)

    return get_mapped


# class SubInt(int):
#     def __new__(cls, value, *args, **kwargs):
#         return super(SubInt, cls).__new__(cls, value)
#
#     def __add__(self, other):
#         return SubInt(int(self)+int(other))
#
#     def __radd__(self, other):
#         # needed because regular sum starts with 0
#         return SubInt(int(self)+int(other))


# class Loop:
#     def __init__(self, length, auto_assign=True):
#         # rv_class = interface.rv_factory()
#         self.auto_assign = auto_assign
#         self.length = length
#         self.generated_rvs = []
#         self.loop_vars = {}
#
#         self.range = interface.rv_factory(Constant(range(length)))
#         self.i = self.range[0]
#
#         # TODO: instead of creating new class, create new factory?
#         class NewRV(OperatorRV):
#             def __init__(myself, *args, **vargs):
#                 self.generated_rvs.append(myself)
#                 myself.my_loop = self
#                 super().__init__(*args, **vargs)
#
#         self.new_rv_class = NewRV
#
#     def __enter__(self) -> OperatorRV:
#         interface.rv_factories.append(self.new_rv_class)
#         return self.i
#
#     def __exit__(self, exc_type, exc_value, exc_tb):
#         assert self.new_rv_class == interface.rv_factories.pop()
#         if self.auto_assign:
#             loop_vars = [l for l in self.loop_vars]
#             dummy_loop_vars = [self.loop_vars[l] for l in loop_vars]
#             new_vars = vmap_subgraph(
#                 [self.range], [self.i], [0], self.length, self.generated_rvs, dummy_loop_vars
#             )
#
#             for loop_var, new in zip(loop_vars, new_vars, strict=True):
#                 loop_var.finalize(new.op, *new.parents)
#
#
# class LoopVar(OperatorRV):
#     def __init__(self):
#         # do NOT call super.__init__() for now
#         pass
#
#     def __setitem__(self, idx, dummy_rv):
#         loop = dummy_rv.my_loop
#         loop.loop_vars[self] = dummy_rv
#         self.dummy_rv = dummy_rv
#
#     def __getitem__(self, loop):
#         return self.dummy_rv
#
#     def finalize(self, op, *parents):
#         super().__init__(op, *parents)
#
#     def __repr__(self):
#         out = "LoopVar"
#         if hasattr(self, "dummy_rv"):
#             out += "(" + repr(self.dummy_rv) + ")"
#         return out


# indexed_vmap(
#     lambda i, x, y:
#     (z := VMapRV())[:,i] = x[:,i] * y[:,i]
# )
