from pangolin.ir import Op, VMap, Constant, print_upstream
from pangolin import dag, ir, util
from collections.abc import Callable
from .base import makerv, create_rv, RVLike, constant, exp, log, InfixRV
from typing import Sequence, Type
import jax.tree_util
from pangolin.ir import split_shape

from typing import Protocol, TypeVar, Any, TYPE_CHECKING
from numpy.typing import ArrayLike
import numpy as np
from jax import numpy as jnp
from typing import Protocol
from jaxtyping import PyTree
from .tracing import AbstractOp, generated_nodes, FlatCallable, pack


Shape = ir.Shape


def is_full_slice(x: InfixRV, len: int) -> int | None:
    """

    Examples
    --------
    >>> x = ir.RV(ir.Constant([0,1,2]))
    >>> is_full_slice(x, 3)
    True
    >>> x = ir.RV(ir.Constant([0,1,2]))
    >>> is_full_slice(x, 5)
    False
    >>> x = ir.RV(ir.Constant(1))
    >>> y = ir.RV(ir.Exp(),x)
    >>> is_full_slice(y, 5)
    False
    """
    return isinstance(x.op, ir.Constant) and (x.op == ir.Constant(range(len)))


def is_vmap_over_scalar_index(x: InfixRV) -> int | None:
    """

    Examples
    --------
    >>> a = ir.RV(ir.Constant([1.1, 2.2, 3.3]))
    >>> b = ir.RV(ir.Constant([0, 1, 2]))
    >>> c = ir.RV(ir.VMap(ir.Index(), [None, 0], 3), a, b)
    >>> is_vmap_over_scalar_index(c)
    0

    >>> a = ir.RV(ir.Constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]]))
    >>> b = ir.RV(ir.Constant([0, 1]))
    >>> c = ir.RV(ir.Constant([0, 1, 2]))
    >>> d = ir.RV(ir.VMap(ir.Index(), [None, 0, None], 2), a, b, c)
    >>> is_vmap_over_scalar_index(d)
    0
    >>> e = ir.RV(ir.VMap(ir.Index(), [None, None, 0], 3), a, b, c)
    >>> is_vmap_over_scalar_index(e)
    1

    """

    if isinstance(x.op, ir.VMap):
        if isinstance(x.op.base_op, ir.Index):
            indexed_rv = x.parents[0]
            index_rvs = x.parents[1:]

            indexed_axis = x.op.in_axes[0]
            index_axes = x.op.in_axes[1:]

            if indexed_axis is None:
                all_full_slices = all(is_full_slice(p, i) for (p, i) in zip(index_rvs, indexed_rv.shape, strict=True))
                if all_full_slices:
                    num_0 = sum(1 for ax in index_axes if ax == 0)
                    num_None = sum(1 for ax in index_axes if ax is None)
                    only_one_mapped = (num_0 == 1) and (num_None == indexed_rv.ndim - 1)
                    if only_one_mapped:
                        mapped_axis = index_axes.index(0)
                        return mapped_axis
    return None


def reduce_vmap_parent(p: InfixRV, axis: int | None) -> tuple[InfixRV, int | None]:
    """
    Given an RV and an axis it will be vmapped over, give another RV and another axis that yields equivalent results
    but is (possibly) simpler

    Parameters
    ----------
    p
        Random variable that will be mapped over
    axis
        axis to be mapped over

    Examples
    --------
    >>> a = ir.RV(ir.Constant([1.1, 2.2, 3.3]))
    >>> b = ir.RV(ir.Constant([0, 1, 2]))
    >>> c = ir.RV(ir.VMap(ir.Index(), [None, 0], 3), a, b)
    >>> d, axis = reduce_vmap_parent(c, 0)
    >>> d == a
    True
    >>> axis
    0

    >>> a = ir.RV(ir.Constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]]))
    >>> b = ir.RV(ir.Constant([0, 1]))
    >>> c = ir.RV(ir.Constant([0, 1, 2]))
    >>> d = ir.RV(ir.VMap(ir.Index(), [None, 0, None], 2), a, b, c)
    >>> e, axis = reduce_vmap_parent(d, 0)
    >>> e == a
    True
    >>> axis
    0

    >>> f = ir.RV(ir.VMap(ir.Index(), [None, None, 0], 3), a, b, c)
    >>> g, axis = reduce_vmap_parent(f, 0)
    >>> g == a
    True
    >>> axis
    1

    """

    if axis != 0:
        return p, axis

    new_axis = is_vmap_over_scalar_index(p)
    if new_axis is None:
        return p, axis

    return p.parents[0], new_axis


def reduce_vmap_parents(parents, in_axes):
    return util.unzip([reduce_vmap_parent(p, i) for (p, i) in zip(parents, in_axes, strict=True)])


def vmap_subgraph(
    dummy_roots: list[InfixRV],
    dummy_nodes: list[InfixRV],
    dummy_outputs: list[InfixRV],
    roots: list[InfixRV],
    roots_axes: list[int | None],
    axis_size: int | None,
) -> list[InfixRV]:
    """
    Takes a graph of "dummy" RVs that represent some non-vmapped computation, then creates a parallel graph of "real" RVs that represent a vmapped computation


    Parameters
    ----------
    dummy_roots
        Root notes for non-vmapped graph.
    dummy_nodes
        Rest of nodes for non-vmapped graph
    dummy_outputs
        Output nodes for non-vmapped graph (must be in dummy_node)
    roots
        Root notes for the desired vmapped graph
    roots_axes
        the axes along which the roots should be vectorized
    axis_size
        the axis size for all mapped nodes (optional unless no args vmapped)

    Returns
    -------
    real_outputs
        vmapped nodes corresponding to `dummy_outputs`, but with everything vectorized

    Examples
    --------
    >>> a_dummy = InfixRV(AbstractOp())
    >>> b_dummy = InfixRV(AbstractOp())
    >>> c_dummy = InfixRV(ir.Add(), a_dummy, b_dummy)
    >>> a = InfixRV(Constant([0, 1, 2]))
    >>> b = InfixRV(Constant([4, 5, 6]))
    >>> [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], 3)
    >>> print(str(c_dummy))
    add(abstract_op, abstract_op)
    >>> print(str(c))
    vmap(add, [0, 0], 3)([0 1 2], [4 5 6])
    >>> print(repr(c_dummy))
    InfixRV(Add(), InfixRV(AbstractOp()), InfixRV(AbstractOp()))
    >>> print(repr(c))
    InfixRV(VMap(Add(), [0, 0], 3), InfixRV(Constant([0,1,2])), InfixRV(Constant([4,5,6])))

    >>> d_dummy = InfixRV(ir.Mul(), a_dummy, c_dummy)
    >>> print_upstream(d_dummy)
    shape | statement
    ----- | ---------
    ()    | a = abstract_op
    ()    | b = abstract_op
    ()    | c = add(a, b)
    ()    | d = mul(a, c)
    >>> [d] = vmap_subgraph([a_dummy, b_dummy], [c_dummy, d_dummy], [d_dummy], [a, b], [0, 0], 3)
    >>> print_upstream(d)
    shape | statement
    ----- | ---------
    (3,)  | a = [0 1 2]
    (3,)  | b = [4 5 6]
    (3,)  | c = vmap(add, [0, 0], 3)(a, b)
    (3,)  | d = vmap(mul, [0, 0], 3)(a, c)
    """
    # TODO: Should we allow axis_size=None here?

    if any(a in dummy_roots for a in dummy_nodes):
        raise ValueError("dummy_roots cannot be included in dummy_nodes")

    if not all(a in dummy_nodes for a in dummy_outputs):
        raise ValueError("dummy_outputs must all be included in dummy_nodes")

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

        # if no mapped axes AND non-random AND not in output, no need to map
        no_vmap_needed = no_mapped_axes and not dummy_node.op.random and dummy_node not in dummy_outputs
        if no_vmap_needed:
            new_op = dummy_node.op
            new_axis = None

            dummy_to_real[dummy_node] = create_rv(new_op, *parents)
            dummy_mapped_axis[dummy_node] = new_axis

        else:
            # print("")
            # print(f"A {parents=}")
            # print(f"A {my_in_axes=}")
            parents, my_in_axes = reduce_vmap_parents(parents, my_in_axes)
            # print(f"B {parents=}")
            # print(f"B {my_in_axes=}")

            new_op = VMap(dummy_node.op, in_axes=my_in_axes, axis_size=axis_size)
            new_axis = 0

            new_rv = create_rv(new_op, *parents)
            simple_rv, axis = reduce_vmap_parent(new_rv, 0)
            if axis == 0:
                new_rv = simple_rv

            dummy_to_real[dummy_node] = new_rv
            dummy_mapped_axis[dummy_node] = 0

            # dummy_to_real[dummy_node] = create_rv(new_op, *parents)
            # dummy_mapped_axis[dummy_node] = new_axis

    real_nodes = [dummy_to_real[dummy_node] for dummy_node in dummy_outputs]
    return real_nodes


def vmap_dummy_args(
    args: list[InfixRV], in_axes: Sequence[int | None], axis_size: int | None
) -> tuple[list[InfixRV[AbstractOp]], int]:
    """
    Given a "full" arguments, get a list of dummy/sliced arguments.

    Parameters
    ----------
    args
        Sequence of RVs for which sliced "dummies" are required.
    in_axes
        What axis to map each argument over. Should have same length as `args`.
    axis_size
        Anticipated axis size (or None if it should be inferred)

    Returns
    -------
    dummy_args
        abstract RVs with sliced shapes
    axis_size
        inferred axis_size (or arg if provided)

    Examples
    --------
    >>> A = InfixRV(Constant([[0,1,2],[4,5,6]]))
    >>> B = InfixRV(Constant([7,8,9]))
    >>> dummy_args, axis_size = vmap_dummy_args([A, B], [1, 0], None)
    >>> dummy_args
    [InfixRV(AbstractOp((2,))), InfixRV(AbstractOp())]
    """

    in_axes = list(in_axes)

    # TODO: this formerly existed. why??
    # if not util.all_unique(args):
    #    raise ValueError("vmap_dummy_args requires all unique arguments")

    dummy_args = []
    for i, a in zip(in_axes, args, strict=True):
        new_shape, new_axis_size = split_shape(a.shape, i)

        # once upon a time we did this—but don't remember the point of it
        # if isinstance(a.op, VMap) and i == 0:
        #     new_op = a.op.base_op  # why do we care?
        # else:
        #     new_op = AbstractOp(new_shape, a.op.random)

        new_op = AbstractOp(new_shape)
        my_dummy = create_rv(new_op)  # no parents!

        dummy_args.append(my_dummy)
        if axis_size is None:
            axis_size = new_axis_size
        elif new_axis_size is not None:
            assert axis_size == new_axis_size, "incoherent axis size"
    if axis_size is None:
        raise ValueError("axis_size could not be inferred")

    return dummy_args, axis_size


def vmap_eval_flat(
    f: FlatCallable,
    in_axes: Sequence[int | None],
    axis_size: int | None,
    args: list[RVLike],
) -> list[InfixRV]:
    """
    This function (but not vmap itself) works on "flat" function f, meaning that each
    argument of the function is just a RV. And the function must return
    a list of arguments which again are each just a RV.

    Parameters
    ----------
    f: FlatCallable
        The function to be vmapped. Must be "flat"
    in_axes
        axes to vmap for each argument
    axis_size
        length of vmap.
    args
        arguments

    Returns
    -------
    vmapped_outputs
        The result of the vmap

    Examples
    --------
    >>> @pack
    ... def f(a, b):
    ...     return [a+b]
    >>> A = constant([0,1])
    >>> B = constant([2,3])
    >>> [C] = vmap_eval_flat(f, (0,0), 2, [A, B])
    >>> print(repr(C))
    InfixRV(VMap(Add(), [0, 0], 2), InfixRV(Constant([0,1])), InfixRV(Constant([2,3])))
    """

    # make sure inputs are RVs
    in_axes = list(in_axes)
    rv_args = [makerv(a) for a in args]
    dummy_args, axis_size = vmap_dummy_args(rv_args, in_axes, axis_size)
    dummy_nodes, dummy_outputs = generated_nodes(f, dummy_args)

    return vmap_subgraph(dummy_args, dummy_nodes, dummy_outputs, rv_args, in_axes, axis_size)


def get_dummy_args(in_axes, args) -> tuple[list[int], list[InfixRV]]:
    """Converts PyTree args and axes to flat args and axes

    Parameters
    ----------
    in_axes
        PyTree of input axes
    args
        PyTree of inputs

    Returns
    -------
    flat_in_axes
        Flat input axes
    flat_args
        flat args

    Examples
    --------
    >>> get_dummy_args(0, makerv([0,1,2]))
    InfixRV(AbstractOp())
    >>> get_dummy_args([0,0], [makerv([0,1,2]), makerv([3,4,5])])
    [InfixRV(AbstractOp()), InfixRV(AbstractOp())]
    >>> get_dummy_args(0, [makerv([0,1,2]), makerv([3,4,5])])
    [InfixRV(AbstractOp()), InfixRV(AbstractOp())]
    >>> get_dummy_args(0, {1:makerv([0,1,2])})
    {1: InfixRV(AbstractOp())}
    >>> get_dummy_args(0, {'cat':makerv([0,1,2])})
    {'cat': InfixRV(AbstractOp())}
    >>> get_dummy_args({'cat':0}, {'cat':makerv([0,1,2])})
    {'cat': InfixRV(AbstractOp())}
    >>> get_dummy_args({'cat':0, 'dog':None}, {'cat':makerv([0,1,2]), 'dog':makerv(3)})
    {'cat': InfixRV(AbstractOp()), 'dog': InfixRV(AbstractOp())}

    >>> A = constant([0,1,2])
    >>> B = constant([3,4,5])
    >>> x = {"dog": A, "cat": B}
    >>> in_axes = {"dog": 0, "cat": 0}
    >>> get_dummy_args(in_axes, x)
    {'cat': InfixRV(AbstractOp()), 'dog': InfixRV(AbstractOp())}
    """

    def get_dummy(i, x):
        if i is None:
            new_shape = x.shape
        else:
            lo, mid, hi = (x.shape[:i], x.shape[i], x.shape[i + 1 :])
            new_shape = lo + hi

        # In old code tried to preserve x.op when isinstance(x.op, VMap)
        op = AbstractOp(new_shape)
        return create_rv(op)

    dummy_args = util.tree_map_recurse_at_leaf(get_dummy, in_axes, args, is_leaf=util._is_leaf_with_none)

    return dummy_args


# RVCallable = Callable[[PyTree[InfixRV]], PyTree[InfixRV]]


# def vmap(f: Callable, in_axes: Any = 0, axis_size: int | None = None) -> Callable:
def vmap[*Args](
    f: Callable[[*Args], PyTree[InfixRV]], in_axes: Any = 0, axis_size: int | None = None
) -> Callable[[*Args], PyTree[InfixRV]]:
    """
    Vectorizing map. Create a function which maps ``f`` over argument axes.

    This function matches exactly the interface of `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_, although it doesn't provide some of the optional arguments ``jax.vmap`` does.

    Parameters
    ----------
    f
        The function to vmap. Should take a pytree of `RV` as inputs and return a pytree of `RV` as outputs.
    in_axes
        An int, None, or pytree with roots that are int or None. Specifies which axis of each RV should be mapped (if int) or that no axis shuld be mapped (if None). Can be a pytree matching the structure of all arguments to ``f``. Or, can be a pytree that is a prefix to the pytree representing all arguments. By default, in_axes is zero, meaning all RVs are mapped over the first axis.
    axis_size
        An integer indicating the size of the axis to be mapped. This is optional unless all leaves of ``in_axes`` are ``None``.

    Returns
    -------
    vec_f
        batched/vectorized version of ``f`` with arguments matching those of ``f`` with extra axes at positions indicated by ``in_axes`` and a return value that corresponds to that of ``f`` but with an extra axis in the first position.

    Examples
    --------
    Here's the simplest possible example.

    >>> def fun(a):
    ...     return exp(a)
    >>> A = constant([0,1,2])
    >>> vmap(fun)(A)
    InfixRV(VMap(Exp(), [0], 3), InfixRV(Constant([0,1,2])))

    Multiple inputs are OK.

    >>> def fun(a,b):
    ...     return a*b
    >>> A = constant([1,2,3])
    >>> B = constant([4,5,6])
    >>> vmap(fun)(A, B)
    InfixRV(VMap(Mul(), [0, 0], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant([4,5,6])))

    Unmapped inputs are OK.

    >>> def fun(a,b):
    ...     return a*b
    >>> A = constant([1,2,3])
    >>> vmap(fun, [0, None])(A, constant(7))
    InfixRV(VMap(Mul(), [0, None], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant(7)))

    Multiple outputs are OK.

    >>> def fun(a):
    ...     return [exp(a), log(a)]
    >>> [out1, out2] = vmap(fun)(A)
    >>> out1
    InfixRV(VMap(Exp(), [0], 3), InfixRV(Constant([1,2,3])))
    >>> out2
    InfixRV(VMap(Log(), [0], 3), InfixRV(Constant([1,2,3])))

    Pytree inputs and pytree in_axes are OK

    >>> def fun(x):
    ...     return x['cat']*x['dog']
    >>> x = {'cat': A, 'dog': constant(3)}
    >>> in_axes = {'cat': 0, 'dog': None}
    >>> vmap(fun, in_axes)(x)
    InfixRV(VMap(Mul(), [0, None], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant(3)))

    Pytree outputs are OK

    >>> def fun(a, b):
    ...     return {"add": a+b, "mul": a*b}
    >>> vmap(fun)(A, B)
    {'add': InfixRV(VMap(Add(), [0, 0], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant([4,5,6]))), 'mul': InfixRV(VMap(Mul(), [0, 0], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant([4,5,6])))}

    Pytree in_axis prefixes are OK

    >>> def fun(x):
    ...     [a, (b,c)] = x
    ...     return (a*b)+c
    >>> x = [A, (constant(7), constant(8))]
    >>> in_axes1 = [0, (None, None)] # axes for each leaf
    >>> in_axes2 = [0, None]         # single None for (b,c) tuple!
    >>> out1 = vmap(fun, in_axes1)(x)
    >>> out2 = vmap(fun, in_axes2)(x)
    >>> out1.op == out2.op
    True
    >>> print_upstream(out1)
    shape | statement
    ----- | ---------
    (3,)  | a = [1 2 3]
    ()    | b = 7
    (3,)  | c = vmap(mul, [0, None], 3)(a, b)
    ()    | d = 8
    (3,)  | e = vmap(add, [0, None], 3)(c, d)
    >>> print_upstream(out2)
    shape | statement
    ----- | ---------
    (3,)  | a = [1 2 3]
    ()    | b = 7
    (3,)  | c = vmap(mul, [0, None], 3)(a, b)
    ()    | d = 8
    (3,)  | e = vmap(add, [0, None], 3)(c, d)

    See Also
    --------
    pangolin.ir.VMap
    """

    # TODO: support negative in_axes

    def call(*args):
        if len(args) == 1:
            # handles vmap(f, 0)(x) instead vmap(f,(0,))(x)
            my_in_axes = (in_axes,)
        elif isinstance(in_axes, list):
            # handles vmap(f, [0,1])(x,y) instead of vmap(f,(0,1))(x,y)
            my_in_axes = tuple(in_axes)
        else:
            my_in_axes = in_axes

        # no greedy casting because this leads to ambiguity
        # if the user sends [(1,2),(3,4)] is that a list of two
        # arrays?
        args = jax.tree_util.tree_map(makerv, args)

        dummy_args = get_dummy_args(my_in_axes, args)

        flat_in_axes, flat_args = util.dual_flatten(my_in_axes, args)

        flat_f, flatten_inputs, unflatten_output = util.flatten_fun(f, *dummy_args, is_leaf=util._is_leaf_with_none)
        flat_args = flatten_inputs(*args)
        flat_output = vmap_eval_flat(flat_f, flat_in_axes, axis_size, flat_args)
        output = unflatten_output(flat_output)
        return output

    return call


def vmap_flat(f: FlatCallable, in_axes: Sequence[int | None], axis_size: int | None):
    """
    vmap a flat function (one that takes some number of RV arguments and returns a list of RV
    arguments)
    """
    in_axes = list(in_axes)

    def vec_f(args: list[RVLike]):
        rv_args: list[InfixRV] = list(makerv(a) for a in args)
        dummy_args, my_axis_size = vmap_dummy_args(rv_args, in_axes, axis_size)
        dummy_nodes, dummy_outputs = generated_nodes(f, dummy_args)
        return vmap_subgraph(dummy_args, dummy_nodes, dummy_outputs, rv_args, in_axes, my_axis_size)

    return vec_f


# TODO: delete?


def convert_args(rv_type: Type[InfixRV], *args: InfixRV):
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
        new_parents: list[InfixRV] = [abstract_args[p] if p in abstract_args else p for p in a.parents]
        abstract_a = rv_type(a.op, *new_parents)
        abstract_args[a] = abstract_a
    return tuple(abstract_args[a] for a in args)
