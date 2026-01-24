"""
The idea of this (sub)-module is to support a notation like this

i = Index(5)
j = Index(10)
x = Field()
x[i, j] = a[i] * b[i,j] * c[j]

i = Index(5)
j = Index(10)
x = Field()
x[i, j, :] = a[i, :] * b[i,j] * c[j, :]

x = field(
    (5,10),
    lambda i, j : a[i] * b[i,j] * c[j]
)
x = field(
    (5,10),
    lambda i, j : a[i, :] * b[i,j] * c[j, :]
)

Or, to combine with autoregressive

i = Index(5)
j = Index(10)
x = Field()
x[start,j] = a[j]
x[i,j] = x[i-1,j] + b[i,j]

x = field(
    (5, 10),
    rule = lambda i, j, x: x[i-1,j] + B[i,j]
    init = lambda i, j: a[j]
)

Strongly considering to change name to auto

z = auto.vmap @ lambda i, j: x[i] + y[j]

z = auto.scan(lambda i,j: x[i] + y[j]) @ lambda i, j, z: x[i] + y[j] + z[i-1,j]

Could have some notation for declaring sizes

I think we need to forbid complex things like
    def fun(i, j):
        a = x[i,j,:]
        return a[i]

- What we need is an extract_inputs(fun, num_dims) function
    - First, create dummy scalars for each input dim.
    - Then, run the function on those dummy scalars
    - Then do a graph search from the outputs and find the original inputs and which dimensions are getting indexed with each dummy scalar.
    - Make sure that the dummy scalars are only referenced at the "top" of the graph
    - After that, it's all pretty easy?


Question: Should this also support vindex?
- I think the answer is: That question doesn't make sense
- We require all indices to either be scalars (i,j) or full slices
- So no actual indexing takes place. It just "looks" like indexing.
"""

from .base import InfixRV, AbstractOp, generated_nodes, constant, vmap_subgraph, vmap, print_upstream
from pangolin import ir
from typing import Sequence, Callable
import warnings
import inspect
from pangolin import dag

# How to proceed:
#
# The main problem is: Given a function like
#
# fun = lambda i, j: solve(a[i,:,:]+1, 2*b[:,i,j])
#
# How do we get:
# 1) The arrays a and b being referenced
# 2) The indices where a and b are being indexed with i / j
# 3) dummy_fun = lambda a, b: solve(a+1, 2*b)
#
# If we had those things, then it would be easy to call vmap to do the actual work
#
# So how do we do that?
#
# - Create dummy scalar variables dummy_i, dummy_j
# - Call the function on those dummy scalar variables
# - Do a graph search on the output of the function to find all the "roots" (variables that are indexed by the dummies)


def get_positional_count(func: Callable) -> int:
    """Counts the number of arguments that can be passed positionally.

    Args:
        func: The function to inspect.

    Returns:
        int: The total number of positional-capable parameters.

    Raises:
        ValueError: If the function defines any keyword-only arguments.

    Examples:
        >>> # Standard arguments
        >>> def test_a(x, y): pass
        >>> get_positional_count(test_a)
        2

        >>> # Arguments with defaults
        >>> def test_b(x, y=10, z=20): pass
        >>> get_positional_count(test_b)
        3

        >>> # Positional-only (using /)
        >>> def test_c(a, b, /, c): pass
        >>> get_positional_count(test_c)
        3

        >>> # No arguments
        >>> def test_d(): pass
        >>> get_positional_count(test_d)
        0

        >>> # Including *args (does not count toward the total)
        >>> def test_e(a, b, *args): pass
        >>> get_positional_count(test_e)
        2

        >>> # Error: Keyword-only argument (after *)
        >>> def test_f(a, *, b): pass
        >>> get_positional_count(test_f)
        Traceback (most recent call last):
            ...
        ValueError: Function contains keyword-only arguments.

        >>> # Error: Keyword-only argument (after *args)
        >>> def test_g(a, *args, b): pass
        >>> get_positional_count(test_g)
        Traceback (most recent call last):
            ...
        ValueError: Function contains keyword-only arguments.

        >>> # Built-in functions (that don't use keyword-only)
        >>> get_positional_count(len)
        1
    """
    sig = inspect.signature(func)
    params = sig.parameters.values()

    if any(p.kind == p.KEYWORD_ONLY for p in params):
        raise ValueError("Function contains keyword-only arguments.")

    return sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))


def find_indices(nodes: Sequence[InfixRV], indices: Sequence[InfixRV[AbstractOp]]):
    return [nodes.index(idx) if idx in nodes else None for idx in indices]


def extract_from_rvs(dummy_indices: Sequence[InfixRV], output_rv: InfixRV):
    """
    Examples
    --------

    Single array, one index

    >>> a = constant([1,2,3])
    >>> i = constant(0)
    >>> out = a[i] + 2.2
    >>> arrays, dims, replay = extract_from_rvs([i], out)
    >>> arrays == [a]
    True
    >>> dims == [[0]]
    True
    >>> replay(constant(3.3))
    InfixRV(Add(), InfixRV(Constant(3.3)), InfixRV(Constant(2.2)))

    Two arrays, two indices

    >>> a = constant([1,2,3])
    >>> b = constant([4,5,6])
    >>> i = constant(0)
    >>> j = constant(0)
    >>> out = a[i] * b[j]
    >>> arrays, dims, replay = extract_from_rvs([i, j], out)
    >>> arrays == [a, b]
    True
    >>> dims == [[0, None], [None, 0]]
    True
    >>> replay(constant(1.1), constant(2.2))
    InfixRV(Mul(), InfixRV(Constant(1.1)), InfixRV(Constant(2.2)))
    """

    n_before_indices = min(idx._n for idx in dummy_indices)

    def is_abstract(rv: InfixRV) -> bool:
        return rv._n >= n_before_indices

    def not_abstract(var: InfixRV):
        return not is_abstract(var)

    all_rvs = dag.upstream_nodes(output_rv, node_block=not_abstract)

    indexed_rvs = []
    indexed_dims = []
    for rv in all_rvs:
        if isinstance(rv.op, ir.Index):
            if rv.parents[0] in dummy_indices:
                raise ValueError("Dummy index itself is indexed")

            where_dummies = find_indices(rv.parents[1:], dummy_indices)

            if any(i is not None for i in where_dummies):
                indexed_rvs.append(rv.parents[0])
                indexed_dims.append(where_dummies)

    def replay(*indexed_rvs_replayed):
        rv_to_replayed = {}

        n = 0
        for rv in all_rvs:
            if isinstance(rv.op, ir.Index):
                where_dummies = find_indices(rv.parents[1:], dummy_indices)

                if any(i is not None for i in where_dummies):
                    rv_to_replayed[rv] = indexed_rvs_replayed[n]
                    n += 1
                    continue

            replayed_rv = InfixRV(rv.op, *[rv_to_replayed[p] for p in rv.parents])
            rv_to_replayed[rv] = replayed_rv

        return rv_to_replayed[output_rv]

    return indexed_rvs, indexed_dims, replay


def extract_from_trace(dummy_indices, all_rvs, output_rv):
    indexed_rvs = []
    indexed_dims = []
    for rv in all_rvs:
        if isinstance(rv.op, ir.Index):
            if rv.parents[0] in dummy_indices:
                raise ValueError("Dummy index itself is indexed")

            where_dummies = find_indices(rv.parents[1:], dummy_indices)

            if any(i is not None for i in where_dummies):
                indexed_rvs.append(rv.parents[0])
                indexed_dims.append(where_dummies)

    def replay(*indexed_rvs_replayed):
        rv_to_replayed = {}

        n = 0
        for rv in all_rvs:
            if isinstance(rv.op, ir.Index):
                where_dummies = find_indices(rv.parents[1:], dummy_indices)

                if any(i is not None for i in where_dummies):
                    rv_to_replayed[rv] = indexed_rvs_replayed[n]
                    n += 1
                    continue

            replayed_rv = InfixRV(rv.op, *[rv_to_replayed[p] for p in rv.parents])
            rv_to_replayed[rv] = replayed_rv

        return rv_to_replayed[output_rv]

    return indexed_rvs, indexed_dims, replay


def extract_inputs(fun: Callable, num_inputs: int | None = None):
    """
    Given a fun involving dummy indexing, get the arrays being indexed and the positions of each dummy index

    Args:
        fun: A function taking (integer) indices and returning a single RV
        num_inputs: the number of indices (optional)

    Note:
    - The same array may appear multiple times in the output

    Examples
    --------

    Simple single array, one index

    >>> a = constant([1,2,3])
    >>> fun = lambda i: a[i] + 2.2
    >>> arrays, dims, replay = extract_inputs(fun)
    >>> arrays
    [InfixRV(Constant([1,2,3]))]
    >>> dims
    [[0]]
    >>> replay(constant(3.3))
    InfixRV(Add(), InfixRV(Constant(3.3)), InfixRV(Constant(2.2)))

    Two indices with single array

    >>> a = constant([[1,2,3],[4,5,6]])
    >>> fun = lambda i, j: a[j,i] + 2.2
    >>> arrays, dims, replay = extract_inputs(fun)
    >>> arrays == [a]
    True
    >>> dims
    [[1, 0]]
    >>> replay(constant(3.3))
    InfixRV(Add(), InfixRV(Constant(3.3)), InfixRV(Constant(2.2)))

    Single array indexed twice

    >>> a = constant([1,2,3])
    >>> fun = lambda i: a[i] * a[i]
    >>> arrays, dims, replay = extract_inputs(fun)
    >>> arrays == [a, a]
    True
    >>> dims
    [[0], [0]]
    >>> replay(constant(3.3), constant(3.3))
    InfixRV(Mul(), InfixRV(Constant(3.3)), InfixRV(Constant(3.3)))

    Different indices

    >>> a = constant([1,2,3])
    >>> b = constant([4,5,6])
    >>> fun = lambda i, j: a[i] * b[j]
    >>> arrays, dims, replay = extract_inputs(fun)
    >>> arrays == [a, b]
    True
    >>> dims
    [[0, None], [None, 0]]
    >>> replay(constant(3.3), constant(4.4))
    InfixRV(Mul(), InfixRV(Constant(3.3)), InfixRV(Constant(4.4)))

    Slice first dim of a

    >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> b = constant([7.7, 8.8])
    >>> fun = lambda i: a[i,:] * b[i]
    >>> arrays, dims, replay = extract_inputs(fun)
    >>> arrays == [a, b]
    True
    >>> dims == [[0], [0]]
    True
    >>> print(replay(constant([1.1, 2.2, 3.3]), constant(4.4)))
    vmap(mul, [0, None], 3)([1.1 2.2 3.3], 4.4)

    Slice the second dimension of ``a`` and first dimension of ``b``

    >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> b = constant([[7.7, 8.8],[9.9, 10.10],[11.11, 12.12]])
    >>> fun = lambda i: a[:,i] @ b[i,:]
    >>> arrays, dims, replay = extract_inputs(fun)
    >>> arrays == [a,b]
    True
    >>> dims == [[1],[0]]
    True
    >>> print(replay(constant([1.1, 4.4]), constant([7.7, 8.8])))
    matmul([1.1 4.4], [7.7 8.8])

    Slice the second dimension of ``a`` and first dimension of ``b``, different indices

    >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> b = constant([[7.7, 8.8],[9.9, 10.10],[11.11, 12.12]])
    >>> fun = lambda i, j: a[:,i] @ b[j,:]
    >>> arrays, dims, replay = extract_inputs(fun)
    >>> arrays == [a,b]
    True
    >>> dims == [[1, None],[None, 0]]
    True
    >>> print(replay(constant([1.1, 4.4]), constant([7.7, 8.8])))
    matmul([1.1 4.4], [7.7 8.8])
    """

    if num_inputs is None:
        num_inputs = get_positional_count(fun)

    dummy_indices = [InfixRV(AbstractOp(())) for _ in range(num_inputs)]

    # flat_fun = lambda *args: [fun(*args)]
    # all_rvs, [output_rv] = generated_nodes(flat_fun, *dummy_indices)
    # return extract_from_trace(dummy_indices, all_rvs, output_rv)

    output_rv = fun(*dummy_indices)
    return extract_from_rvs(dummy_indices, output_rv)


def vfor(fun, size: None | Sequence[int] = None):
    """
    Examples
    --------

    Mutiply each element by two.

    >>> x = constant([1.1, 2.2, 3.3])
    >>> y = vfor(lambda i: x[i]*2, [3])
    >>> print(y)
    vmap(mul, [0, None], 3)([1.1 2.2 3.3], 2)

    Two indices with single array.

    >>> x = constant([[1,2,3],[4,5,6]])
    >>> y = vfor(lambda i, j: x[j,i] + 2.2)
    >>> print(y)
    vmap(vmap(add, [0, None], 2), [1, None], 3)([[1 2 3] [4 5 6]], 2.2)

    Slice first dimension of a and only dimension of b.

    >>> a = constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    >>> b = constant([7.7, 8.8])
    >>> c = vfor(lambda i: a[i,:] * b[i])
    >>> print(c.op)
    vmap(vmap(mul, [0, None], 3), [0, 0], 2)
    >>> c.parents == (a, b)
    True

    Slice the second dimension of ``a`` and first dimension of ``b``, different indices

    >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> b = constant([[7.7, 8.8],[9.9, 10.10],[11.11, 12.12]])
    >>> c = vfor(lambda i, j: a[:,i] @ b[j,:])
    >>> print_upstream(c)
    shape  | statement
    ------ | ---------
    (2, 3) | a = [[1.1 2.2 3.3] [4.4 5.5 6.6]]
    (3, 2) | b = [[ 7.7  8.8 ] [ 9.9 10.1 ] [11.11 12.12]]
    (3, 3) | c = vmap(vmap(matmul, [None, 0], 3), [1, None], 3)(a, b)

    For technical reasons, returning an array without further processing will cause an error

    >>> x = constant([1.1, 2.2, 3.3])
    >>> y = vfor(lambda i: x[i])
    Traceback (most recent call last):
    ...
    ValueError: fun passed to generated_nodes cannot return input values
    """

    arrays, dims, replay = extract_inputs(fun)
    num_indices = len(dims[0])

    if size:
        if len(size) != num_indices:
            raise ValueError(f"Given size {size} has length different from num_indices {num_indices}")

    new_fun = replay

    for idx in range(num_indices - 1, -1, -1):
        in_axes = tuple(dim[idx] for dim in dims)
        if size:
            axis_size = size[idx]
        else:
            axis_size = None

        if len(arrays) == 1:
            new_fun = vmap(new_fun, in_axes[0], axis_size)
        else:
            new_fun = vmap(new_fun, in_axes, axis_size)
    return new_fun(*arrays)


# TODO: ensure no recursive indexing
# TODO: ensure all other dimensions are full slices

# def field1(fun, axis_size: int | None = None):
#     """
#     Examples
#     --------

#     Mutiply each element by two

#     >>> x = constant([1.1, 2.2, 3.3])
#     >>> y = field1(lambda i: x[i]*2)
#     >>> print(y)
#     vmap(mul, [0, None], 3)([1.1 2.2 3.3], 2)
#     >>> y.parents[0] == x
#     True
#     >>> y.parents[1].op == ir.Constant(2)
#     True

#     Slice the first dimension of ``a``

#     >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
#     >>> b = constant([7.7, 8.8])
#     >>> c = field1(lambda i: a[i,:] * b[i])
#     >>> ir.print_upstream(c)
#     shape  | statement
#     ------ | ---------
#     (2, 3) | a = [[1.1 2.2 3.3] [4.4 5.5 6.6]]
#     (2,)   | b = [7.7 8.8]
#     (2, 3) | c = vmap(vmap(mul, [0, None], 3), [0, 0], 2)(a, b)

#     Slice the second dimension of ``a`` and first dimension of ``b``

#     >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
#     >>> b = constant([[7.7, 8.8],[9.9, 10.10],[11.11, 12.12]])
#     >>> c = field1(lambda i: a[:,i] + b[i,:])
#     >>> ir.print_upstream(c)
#     shape  | statement
#     ------ | ---------
#     (2, 3) | a = [[1.1 2.2 3.3] [4.4 5.5 6.6]]
#     (3, 2) | b = [[ 7.7  8.8 ] [ 9.9 10.1 ] [11.11 12.12]]
#     (3, 2) | c = vmap(vmap(add, [0, 0], 2), [1, 0], 3)(a, b)

#     Slice the second dimension of ``a`` and first dimension of ``b``

#     >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
#     >>> b = constant([[7.7, 8.8],[9.9, 10.10],[11.11, 12.12]])
#     >>> c = field1(lambda i: a[:,i] @ b[i,:])
#     >>> ir.print_upstream(c)
#     shape  | statement
#     ------ | ---------
#     (2, 3) | a = [[1.1 2.2 3.3] [4.4 5.5 6.6]]
#     (3, 2) | b = [[ 7.7  8.8 ] [ 9.9 10.1 ] [11.11 12.12]]
#     (3,)   | c = vmap(matmul, [1, 0], 3)(a, b)


#     For technical reasons, returning an array without further processing will cause an error

#     >>> x = constant([1.1, 2.2, 3.3])
#     >>> y = field1(lambda i: x[i])
#     Traceback (most recent call last):
#     ...
#     ValueError: Output not in generated nodes. Did you perhaps return an input array?
#     """
#     dummy_index = InfixRV(AbstractOp(()))

#     def is_root(rv):
#         if isinstance(rv.op, ir.Index):
#             if rv.parents[0] is dummy_index:
#                 raise ValueError("Dummy index itself is indexed")
#             if dummy_index in rv.parents[1:]:
#                 return True
#         return False

#     flat_fun = lambda i: [fun(i)]
#     all_rvs, [dummy_output] = generated_nodes(flat_fun, dummy_index)

#     dummy_roots = []
#     dummy_nodes = []
#     roots = []
#     roots_axes = []

#     for rv in all_rvs:
#         if is_root(rv):
#             for p in rv.parents:
#                 if p in dummy_roots:
#                     print(f"parent {p=} found in dummy_roots")

#                 # if p in dummy_nodes:
#                 #     print(f"parent {p=} found in dummy_nodes")

#             # assert not any(p in dummy_roots + dummy_nodes for p in rv.parents[1:])

#             root = rv.parents[0]
#             root_axis = rv.parents[1:].index(dummy_index)

#             if axis_size is None:
#                 axis_size = root.shape[root_axis]

#             if root.shape[root_axis] != axis_size:
#                 raise ValueError(
#                     f"var mapped over axis with shape {root.shape[root_axis]} but axis size was {axis_size}"
#                 )

#             dummy_roots.append(rv)
#             roots.append(root)
#             roots_axes.append(root_axis)
#         else:
#             dummy_nodes.append(rv)

#     if dummy_output not in dummy_nodes:
#         raise ValueError("Output not in generated nodes. Did you perhaps return an input array?")

#     [out] = vmap_subgraph(dummy_roots, dummy_nodes, [dummy_output], roots, roots_axes, axis_size)
#     return out


# def field(fun, axis_sizes: ir.Shape):
#     """
#     Examples
#     --------
#     >>> A = constant([[1,2,3],[4,5,6]])
#     >>> B = constant([7,8,9])
#     >>> field(lambda i,j: A[i,j] * B[j], (2,3))
#     """

#     dummy_indices = [InfixRV(AbstractOp(())) for _ in axis_sizes]

#     def is_root(rv):
#         if isinstance(rv.op, ir.Index):
#             if rv.parents[0] in dummy_indices:
#                 raise ValueError("Dummy index itself is indexed")
#             if any(idx in rv.parents[1:] for idx in dummy_indices):
#                 return True
#         return False

#     flat_fun = lambda *args: [fun(*args)]
#     all_rvs, [dummy_output] = generated_nodes(flat_fun, *dummy_indices)

#     for rv in all_rvs:
#         print(rv)
#     print(dummy_output)
#     ir.print_upstream(dummy_output)

#     dummy_roots = []
#     dummy_nodes = []
#     roots = []
#     roots_axes = []

#     for rv in all_rvs:
#         if is_root(rv):
#             for p in rv.parents:
#                 if p in dummy_roots:
#                     print(f"parent {p=} found in dummy_roots")

#             root = rv.parents[0]
#             root_axes = [rv.parents[1:].index(d) if d in rv.parents[1:] else None for d in dummy_indices]

#             for root_axis, axis_size in zip(root_axes, axis_sizes, strict=True):
#                 if root_axis is not None and root.shape[root_axis] != axis_size:
#                     raise ValueError(
#                         f"var mapped over axis with shape {root.shape[root_axis]} but axis size was {axis_size}"
#                     )

#             dummy_roots.append(rv)
#             roots.append(root)
#             roots_axes.append(root_axes)
#         else:
#             dummy_nodes.append(rv)

#     if dummy_output not in dummy_nodes:
#         raise ValueError("Output not in generated nodes. Did you perhaps return an input array?")

#     # [out] = vmap_subgraph(dummy_roots, dummy_nodes, [dummy_output], roots, roots_axes, axis_size)
#     # return out
