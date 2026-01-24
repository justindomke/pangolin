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
from pangolin import dag, util
from jaxtyping import PyTree
import jax.tree_util


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


def extract_from_rvs(dummy_indices: Sequence[InfixRV], output: PyTree[InfixRV]):
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

    Two arrays, two indices, two outputs in a pytree

    >>> a = constant([1,2,3])
    >>> b = constant([4,5,6])
    >>> i = constant(0)
    >>> j = constant(0)
    >>> out = {'alice': a[i] * b[j], 'bob': a[j] + b[i]}
    >>> arrays, dims, replay = extract_from_rvs([i, j], out)
    >>> arrays == [a, b, a, b]
    True
    >>> dims == [[0, None], [None, 0], [None, 0], [0, None]]
    True
    >>> replay(constant(1.1), constant(2.2), constant(1.1), constant(2.2))
    {'alice': InfixRV(Mul(), InfixRV(Constant(1.1)), InfixRV(Constant(2.2))), 'bob': InfixRV(Add(), InfixRV(Constant(1.1)), InfixRV(Constant(2.2)))}
    """

    n_before_indices = min(idx._n for idx in dummy_indices)

    def is_abstract(rv: InfixRV) -> bool:
        return rv._n >= n_before_indices

    def not_abstract(var: InfixRV):
        return not is_abstract(var)

    output_rvs, output_treedef = jax.tree_util.tree_flatten(output)

    all_rvs = dag.upstream_nodes(output_rvs, node_block=not_abstract)

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

        results_flat = [rv_to_replayed[out_rv] for out_rv in output_rvs]

        return jax.tree_util.tree_unflatten(output_treedef, results_flat)

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

    num_indices = get_positional_count(fun)

    if size:
        if len(size) != num_indices:
            raise ValueError(f"Given size {size} has length different from num_indices {num_indices}")

    dummy_indices = [InfixRV(AbstractOp(())) for _ in range(num_indices)]
    output_rv = fun(*dummy_indices)
    arrays, dims, replay = extract_from_rvs(dummy_indices, output_rv)

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
