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

from .base import InfixRV, AbstractOp, generated_nodes, constant, vmap_subgraph
from pangolin import ir
from typing import Sequence
import warnings


def find_indices(nodes: Sequence[InfixRV], indices: Sequence[InfixRV[AbstractOp]]):
    return [nodes.index(idx) if idx in nodes else None for idx in indices]


def extract_inputs(fun, num_inputs):
    """
    Given a fun involving dummy indexing, get the arrays being indexed and the positions of each dummy index

    Note:
    - The same array may appear multiple times in the output

    Examples
    --------

    Simple single array, one index

    >>> a = constant([1,2,3])
    >>> fun = lambda i: a[i]
    >>> arrays, dims = extract_inputs(fun, 1)
    >>> arrays
    [InfixRV(Constant([1,2,3]))]
    >>> dims
    [[0]]

    Two indices with single array

    >>> a = constant([[1,2,3],[4,5,6]])
    >>> fun = lambda i, j: a[j,i]
    >>> arrays, dims = extract_inputs(fun, 2)
    >>> arrays == [a]
    True
    >>> dims
    [[1, 0]]

    Single array indexed twice

    >>> a = constant([1,2,3])
    >>> fun = lambda i: a[i] * a[i]
    >>> arrays, dims = extract_inputs(fun, 1)
    >>> arrays == [a, a]
    True
    >>> dims
    [[0], [0]]

    Different indices

    >>> a = constant([1,2,3])
    >>> b = constant([4,5,6])
    >>> fun = lambda i, j: a[i] * b[j]
    >>> arrays, dims = extract_inputs(fun, 2)
    >>> arrays == [a, b]
    True
    >>> dims
    [[0, None], [None, 0]]
    """

    flat_fun = lambda *args: [fun(*args)]

    dummy_indices = [InfixRV(AbstractOp(())) for _ in range(num_inputs)]
    all_rvs, outputs = generated_nodes(flat_fun, *dummy_indices)
    indexed_rvs = []
    indexed_dims = []
    for rv in all_rvs:
        if isinstance(rv.op, ir.Index):
            if rv.parents[0] in dummy_indices:
                raise ValueError("Dummy index itself is indexed")

            where_dummies = find_indices(rv.parents[1:], dummy_indices)

            if all(i is None for i in where_dummies):
                continue

            indexed_rvs.append(rv.parents[0])
            indexed_dims.append(where_dummies)
    return indexed_rvs, indexed_dims

    # dummy_roots
    #     Root notes for non-vmapped graph.
    # dummy_nodes
    #     Rest of nodes for non-vmapped graph
    # dummy_outputs
    #     Output nodes for non-vmapped graph (must be in dummy_node)
    # roots
    #     Root notes for the desired vmapped graph
    # roots_axes
    #     the axes along which the roots should be vectorized
    # axis_size
    #     the axis size for all mapped nodes (optional unless no args vmapped)


# TODO: ensure no recursive indexing
# TODO: ensure all other dimensions are full slices


def field1(axis_size: int, fun):
    """
    Examples
    --------

    Mutiply each element by two

    >>> x = constant([1.1, 2.2, 3.3])
    >>> y = field1(3, lambda i: x[i]*2)
    >>> print(y)
    vmap(mul, [0, None], 3)([1.1 2.2 3.3], 2)
    >>> y.parents[0] == x
    True
    >>> y.parents[1].op == ir.Constant(2)
    True

    >>> x = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> y = constant([7.7, 8.8])
    >>> z = field1(2, lambda i: x[i,:] * y[i])
    >>> ir.print_upstream(z)
    shape  | statement
    ------ | ---------
    (2, 3) | a = [[1.1 2.2 3.3] [4.4 5.5 6.6]]
    (2,)   | b = [7.7 8.8]
    (2, 3) | c = vmap(vmap(mul, [0, None], 3), [0, 0], 2)(a, b)

    For technical reasons, returning an array without further processing will cause an error

    >>> x = constant([1.1, 2.2, 3.3])
    >>> y = field1(3, lambda i: x[i])
    Traceback (most recent call last):
    ...
    ValueError: Output not in generated nodes. Did you perhaps return an input array?
    """
    dummy_index = InfixRV(AbstractOp(()))

    def is_root(rv):
        if isinstance(rv.op, ir.Index):
            if rv.parents[0] is dummy_index:
                raise ValueError("Dummy index itself is indexed")
            if dummy_index in rv.parents[1:]:
                return True
        return False

    flat_fun = lambda i: [fun(i)]
    all_rvs, [dummy_output] = generated_nodes(flat_fun, dummy_index)

    dummy_roots = []
    dummy_nodes = []
    roots = []
    roots_axes = []

    for rv in all_rvs:
        if is_root(rv):
            for p in rv.parents:
                if p in dummy_roots:
                    print(f"parent {p=} found in dummy_roots")

                # if p in dummy_nodes:
                #     print(f"parent {p=} found in dummy_nodes")

            # assert not any(p in dummy_roots + dummy_nodes for p in rv.parents[1:])

            root = rv.parents[0]
            root_axis = rv.parents[1:].index(dummy_index)

            assert root.shape[root_axis] == axis_size
            if root.shape[root_axis] != axis_size:
                raise ValueError(
                    f"var mapped over axis with shape {root.shape[root_axis]} but axis size was {axis_size}"
                )

            # new_shape, new_axis_size = ir.split_shape(root.shape, root_axis)
            # dummy_root = InfixRV(AbstractOp(new_shape))
            dummy_root = rv

            dummy_roots.append(dummy_root)
            roots.append(root)
            roots_axes.append(root_axis)
        else:
            dummy_nodes.append(rv)

    if dummy_output not in dummy_nodes:
        raise ValueError("Output not in generated nodes. Did you perhaps return an input array?")

    [out] = vmap_subgraph(dummy_roots, dummy_nodes, [dummy_output], roots, roots_axes, axis_size)
    return out
