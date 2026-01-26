"""
The idea of this (sub)-module is to support a notation like this:

X = Slot()
with Index() as i:
    with Index() as j:
        X[i,j,:] = A[i,:] * B[j,:]

or

X = vfor(lambda i,j:
    A[i,:] * B[j,:]
)

Instead of the mysterious:

X = A[:,None,:] * B[None,:,:]

Or this insanity:

X = vmap(vmap(lambda ai, bj: ai * bj, [0, None]), [None, 0])

We'd also like to support scans like this

X = Slot()
with Index() as i:
    X[i, _, :] = A[i, :]
    with Index() as j:
        X[i, j,:] = X[i, j.prev,:] + A[i, j, :]

So how to do this? The tricky part is that we could have multiple assignments inside of a single block, e.g.

X = Slot()
Y = Slot()
with Index() as i:
    tmp1 = expensive_fun(A[i], B[i])
    tmp2 = dist(tmp1, C[i])
    X[i] = tmp2 * 2
    Y[i] = other_dist(X[1], tmp2)

In this case it is absolutely necessary that the same tmp2 object is shared between X[i] and Y[i], otherwise results will not be correct.

It would appear that the only way to achieve that is to vmap X and Y together, in a single call. This means that the vmap can't come when the Slot is assigned, but somehow must when the Index context manager exits. But that is tricky! Because what if you do this?

X = Slot()
Y = Slot()
Z = Slot()
with Index as i:
    X[i] = ...
    with Index() as j:
        Y[i,j] = X[i] + A[j]
    Z[i] = X[i] + ...

Perhaps the rule is: Once *all* the Index context managers have exited for a given variable, THEN you can vmap it?

OK... But what about this?

X = Slot()
with Index as i:
    Z = Slot()
    with Index() as j:
        Z[j] = B[i] * A[j]

STAGE 1: Slots only take one Index. Only one Index can ever be created at a time.

X = Slot()
Y = Slot()
with Index() as i:
    X[i] = dist(A[i], B[i])
    Y[i] = X[i] * X[i]

This seems fine.

- When the context manager is entered, it records the current RV._n
- Everything inside the context manager is totally "normal"
- When a Slot is assigned, it just remembers the value assigned to it. It also places itself on the Index's list of assigned slots.
- When the context manager for an Index exists, everything inside is vmapped.
- The Index itself is a scalar abstract Op.
- If further operations see an abstract Op upstream this triggers an error (e.g. if someone references a "temp" variable created inside the context manager)

Don't see any way for this to go wrong.

But what if you do this?

X = Slot()
with Index() as i:
    X[i] = dist(A, B)



STAGE 2: Same, but you're allowed to create a Slot inside another context manager.

X = Slot()
Y = Slot()
with Index() as i:
    X[i] = dist(A[i], B[i])
    Z = Slot()
    with Index() as j:
        Z[j] = fun(X[i], C[j])
    Y[i] = Z

I think this is also fine?

- When the context manager for j exits, Z will be traced over j and all references to j in Z will be replaced with vmap. But Z will retain references to i.
- When the context manager for i exits, X and Y and Z will all be traced over i. All references to i in X and Y will be replaced with vmap, meaning these are now "fully legal".
- When the context manager for i exits, nothing happens for Z. It retains a refernce to i upstream, meaning that if you try to do inference w.r.t. Z, it won't be properly defined.

STAGE 3: Full slot life

X = Slot()
Y = Slot()
with Index() as i:
    X[i] = dist(A[i], B[i])
    with Index() as j:
        Y[i,j] = fun(X[i], C[j])

Now, Y[i,j] = blah just means that Y is a Slot that has two upward references. When the context manager for j exits, the first one is traced out. When the context manager for i exits, the second is traced out.

OK.. but what if you do this?


X = Slot()
Y = Slot()
with Index() as i:
    X[i] = dist1(A[i], B[i])
    with Index() as j:
        Y[i,j] = dist2(X[i], C)

Now we'd like Y[i,:] to be i.i.d. But Y has no upward reference to j.


...

STAGE 1:

I thiiiiink everything can just work like this? First, let's create a "fully functional core" without any magical syntax for "assigning". I think we just need three ingredients:

1) Axis: A special Op (basically an int)
 - You can index RVs with an InfixRV[Axis]. Nothing special needs to happen.
 - Trying to use an Axis Op in a backend should always trigger an error

2) vmap_axis(rv_list, ax)
 - take a sequence of rvs that have ancestors that index some Axis
 - return a new sequence of rvs with new ancestors that don't index that Axis

Example:

x = constant([1.1, 2.2, 3.3])
y = constant([4.4, 5.5])
i = InfixRV(Axis(3))
j = InfixRV(Axis(2))
xi = x[i]
yj = y[j]
zij = xi * yj
uij = normal(zij, yj)
zi, ui = vmap_axis([zij, uij], j)
z, u = vmap_axis([zi, ui], i)
"""

from .base import InfixRV, AbstractOp, generated_nodes, constant, vmap_subgraph, vmap, print_upstream
from pangolin import ir
from typing import Sequence, Callable
import warnings
import inspect
from pangolin import dag, util
from jaxtyping import PyTree
import jax.tree_util
from pangolin import interface as pi


# class Index(InfixRV):
#    def __init__(self, size):
#        self.size = size


class Axis(ir.Constant):
    """
    A scalar value but "special"
    """

    def __init__(self, size: int):
        self.size = size
        super().__init__(size)


def axis(size):
    return InfixRV(Axis(size))


class Slot(InfixRV):
    """
    What a Slot does:
    - Initially, it throws an error if you try to do basically anything with it except __setitem__

    Examples
    --------
    >>> x = Slot()
    >>> x.shape
    Traceback (most recent call last):
        ...
    RuntimeError: Locked! You must assign a value before accessing 'shape'
    >>> value = pi.constant([5,6,7])
    >>> i = axis(10)
    >>> x[i] = value
    >>> x.shape
    (10, 3)
    >>> x_i = x[i]
    """

    def __init__(self):
        # We use object.__setattr__ to avoid triggering recursion if we had a __setattr__ override
        object.__setattr__(self, "_initialized", False)
        # Note: We do not call super().__init__() here because you likely
        # want to defer that until initialize(), depending on your logic.

    def __setitem__(self, idx: InfixRV[Axis], value: InfixRV):

        # 1. Do your setup logic here
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "value", value)

        # 2. Flip the switch
        object.__setattr__(self, "_initialized", True)

    def __getattribute__(self, item):
        # 1. Retrieve the flag safely
        # We handle '_initialized' explicitly to prevent recursion
        if item == "_initialized":
            return object.__getattribute__(self, item)

        # 2. Check if initialized
        if not object.__getattribute__(self, "_initialized"):
            # Allow __setitem__ explicitly (in case someone calls obj.__setitem__ directly)
            # Allow __class__ and __repr__ so debuggers/print() don't crash immediately
            if item in ("__setitem__", "__class__", "__repr__"):
                return object.__getattribute__(self, item)

            raise RuntimeError(f"Locked! You must assign a value before accessing '{item}'")

        return super().__getattribute__(item)

    @property
    def shape(self):
        axis_size = self.idx.op.size
        return (axis_size,) + self.value.shape

    @property
    def ndim(self):



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
