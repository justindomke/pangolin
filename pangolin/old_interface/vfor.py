"""
The idea of this (sub)-module is to support a notation like this:

X = vfor(lambda i,j: A[i,:] * B[j,:])

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

# from jax._src.core import pp_toplevel_jaxpr
from .base import InfixRV, constant
from .tracing import AbstractOp
from .vmapping import vmap
from pangolin import ir
from typing import Sequence, Callable
import warnings
import inspect
from pangolin import dag, util
from jaxtyping import PyTree
import jax.tree_util
from pangolin import interface as pi


class Axis(ir.Op):
    """A scalar, but 'special'. Never legal in a backend."""

    _random = False
    _get_shape = lambda *args: ()

    def __init__(self, size: int | None = None):
        self.size = size
        super().__init__()


def axis(size):
    return InfixRV(Axis(size))


# def caxis(size):
#     return ContextRV(Axis(size))


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


def get_positional_names(func: Callable) -> tuple[str, ...]:
    """Gets the names of arguments that can be passed positionally.

    Args:
        func: The function to inspect.

    Returns:
        Names of all position-capable parameters.

    Raises:
        ValueError: If the function signature contains parameters with default values.
        ValueError: If the function signature contains keyword-only arguments.
        ValueError: If the function signature contains variable positional arguments (*args).
        ValueError: If the function signature contains variable keyword arguments (**kwargs).

    Examples:
        >>> def func_empty(): pass
        >>> get_positional_names(func_empty)
        ()

        >>> def func_standard(a, b, c): pass
        >>> get_positional_names(func_standard)
        ('a', 'b', 'c')

        >>> def func_pos_only(a, b, /, c): pass
        >>> get_positional_names(func_pos_only)
        ('a', 'b', 'c')

        >>> def func_default(a, b=2): pass
        >>> get_positional_names(func_default)
        Traceback (most recent call last):
            ...
        ValueError: Function 'func_default' has default arguments.

        >>> def func_kwonly(a, *, b): pass
        >>> get_positional_names(func_kwonly)
        Traceback (most recent call last):
            ...
        ValueError: Function 'func_kwonly' has keyword-only arguments.

        >>> def func_args(a, *args): pass
        >>> get_positional_names(func_args)
        Traceback (most recent call last):
            ...
        ValueError: Function 'func_args' has *args.

        >>> def func_kwargs(a, b, **kwargs): pass
        >>> get_positional_names(func_kwargs)
        Traceback (most recent call last):
            ...
        ValueError: Function 'func_kwargs' has **kwargs.
    """
    sig = inspect.signature(func)
    positional_names = []

    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            raise ValueError(f"Function '{func.__name__}' has default arguments.")
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(f"Function '{func.__name__}' has keyword-only arguments.")
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise ValueError(f"Function '{func.__name__}' has *args.")
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            raise ValueError(f"Function '{func.__name__}' has **kwargs.")

        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional_names.append(name)

    return tuple(positional_names)


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

            # replayed_rv = InfixRV(rv.op, *[rv_to_replayed[p] for p in rv.parents])

            my_parents = []
            for p in rv.parents:
                if p in rv_to_replayed:
                    my_p = rv_to_replayed[p]
                else:
                    my_p = p
                my_parents.append(my_p)
            replayed_rv = InfixRV(rv.op, *my_parents)

            rv_to_replayed[rv] = replayed_rv

        results_flat = [rv_to_replayed[out_rv] for out_rv in output_rvs]

        return jax.tree_util.tree_unflatten(output_treedef, results_flat)

    return indexed_rvs, indexed_dims, replay


def index_if_necessary(base, *indices):
    if all(idx.op == ir.Constant(range(size)) for size, idx in zip(base.shape, indices)):
        return base
    else:
        return InfixRV(ir.Index(), base, *indices)


def popout_axis(rv: InfixRV[ir.Index], ax: InfixRV[Axis]):
    """
    - rv is some RV that's been indexed with some number of axes
    - we want to "recover" one of those axes, i.e. remove that axis from the indexing statement
    - and we want to know where to find it
    """

    where = 0
    base = rv.parents[0]
    indices = rv.parents[1:]
    for n, idx in enumerate(indices):
        if idx is ax:
            if base.shape[n] != ax.op.size:
                raise ValueError(f"Axis size {ax.op.size} does not match dim {n} of shape {base.shape}")

            new_idx = pi.constant(range(ax.op.size))
            new_indices = indices[:n] + (new_idx,) + indices[n + 1 :]
            return index_if_necessary(base, *new_indices), where
        else:
            where += idx.ndim
    raise ValueError(f"axis rv {ax} not found in Index rv {rv}")


def extract_from_rvs_single_axis(dummy_index: InfixRV[Axis], output: PyTree[InfixRV]):
    n_before_index = dummy_index._n

    def is_abstract(rv: InfixRV) -> bool:
        return rv._n >= n_before_index

    def not_abstract(var: InfixRV):
        return not is_abstract(var)

    output_rvs, output_treedef = jax.tree_util.tree_flatten(output)

    all_rvs = dag.upstream_nodes(output_rvs, node_block=not_abstract)

    indexed_rvs = []
    indexed_dims = []
    touched_rvs = set()
    for rv in all_rvs:
        if any(p in touched_rvs for p in rv.parents):
            touched_rvs.add(rv)

        if isinstance(rv.op, ir.Index):
            # rv_base = rv.parents[0]
            rv_indices = rv.parents[1:]
            if dummy_index in rv_indices:
                new_rv, where = popout_axis(rv, dummy_index)
                indexed_rvs.append(new_rv)
                indexed_dims.append(where)
                touched_rvs.add(rv)

    def replay(*indexed_rvs_replayed):
        rv_to_replayed = {}

        n = 0
        for rv in all_rvs:
            if rv not in touched_rvs:
                rv_to_replayed[rv] = rv
                continue

            if isinstance(rv.op, ir.Index):
                if dummy_index in rv.parents[1:]:
                    rv_to_replayed[rv] = indexed_rvs_replayed[n]
                    n += 1
                    continue

            new_parents = []
            for p in rv.parents:
                if p in rv_to_replayed:
                    new_parents.append(rv_to_replayed[p])
                else:
                    new_parents.append(p)

            rv_to_replayed[rv] = InfixRV(rv.op, *new_parents)

        results_flat = [rv_to_replayed[out_rv] for out_rv in output_rvs]
        return jax.tree_util.tree_unflatten(output_treedef, results_flat)

    return indexed_rvs, indexed_dims, replay


def live_axis(dim, idx):
    """
    Compute the vmap in_axis for one array at one step of vfor's loop.

    vfor builds the mapped computation as NESTED vmaps: the loop runs from
    the last dummy index to the first, wrapping each new vmap around the
    previous one, so dummy index 0 ends up OUTERMOST. At call time the
    outer maps slice first, and each inner map receives only the slice the
    outer maps produced.

    Consequently, when the map for dummy ``idx`` runs:
      * axes of dummies at positions ``p < idx`` (outer maps) are already
        SLICED AWAY;
      * axes of dummies at positions ``p > idx`` (inner maps) are STILL
        PRESENT (they get sliced deeper inside);
      * slicing removes an axis but never reorders the survivors.

    ``dims[k][idx]`` records the slot dummy ``idx`` occupied in the ORIGINAL
    array. Its live position is that slot minus the number of already-sliced
    axes (``p < idx``) that were AHEAD of it. Removed axes behind the slot
    do not move it.

    Parameters
    ----------
    dim : list
        One entry of ``dims``: for a single array, the original slot (int)
        of each dummy index, or ``None`` where that dummy does not index
        this array.
    idx : int
        Position of the dummy index being mapped at this loop step.

    Returns
    -------
    int or None
        The in_axis for vmap at this step, or ``None`` if dummy ``idx``
        does not index this array.

    Examples
    --------
    Single dummy: x[i] -> map its only axis.

    >>> live_axis([0], 0)
    0

    trouble2 pattern: b[j] with dummies [i, j]; b not indexed by i.

    >>> live_axis([None, 0], 1)
    0
    >>> live_axis([None, 0], 0) is None
    True

    Ascending pattern: x[i, j], dims [0, 1]. Mapping j (idx=1): the outer
    i-map sliced away slot 0, which was ahead of j's slot 1 -> shift to 0.

    >>> live_axis([0, 1], 1)
    0

    Mapping i (idx=0): i is outermost and sees the original array, raw slot.

    >>> live_axis([0, 1], 0)
    0

    Transposed pattern: x[j, i] with x of shape (2, 3), so dims [1, 0]
    (i's slot is 1, j's slot is 0). Mapping j (idx=1): the outer i-map
    sliced slot 1, which was BEHIND j's slot 0, so j does not move.

    >>> live_axis([1, 0], 1)
    0

    Mapping i (idx=0): outermost, sees the original array, raw slot 1.

    >>> live_axis([1, 0], 0)
    1

    That pair (j over axis 0 of each x[:, i] slice, i over axis 1 of x)
    stacks j inner and i outer, giving y.shape == (3, 2) for
    vfor(lambda i, j: x[j, i] + 2.2) -- dummy order determines output axes.

    Array not indexed by this dummy -> None (leave unmapped).

    >>> live_axis([0, None], 1) is None
    True
    """
    if dim[idx] is None:
        return None
    return dim[idx] - sum(1 for p in range(idx) if dim[p] is not None and dim[p] < dim[idx])


from typing import overload, Any


# Overload 1: Direct call or bare decorator (@vfor or vfor(fun, i=3))
@overload
def vfor[P: PyTree[InfixRV]](fun: Callable[..., P], /, **kwargs: int) -> P: ...


# Overload 2: Decorator factory (@vfor(i=3) or vfor(i=3)(fun))
@overload
def vfor[P: PyTree[InfixRV]](fun: None = None, /, **kwargs: int) -> Callable[[Callable[..., P]], P]: ...


def vfor[P: PyTree[InfixRV]](
    fun: Callable[..., P] | None = None, /, **kwargs: int
) -> P | Callable[[Callable[..., P]], P]:
    """
    ``vfor`` is an alternative (hopefully more natural) interface for vmapping, where you align dimensions by name instead of through ``in_axes`` arguments. (Inspired by `Dex <https://github.com/google-research/dex-lang>`_.)

    Say you wanted to write a function to compute ``c[i,j] = sqrt(a[i]*b[j])``. Using vmap, this would be:

    >>> a = constant([1., 2., 3,])
    >>> b = constant([4., 5., 6.])
    >>> e = vmap(
    ...         vmap(
    ...             lambda a_i, b_j: (a_i * b_j)**0.5,
    ...             [None, 0]
    ...         ),
    ...         [0, None]
    ... )(a, b)
    >>> print_upstream(e)
    shape  | statement
    ------ | ---------
    (3,)   | a = [1. 2. 3.]
    (3,)   | b = [4. 5. 6.]
    (3, 3) | c = vmap(vmap(mul, [None, 0], 3), [0, None], 3)(a, b)
    ()     | d = 0.5
    (3, 3) | e = vmap(vmap(pow, [0, None], 3), [0, None], 3)(c, d)

    Using ``vfor`` you can simply write

    >>> a = constant([1., 2., 3,])
    >>> b = constant([4., 5., 6.])
    >>> e = vfor(lambda i, j: (a[i] * b[j])**0.5)
    >>> print_upstream(e)
    shape  | statement
    ------ | ---------
    (3,)   | a = [1. 2. 3.]
    (3,)   | b = [4. 5. 6.]
    (3, 3) | c = vmap(vmap(mul, [None, 0], 3), [0, None], 3)(a, b)
    ()     | d = 0.5
    (3, 3) | e = vmap(vmap(pow, [0, None], 3), [0, None], 3)(c, d)

    This function supports two different call signatures to make it convenient to use as a wrapper or decorator.

    * ``vfor()(fun)`` is equivalent to ``vfor(fun)``
    * ``vfor(i=3, j=4)(fun)`` is equivalent to ``vfor(fun, i=3, j=4)``.

    Args:
        fun: A function that takes "indices" and produces an output or ``None`` to create a decorator. (Default: None)
        **kwargs: names of indices declaring sizes. (Optional.)

    Returns:
        A pytree of RVs representing ``fun`` mapped over all the indices. Or, if ``fun`` is ``None``, returns a Callable that will create said pytree when passed a function.

    Examples
    --------

    Mutiply each element by two.

    >>> x = constant([1.1, 2.2, 3.3])
    >>> y = vfor(lambda i: x[i]*2)
    >>> print(y)
    vmap(mul, [0, None], 3)([1.1 2.2 3.3], 2)

    You can provide an axis size for safety.

    >>> x = constant([1.1, 2.2, 3.3])
    >>> y = vfor(lambda i: x[i]*2, i=3)
    >>> print(y)
    vmap(mul, [0, None], 3)([1.1 2.2 3.3], 2)

    Or, you can provide the axis size before the function.

    >>> x = constant([1.1, 2.2, 3.3])
    >>> y = vfor(i=3)(lambda i: x[i]*2)
    >>> print(y)
    vmap(mul, [0, None], 3)([1.1 2.2 3.3], 2)

    Or, you can use ``vfor`` as a decorator.

    >>> x = constant([1.1, 2.2, 3.3])
    >>> @vfor
    ... def y(i):
    ...     return x[i]*2
    >>> print(y)
    vmap(mul, [0, None], 3)([1.1 2.2 3.3], 2)

    Or, you can use ``vfor`` as a decorator with an axis size for safety.

    >>> x = constant([1.1, 2.2, 3.3])
    >>> @vfor(i=3)
    ... def y(i):
    ...     return x[i]*2
    >>> print(y)
    vmap(mul, [0, None], 3)([1.1 2.2 3.3], 2)

    An example passing two indices with a single parent RV:

    >>> x = constant([[1,2,3],[4,5,6]])
    >>> x.shape
    (2, 3)
    >>> y = vfor(lambda i, j: x[j,i] + 2.2)
    >>> y.shape
    (3, 2)
    >>> print(y)
    vmap(vmap(add, [0, None], 2), [1, None], 3)([[1 2 3] [4 5 6]], 2.2)

    Slice the first dimension of ``a`` and the first/only dimension of ``b``.

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

    Or, with explicit axis lengths

    >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> b = constant([[7.7, 8.8],[9.9, 10.10],[11.11, 12.12]])
    >>> c = vfor(lambda i, j: a[:,i] @ b[j,:], i=3, j=3)
    >>> print_upstream(c)
    shape  | statement
    ------ | ---------
    (2, 3) | a = [[1.1 2.2 3.3] [4.4 5.5 6.6]]
    (3, 2) | b = [[ 7.7  8.8 ] [ 9.9 10.1 ] [11.11 12.12]]
    (3, 3) | c = vmap(vmap(matmul, [None, 0], 3), [1, None], 3)(a, b)

    Or as a decorator

    >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> b = constant([[7.7, 8.8],[9.9, 10.10],[11.11, 12.12]])
    >>> @vfor
    ... def c(i,j):
    ...     return a[:,i] @ b[j,:]
    >>> print_upstream(c)
    shape  | statement
    ------ | ---------
    (2, 3) | a = [[1.1 2.2 3.3] [4.4 5.5 6.6]]
    (3, 2) | b = [[ 7.7  8.8 ] [ 9.9 10.1 ] [11.11 12.12]]
    (3, 3) | c = vmap(vmap(matmul, [None, 0], 3), [1, None], 3)(a, b)

    Or as a decorator with marked axes lengths

    >>> a = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> b = constant([[7.7, 8.8],[9.9, 10.10],[11.11, 12.12]])
    >>> @vfor(i=3, j=3)
    ... def c(i,j):
    ...     return a[:,i] @ b[j,:]
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

    if fun is None:
        return lambda fun: vfor(fun, **kwargs)

    arg_names = get_positional_names(fun)
    num_indices = len(arg_names)

    dummy_indices = [InfixRV(AbstractOp(())) for _ in range(num_indices)]
    output_rv = fun(*dummy_indices)
    arrays, dims, replay = extract_from_rvs(dummy_indices, output_rv)

    new_fun = replay

    for idx in range(num_indices - 1, -1, -1):
        in_axes = tuple(live_axis(dim, idx) for dim in dims)

        arg_name = arg_names[idx]

        if arg_name in kwargs:
            axis_size = kwargs[arg_name]
        else:
            axis_size = None

        if len(arrays) == 1:
            new_fun = vmap(new_fun, in_axes[0], axis_size)
        else:
            new_fun = vmap(new_fun, in_axes, axis_size)
    return new_fun(*arrays)


def test():
    A = constant([1, 2, 3])

    @vfor(i=3)
    def B(i):
        return A[i] * 2

    C = B

    # B = vfor(lambda i: A[i] * 2)


def count_positional(func):
    """Returns the number of positional parameters of func.

    Errors if func has any:
      - keyword-only parameters
      - *args / **kwargs
      - parameters with default values
    """
    sig = inspect.signature(func)

    bad = [
        p.name
        for p in sig.parameters.values()
        if (
            p.default is not inspect.Parameter.empty  # has a default
            or p.kind
            in (
                inspect.Parameter.KEYWORD_ONLY,  # keyword-only
                inspect.Parameter.VAR_POSITIONAL,  # *args
                inspect.Parameter.VAR_KEYWORD,
            )
        )
    ]  # **kwargs

    if bad:
        raise ValueError(f"Function has disallowed parameters: {bad}")

    # Only POSITIONAL_ONLY and POSITIONAL_OR_KEYWORD remain
    return len(sig.parameters)


def vfor_simple(fun, *sizes: int):
    """

    Examples
    --------
    Mutiply raw input by two.

    >>> x = vfor_simple(lambda i: i * 4.4, 3)
    >>> print(x)
    vmap(mul, [0, None], 3)([0 1 2], 4.4)

    iid bernoulli

    >>> x = vfor_simple(lambda i: pi.bernoulli(0.5), 3)
    >>> print(x)
    vmap(bernoulli, [None], 3)(0.5)

    multiply array by 2

    >>> x = constant([1.1, 2.2, 3.3])
    >>> y = vfor_simple(lambda i: x[i]*2, 3)
    >>> print(y)
    vmap(mul, [0, None], 3)(vmap(index, [None, 0], 3)([1.1 2.2 3.3], [0 1 2]), 2)

    outer product

    >>> x = vfor_simple(lambda i,j : i*j, 2, 3)
    >>> print(x)
    vmap(vmap(mul, [None, 0], 3), [0, None], 2)([0 1], [0 1 2])

    an index returned unchanged is not (yet) supported

    two outputs

    >>> x = vfor_simple(lambda i: (i+2, i*3), 3)
    >>> print(x[0])
    vmap(add, [0, None], 3)([0 1 2], 2)
    >>> print(x[1])
    vmap(mul, [0, None], 3)([0 1 2], 3)

    multiply 2d array by 2

    >>> x = constant([[1.1, 2.2, 3.3],[4.4, 5.5, 6.6]])
    >>> y = vfor_simple(lambda i, j: x[i,j]*2, 2, 3)
    >>> print(y)
    vmap(mul, [0, None], 3)(vmap(index, [None, 0], 3)([1.1 2.2 3.3], [0 1 2]), 2)


    >>> x = vfor_simple(lambda i:i, 3)
    Traceback (most recent call last):
    ...
    ValueError: fun passed to generated_nodes cannot return input values
    """

    num_indices = len(sizes)
    num_args = count_positional(fun)
    if num_indices != num_args:
        raise ValueError(f"Number of sizes {num_indices} not equal to number of args {num_args}")

    # takes a list to simplify in_axes
    mapped = lambda idx: fun(*idx)

    for idx in range(num_indices - 1, -1, -1):
        in_axes = [0 if k == idx else None for k in range(num_indices)]

        mapped = vmap(mapped, in_axes, axis_size=sizes[idx])

    iotas = [pi.constant(pi.np.arange(size)) for size in sizes]

    return mapped(iotas)


# def simplify_vmap(x: InfixRV):
#     """
#     if a RV is a vmap

#     """

#     pars = [simplify_vmap(x.parents(p)

#     if not isinstance(x.op, ir.VMap):
#         return x

#     assert isinstance(x.op, ir.VMap)

#     base_op = x.op.base_op
#     if isinstance(x.op, ir.Index):
