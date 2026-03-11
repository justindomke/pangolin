from __future__ import annotations
from pangolin.ir import Scan, Composite, Op
from pangolin.interface import (
    InfixRV,
    makerv,
    constant,
    create_rv,
    exp,
    add,
    normal,
    exponential,
)
from pangolin.interface.compositing import composite_flat, make_composite
from .base import generated_nodes, AbstractOp, get_dummy_args
from pangolin import util
import jax.tree_util
from pangolin.interface.base import RVLike

# from pangolin.simple_interface.vmapping import get_flat_vmap_args_and_axes
from typing import Callable, Sequence, Any, Protocol, TypeAlias
from jaxtyping import PyTree
import numpy as np

# would like to insist that the function takes RV args but type system not up to the task


type FlatScannable[O: Op] = (
    Callable[[InfixRV], InfixRV[O]]
    | Callable[[InfixRV, InfixRV], InfixRV[O]]
    | Callable[[InfixRV, InfixRV, InfixRV], InfixRV[O]]
    | Callable[[InfixRV, InfixRV, InfixRV, InfixRV], InfixRV[O]]
    | Callable[[InfixRV, InfixRV, InfixRV, InfixRV, InfixRV], InfixRV[O]]
    | Callable[[InfixRV, InfixRV, InfixRV, InfixRV, InfixRV, InfixRV], InfixRV[O]]
    | Callable[[InfixRV, InfixRV, InfixRV, InfixRV, InfixRV, InfixRV, InfixRV], InfixRV[O]]
)
"""
A type alias for a function that takes one `InfixRV` input plus any number of additional `InfixRV` and returns a single `InfixRV` output. Because of Python's wonderfully limited type system, this is implemented as a union of functions with arity up to 6.
"""

type FlatScanned[O: Op] = Callable[..., InfixRV[Scan[Composite[O]]]]
"""
A type alias intended to indicate a function that takes one `InfixRV` input plus any number of additional `InfixRV` and returns a single scan InfixRV output. Because of Python's wonderfully limited type system, this does not actually check the inputs.
"""


type Scannable[O: Op] = (
    Callable[[InfixRV], InfixRV[O]]
    | Callable[[InfixRV, "PyTree[InfixRV]"], InfixRV[O]]
    | Callable[[InfixRV, "PyTree[InfixRV]", "PyTree[InfixRV]"], InfixRV[O]]
    | Callable[[InfixRV, "PyTree[InfixRV]", "PyTree[InfixRV]", "PyTree[InfixRV]"], InfixRV[O]]
    | Callable[[InfixRV, "PyTree[InfixRV]", "PyTree[InfixRV]", "PyTree[InfixRV]", "PyTree[InfixRV]"], InfixRV[O]]
    | Callable[
        [InfixRV, "PyTree[InfixRV]", "PyTree[InfixRV]", "PyTree[InfixRV]", "PyTree[InfixRV]", "PyTree[InfixRV]"],
        InfixRV[O],
    ]
    | Callable[
        [
            InfixRV,
            "PyTree[InfixRV]",
            "PyTree[InfixRV]",
            "PyTree[InfixRV]",
            "PyTree[InfixRV]",
            "PyTree[InfixRV]",
            "PyTree[InfixRV]",
        ],
        InfixRV[O],
    ]
)
"""
A type alias for a function that takes one `InfixRV` input plus any number of pytrees of `InfixRV` and returns a single `InfixRV` output. Because of Python's wonderfully limited type system, this is implemented as a union of functions with arity up to 6.
"""

type Scanned[O: Op] = Callable[..., InfixRV[Scan[Composite[O]]]]
"""
A type alias intended to indicate a function that takes one `InfixRV` input plus any number of additional ``PyTree[InfixRV]`` and returns a single scan RV output. Because of Python's limited type system, this does not actually check the inputs.
"""


def _get_scan_length(length: int | None, my_in_axes: Sequence[int | None], args: Sequence[InfixRV]) -> int:
    "If length is given, checks that it's compatible with all args. Otherwise, infers from args and chcecks they are compatible."

    my_length = length

    for ax, arg in zip(my_in_axes, args, strict=True):
        if ax is not None:
            if my_length:
                assert my_length == arg.shape[ax]
            else:
                my_length = arg.shape[ax]

    if my_length is None:
        raise ValueError("Can't create scan with length=None and no mapped axis")

    return my_length


def scan_flat[O: Op](flat_fun: FlatScannable[O], length: int, in_axes: tuple[int | None, ...]) -> FlatScanned[O]:
    """
    Given a "flat" function, create a function to generate an RV with an `Scan` Op.

    Doing

    ```python
    z = scan_flat(fun, length=5, in_axes=(0,None))(start, A, B)
    ```

    is semantically similar to

    ```python
    carry = start
    values = []
    for i in range(5):
        carry = fun(carry, A[i], B)
        values.append(carry)
    z = concatenate(values)
    ```

    Parameters
    ----------
    flat_fun
        function for base of scan
    length
        length of scan. Cannot be skipped
    in_axes
        which axis (or ``None``) to slice

    Examples
    --------
    >>> x = scan_flat(exp, length=5, in_axes=())(constant(7.7))
    >>> x.op
    Scan(Composite(1, (Exp(),), [[0]]), 5, (), 0)
    >>> x.op.base_op
    Composite(1, (Exp(),), [[0]])
    >>> x.op.length
    5
    >>> x.op.in_axes
    ()
    >>> x.op.where_self
    0
    >>> x.parents
    (InfixRV(Constant(7.7)),)

    >>> a = makerv(7.7)
    >>> b = makerv([1,2,3,4,5])
    >>> x = scan_flat(add, length=5, in_axes=(0,))(a, b)
    >>> x.op
    Scan(Composite(2, (Add(),), [[0, 1]]), 5, (0,), 0)
    >>> x.op.base_op
    Composite(2, (Add(),), [[0, 1]])
    >>> x.op.length
    5
    >>> x.op.in_axes
    (0,)
    >>> x.op.where_self
    0
    >>> x.parents
    (InfixRV(Constant(7.7)), InfixRV(Constant([1,2,3,4,5])))

    """

    def myfun(init: RVLike, *args0: RVLike):
        init = makerv(init)
        args = tuple(makerv(a) for a in args0)

        # first, get composite op
        init_shape = init.shape
        args_shapes = tuple(a.shape for a in args)
        base_args_shapes = tuple(s[1:] for s in args_shapes)  # assume all mapped along 1st axis
        base_op, constants = make_composite(flat_fun, init_shape, *base_args_shapes)
        where_self = 0
        # where_self = len(constants) # constants always first in composite
        op = Scan(
            base_op,
            length,
            in_axes=[None] * len(constants) + list(in_axes),
            where_self=where_self,
        )
        return create_rv(op, init, *constants, *args)

    return myfun


def scan[O: Op](fun: Scannable[O], length: None | int = None, in_axes: PyTree[int | None] = 0) -> Scanned[O]:
    """
    Given a function, create a function to generate an RV with an `Scan` Op. Doing

    .. code-block:: python

        scan_fun = scan(fun, 5)
        z = scan_fun(start)

    is semantically like

    .. code-block:: python

        carry = start
        values = []
        for i in range(5):
            carry = fun(carry)
            values.append(carry)
        z = concatenate(values)

    As a more complex example, doing

    .. code-block:: python

        scano_fun = scan(fun, 5, [0, None])
        z = scan_fun(start, A, B)

    is semantically like:

    .. code-block:: python

        carry = start
        values = []
        # length = 5
        for i in range(5):
            carry = fun(carry, A[i], B) # A sliced on 0th axis, B unsliced
            values.append(carry)
        z = concatenate(values)

    Even more generally, ``in_axes`` can be any pytree of in-axes that is a tree prefix for the arguments. (See examples below)

    Parameters:
        fun:
            Function to call repeatedly to define the distribution.
            Must take some argument ``carry`` as the first input and return a single `RV` with the same shape as ``carry``.
            This function can only create a single *random* `RV` which (if it exists) must be the return value. But it can create an arbitrary number of non-random `RV` internally.
            This function can take extra inputs, which can be `RVLike` or pytrees of `RVLike`. These can be mapped as determined
            Can optionally take extra inputs that will be mapped.
        length:
            Length of scan. Can be ``None`` if any inputs are mapped along some axis.
        in_axes:
            What axis to map each input other than ``carry`` over (or ``None`` if
            non-mapped). As with `vmap`, can be a pytree of `RV` corresponding to the structure of all
            inputs other than ``carry``.

    Returns:
        Function that takes a single init `RVLike` plus some number of pytrees of `RVLike` with mapped axes and produces a single ``RV[Scan]``

    Examples
    --------
    Distribution where ``z[i] ~ normal(exp(z[i-1]), 1)``.

    >>> x = constant(3.3)
    >>> def fun(carry):
    ...     return normal(exp(carry), 1)
    >>> z = scan(fun, 5)(x)
    >>> isinstance(z.op, Scan)
    True
    >>> z.op.base_op
    Composite(1, (Exp(), Constant(1), Normal()), [[0], [], [1, 2]])
    >>> z.op.length
    5
    >>> z.op.in_axes
    ()
    >>> z.parents == (x,)
    True


    Distribution where ``z[i] ~ exponential(z[i-1]*y[i])``

    >>> x = constant(3.3)
    >>> y = constant([1,2,3])
    >>> def fun(carry, yi):
    ...     return exponential(carry * yi)
    >>> z = scan(fun)(x,y)
    >>> isinstance(z.op, Scan)
    True
    >>> z.op.base_op
    Composite(2, (Mul(), Exponential()), [[0, 1], [2]])
    >>> z.op.length # note this was inferred!
    3
    >>> z.op.in_axes
    (0,)
    >>> z.parents == (x, y)
    True


    You can also pass bare constants

    >>> def fun(carry, yi):
    ...     return exponential(carry * yi)
    >>> z = scan(fun)(3.3, np.array([1,2,3]))
    >>> isinstance(z.op, Scan)
    True
    >>> z.op.base_op
    Composite(2, (Mul(), Exponential()), [[0, 1], [2]])
    >>> z.op.length
    3
    >>> z.op.in_axes
    (0,)


    See Also
    --------
    Scannable
    Scanned
    pangolin.ir.Scan
    repeat
    """

    def myfun(init: RVLike, *args: PyTree[RVLike]):
        if len(args) == 1:
            # handles vmap(f, 0)(x) instead vmap(f,(0,))(x)
            my_in_axes = (in_axes,)
        elif isinstance(in_axes, list):
            # handles vmap(f, [0,1])(x,y) instead of vmap(f,(0,1))(x,y)
            my_in_axes = tuple(in_axes)
        else:
            my_in_axes = in_axes

        init = makerv(init)
        args = jax.tree_util.tree_map(makerv, args)

        abstract_sliced_args = get_dummy_args(my_in_axes, args)
        flat_in_axes, flat_args = util.dual_flatten(my_in_axes, args)

        flat_fun, flatten_inputs, unflatten_output = util.flatten_fun(
            fun, init, *abstract_sliced_args, is_leaf=util._is_leaf_with_none
        )
        # new_flat_fun = lambda *args: flat_fun(*args)[0]  # don't return list

        def new_flat_fun(init: RVLike, *args: RVLike) -> InfixRV:
            return flat_fun(init, *args)[0]

        my_length = _get_scan_length(length, flat_in_axes, flat_args)

        return scan_flat(new_flat_fun, my_length, flat_in_axes)(init, *flat_args)

    return myfun


def repeat[O: Op](length: int | None = None, in_axes: Any = 0) -> Callable[[Scannable[O]], Scanned[O]]:
    """
    Simple decorator to create functions to create scan RVs. The idea is that

    .. code-block:: python

        repeat(length, in_axes)(fun)

    is exactly the same as

    .. code-block:: python

        scan(fun, length, in_axes)

    This can be very convenient as a decorator.

    Args:
        length: the number of repetitions
        in_axes: axis to map arguments other than ``carry`` over

    Returns:
        Decorator that takes a function that transforms a `RVLike` and some number of pytress of `RVLike` into a single `RV` and returns a function that transforms a `RVLike` and some number of pytrees of `RVLike` into a single RV with an `ir.Scan` op.

    Examples
    --------

    Here are two equivalent examples. Here's `scan`:

    >>> x = constant(3.3)
    >>> def fun(carry):
    ...     return normal(exp(carry), 1)
    >>> z1 = scan(fun,5)(x)

    And here's `repeat`:

    >>> @repeat(5)
    ... def fun(carry):
    ...     return normal(exp(carry), 1)
    >>> z2 = fun(x)

    >>> z1.op == z2.op
    True
    >>> z1.parents == z2.parents == (x,)
    True

    See Also
    --------
    scan
    pangolin.ir.Scan
    """

    return lambda fun: scan(fun, length, in_axes)
