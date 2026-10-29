"""Construct vectorized graphs over named integer ranges."""

import inspect
import operator
from collections.abc import Callable
from typing import Protocol, SupportsIndex, overload

from pangolin import ir, util
from pangolin.util import PyTree

from .tracing import require_rv_leaves
from .vmapping import vmap


class VForDecorator(Protocol):
    """Static type of a decorator returned by `vfor`.

    This protocol describes the accepted function signatures for type checkers. Users do not
    construct it directly.
    """

    @overload
    def __call__[Output: PyTree[ir.RV]](self, fun: Callable[[], Output], /) -> Output: ...

    @overload
    def __call__[Output: PyTree[ir.RV]](self, fun: Callable[[ir.RV], Output], /) -> Output: ...

    @overload
    def __call__[Output: PyTree[ir.RV]](
        self, fun: Callable[[ir.RV, ir.RV], Output], /
    ) -> Output: ...

    @overload
    def __call__[Output: PyTree[ir.RV]](
        self, fun: Callable[[ir.RV, ir.RV, ir.RV], Output], /
    ) -> Output: ...

    @overload
    def __call__[Output: PyTree[ir.RV]](
        self, fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], Output], /
    ) -> Output: ...

    @overload
    def __call__[Output: PyTree[ir.RV]](
        self,
        fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], Output],
        /,
    ) -> Output: ...

    @overload
    def __call__[Output: PyTree[ir.RV], *Rest](
        self,
        fun: Callable[
            [ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest],
            Output,
        ],
        /,
    ) -> Output: ...


def axis_size(name: str, value: object) -> int:
    """Validate and normalize one positive integer coordinate-axis size."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"size for axis {name!r} must be an integer; got {type(value).__name__}")
    size = operator.index(value)
    if size <= 0:
        raise ValueError(f"size for axis {name!r} must be positive; got {size}")
    return size


def vfor_simple[Output: PyTree[ir.RV]](
    fun: Callable[..., Output], *sizes: int, interface: ir.RVInterface
) -> Output:
    """Map over positional sizes, attaching ``interface`` to the integer coordinates."""

    num_indices = len(sizes)
    num_args = util.fixed_arity(fun)

    if num_indices != num_args:
        raise ValueError(f"Number of sizes {num_indices} not equal to number of args {num_args}")

    def call_coordinates(indices: tuple[ir.RV, ...]) -> Output:
        return fun(*indices)

    mapped = call_coordinates

    for index_num in range(num_indices - 1, -1, -1):
        in_axes = tuple(0 if parent_num == index_num else None for parent_num in range(num_indices))

        mapped = vmap(mapped, in_axes, sizes[index_num])

    coordinates = tuple(ir.RV(ir.Constant(range(size)), interface=interface) for size in sizes)

    output = mapped(coordinates)
    require_rv_leaves(output, "vfor output")
    return output


def vfor_parameter_names(fun: Callable[..., PyTree[ir.RV]]) -> tuple[str, ...]:
    """Return parameter names after requiring a fixed positional signature."""

    util.fixed_arity(fun)
    return tuple(inspect.signature(fun).parameters)


@overload
def vfor[Output: PyTree[ir.RV]](
    fun: Callable[[], Output],
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int,
) -> Output: ...


@overload
def vfor[Output: PyTree[ir.RV]](
    fun: Callable[[ir.RV], Output],
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int,
) -> Output: ...


@overload
def vfor[Output: PyTree[ir.RV]](
    fun: Callable[[ir.RV, ir.RV], Output],
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int,
) -> Output: ...


@overload
def vfor[Output: PyTree[ir.RV]](
    fun: Callable[[ir.RV, ir.RV, ir.RV], Output],
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int,
) -> Output: ...


@overload
def vfor[Output: PyTree[ir.RV]](
    fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], Output],
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int,
) -> Output: ...


@overload
def vfor[Output: PyTree[ir.RV]](
    fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], Output],
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int,
) -> Output: ...


@overload
def vfor[Output: PyTree[ir.RV], *Rest](
    fun: Callable[
        [ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest],
        Output,
    ],
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int,
) -> Output: ...


@overload
def vfor(
    fun: None = None,
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int,
) -> VForDecorator: ...


def vfor[Output: PyTree[ir.RV]](
    fun: Callable[..., Output] | None = None,
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int,
) -> Output | VForDecorator:
    """Vectorize a function over named integer ranges.

    ``fun`` must have a fixed positional signature. Each parameter receives a scalar `ir.RV`
    representing an integer coordinate from zero through one less than its named size. ``fun``
    must return a PyTree of newly created `ir.RV` objects; it cannot return a coordinate or an
    existing RV captured through a closure. With no coordinate parameters, ``fun`` is called
    directly and its RV outputs need not be newly created.

    Axis sizes are matched to function parameters by name. Output axes occur in
    function-parameter order, regardless of the order in which keyword arguments are supplied.

    This combinator supports direct calls, decorator factories, and multiline
    decorator syntax.

    Parameters
    ----------
    fun : Callable[..., PyTree[ir.RV]] | None
        Function to vectorize, or ``None`` when creating a decorator. It must have a fixed
        positional signature, accept one scalar `ir.RV` coordinate per parameter, and return a
        PyTree of newly created `ir.RV` objects. No parameter may be named ``interface``.
    interface : ir.RVInterface
        Required keyword-only interface attached to the generated coordinate RVs. Output RVs
        retain the interfaces chosen by ``fun``.
    **sizes : int
        Positive integer size for every parameter of ``fun``, supplied under the corresponding
        parameter name.

    Returns
    -------
    output_or_decorator : PyTree[ir.RV] | decorator
        The vectorized output PyTree when ``fun`` is supplied, otherwise a decorator that accepts
        ``fun`` and produces that output.

    Raises
    ------
    TypeError
        If ``fun`` is not callable, has no fixed positional signature, a size is not an integer,
        or the output is not a PyTree of `ir.RV` objects.
    ValueError
        If an axis size is missing, an unknown axis name is supplied, a size is not positive,
        a parameter is named ``interface``, or an output RV was not created by ``fun``.

    Notes
    -----
    Static type checking preserves the number of coordinate arguments and exact PyTree structure
    of the output, and requires the first six arguments to be `ir.RV` objects. Runtime validation
    still applies to every argument.

    Examples
    --------
    Direct call:

    >>> from pangolin import interface as pi
    >>> interface = pi.default
    >>> x = vfor(lambda i, j: 10 * i + j, interface=interface, i=2, j=3)
    >>> x.shape
    (2, 3)

    Keyword order does not affect axis order:

    >>> y = vfor(lambda i, j: 10 * i + j, interface=interface, j=3, i=2)
    >>> y.shape
    (2, 3)
    >>> ir.rv_equal(x, y)
    True

    Decorator factory:

    >>> make_x = vfor(interface=interface, i=2, j=3)
    >>> x = make_x(lambda i, j: 10 * i + j)
    >>> x.shape
    (2, 3)

    Multiline decorator form:

    >>> @vfor(interface=interface, i=2, j=3)
    ... def x(i, j):
    ...     return 10 * i + j
    >>> x.shape
    (2, 3)

    Every function parameter must have a named size:

    >>> vfor(lambda i, j: i + j, interface=interface, i=2)
    Traceback (most recent call last):
    ...
    ValueError: vfor missing size for axis 'j'

    Unknown axis names are rejected:

    >>> vfor(lambda i, j: i + j, interface=interface, i=2, j=3, k=4)
    Traceback (most recent call last):
    ...
    ValueError: vfor got size for unknown axis 'k'; expected ('i', 'j')

    Python defaults are not interpreted as axis sizes:

    >>> vfor(lambda i=3: i + 1, interface=interface)
    Traceback (most recent call last):
    ...
    TypeError: <lambda> has no fixed positional signature: 'i' has a default
    """

    if fun is None:

        def apply_vfor[ActualOutput: PyTree[ir.RV]](
            actual_fun: Callable[..., ActualOutput],
        ) -> ActualOutput:
            return vfor(actual_fun, interface=interface, **sizes)

        return apply_vfor

    if not callable(fun):
        raise TypeError(f"vfor expected a callable; got {type(fun).__name__}")

    parameter_names = vfor_parameter_names(fun)
    if "interface" in parameter_names:
        raise ValueError(
            "vfor function parameters cannot be named 'interface'; the name is reserved"
        )
    parameter_name_set = set(parameter_names)

    unknown_names = [name for name in sizes if name not in parameter_name_set]

    if unknown_names:
        name = unknown_names[0]
        raise ValueError(f"vfor got size for unknown axis {name!r}; " f"expected {parameter_names}")

    missing_names = [name for name in parameter_names if name not in sizes]

    if missing_names:
        name = missing_names[0]
        raise ValueError(f"vfor missing size for axis {name!r}")

    ordered_sizes = tuple(axis_size(name, sizes[name]) for name in parameter_names)

    return vfor_simple(fun, *ordered_sizes, interface=interface)


__all__ = ["vfor"]
