"""Construct graphs over named vectorized and sequential axes."""

import inspect
from typing import Any
from collections.abc import Callable, Mapping, Sequence

from pangolin import ir
from .indexing import index_orthogonal
from .scanning import scan
from .tracing import trace_call
from .vfor_looping import axis_size
from .vmapping import vmap

__all__ = ["loop"]


type Initializer = ir.RV | None
type SequentialAxis = tuple[int] | tuple[int, Initializer] | list[int | Initializer]


def loop_specifications(fun: Callable, sizes: Mapping[str, Any]) -> dict[str, Any]:
    """Read axis specifications from defaults or explicit sizes, never both.

    All parameters must be positional. With explicit sizes, no parameter may have
    a default. Otherwise defaults specify axes; prev must never have a default.
    This only inspects the function and does not validate or coerce axis values.
    """
    if not callable(fun):
        raise TypeError(f"loop expected a callable; got {type(fun).__name__}")
    parameters = inspect.signature(fun).parameters.values()
    defaults = {}
    for parameter in parameters:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                f"loop requires a fixed positional signature; {parameter.name!r} is not positional"
            )
        if parameter.default is not inspect.Parameter.empty:
            if parameter.name == "prev":
                raise TypeError("the prev parameter must not have a default")
            defaults[parameter.name] = parameter.default
    if sizes and defaults:
        raise TypeError("loop cannot mix parameter defaults with explicit axis specifications")
    return dict(sizes) if sizes else defaults


def require_output(output: object) -> ir.RV:
    """Validate the single-RV contract shared by the body and initializer."""
    if not isinstance(output, ir.RV):
        raise TypeError(f"loop function must return a single RV; got {type(output).__name__}")
    return output


def parse_axes(
    specifications: Mapping[str, int | SequentialAxis],
) -> tuple[dict[str, int], dict[str, Initializer]]:
    """Parse named axis specifications without inspecting or calling functions.

    Each value must be a positive integer, [size], or [size, initializer]; tuples
    are also accepted. An initializer is an RV or None. A missing
    initializer means None. Return lengths and initializers keyed by axis name.
    Both mappings preserve specification order;
    function-parameter order determines the eventual array's axis order.

    >>> parse_axes({"j": 3, "i": 2})
    ({'j': 3, 'i': 2}, {})
    >>> initializer = ir.RV(ir.Constant(0.0))
    >>> lengths, initializers = parse_axes({"i": 2, "t": [4, initializer]})
    >>> lengths, initializers["t"] is initializer
    ({'i': 2, 't': 4}, True)
    """
    lengths = {}
    initializers: dict[str, Initializer] = {}
    for name, specification in specifications.items():
        if isinstance(specification, (list, tuple)):
            if len(specification) not in (1, 2):
                raise TypeError(
                    f"axis {name!r}: a sequential axis must be [size] or [size, initializer]"
                )
            size = specification[0]
            initializer = specification[1] if len(specification) == 2 else None
            if initializer is not None and not isinstance(initializer, ir.RV):
                raise TypeError(f"axis {name!r}: initializer must be an RV or None")
            initializers[name] = initializer
        else:
            size = specification
        lengths[name] = axis_size(name, size)
    return lengths, initializers


def loop_parameter_names(
    fun: Callable[..., ir.RV],
    axis_names: Sequence[str],
    initializers: Sequence[Initializer] = (),
) -> tuple[str, ...]:
    """Validate function signatures and return coordinate names in parameter order.

    fun must have a fixed positional signature with exactly the supplied axis
    names, in any order. When initializers are supplied, fun must additionally
    start with prev; this parameter is excluded from the returned names.
    The function is not called. The name interface is reserved in fun's signature.

    >>> loop_parameter_names(lambda i, j: None, ["j", "i"])
    ('i', 'j')
    >>> loop_parameter_names(lambda prev, i, t: None, ["t", "i"], [None])
    ('i', 't')
    """
    if not callable(fun):
        raise TypeError(f"loop expected a callable; got {type(fun).__name__}")
    names = tuple(inspect.signature(fun).parameters)
    if "interface" in names:
        raise ValueError(
            "loop function parameters cannot be named 'interface'; the name is reserved"
        )
    if initializers:
        if not names or names[0] != "prev":
            raise ValueError("a sequential loop function must start with a parameter named 'prev'")
        names = names[1:]
    elif "prev" in names:
        raise ValueError("prev is only allowed when an axis is sequential")
    for name in axis_names:
        if name not in names:
            raise ValueError(f"loop got size for unknown axis {name!r}; expected {names}")
    for name in names:
        if name not in axis_names:
            raise ValueError(f"loop missing size for axis {name!r}")
    return names


def map_grid(fun: Callable[..., ir.RV], coordinates: tuple[ir.RV, ...]) -> ir.RV:
    """Evaluate a single-RV function over a Cartesian product of coordinate arrays.

    Each coordinate must be a nonempty one-dimensional RV. The function receives
    one scalar RV per coordinate and is traced once. The result has those array
    lengths as leading dimensions, in the supplied order. With no coordinates,
    call the function directly. Otherwise, outputs must be newly created, as
    required by vmap. Captured RVs remain shared; newly created randomness is
    vectorized.
    """
    if not coordinates:
        return require_output(fun())

    def row(coordinate: ir.RV) -> ir.RV:
        return map_grid(lambda *rest: fun(coordinate, *rest), coordinates[1:])

    return vmap(row)(coordinates[0])


class Previous:
    """Index a carry selected by a symbolic predecessor expression.

    Parameters
    ----------
    coordinates : sequence of RV
        Every entry must be a scalar RV.
    carries : sequence of RV or None
        Must have the same length as coordinates. An RV marks a sequential
        coordinate and supplies its carry; None marks a non-sequential coordinate.

    Given::

        previous = Previous([r_1, r_2, ..., r_N], [c_1, c_2, ..., c_N])

    an expression::

        previous[i_1, i_2, ..., i_M]

    must have some position k such that:

    - i_1 is r_1, ..., i_{k-1} is r_{k-1}, by object identity.
    - i_k has the structure RV(Sub(), r_k, RV(Constant(1))), with its
      first parent identical to r_k.
    - c_k is an RV, not None.

    The result is::

        c_k[i_{k+1}, ..., i_M]

    The number of selectors M need not equal the number of coordinates N.
    The remaining selectors are passed unchanged to::

        c_k.__getitem__((i_{k+1}, ..., i_M))

    Thus, c_k's interface determines the indexing behavior. If k == M,
    there are no remaining selectors: c_k itself is returned without
    calling c_k.__getitem__.

    Examples
    --------
    Select a carry, then index it using its interface:

    >>> from pangolin import interface as pi
    >>> r, s = pi.constant(7), pi.constant(2)
    >>> x = pi.constant([10, 20, 30])
    >>> previous = Previous([r, s], [None, x])
    >>> previous[r, s-1] is x
    True
    >>> selected = previous[r, s-1, 1]
    >>> selected.op
    Index()
    >>> selected.parents[0] is x
    True
    >>> previous[r, s-1, :].shape
    (3,)

    Different predecessor positions select different carries:

    >>> a, b = pi.constant(1), pi.constant(2)
    >>> x, y = pi.constant([10, 20]), pi.constant([[30, 40], [50, 60]])
    >>> previous = Previous([a, b], [x, y])
    >>> previous[a-1] is x
    True
    >>> previous[a, b-1] is y
    True
    >>> previous[a, b-1, 0, :].shape
    (2,)

    Everything after the first predecessor is ordinary indexing, even another
    subtraction expression:

    >>> t = pi.constant(2)
    >>> x = pi.constant([10, 20, 30])
    >>> previous = Previous([t], [x])
    >>> index = t - 1
    >>> selected = previous[index, index]
    >>> selected.parents == (x, index)
    True
    """

    def __init__(self, coordinates: Sequence[ir.RV], carries: Sequence[ir.RV | None]):
        if not isinstance(coordinates, Sequence) or not isinstance(carries, Sequence):
            raise TypeError("coordinates and carries must be sequences")
        if len(coordinates) != len(carries):
            raise ValueError("coordinates and carries must have equal lengths")
        if any(not isinstance(coordinate, ir.RV) for coordinate in coordinates):
            raise TypeError("coordinates must be RVs")
        if any(coordinate.shape != () for coordinate in coordinates):
            raise ValueError("coordinates must be scalar RVs")
        if any(carry is not None and not isinstance(carry, ir.RV) for carry in carries):
            raise TypeError("carries must be RVs or None")
        self.coordinates = tuple(coordinates)
        self.carries = tuple(carries)

    def __getitem__(self, key: object) -> ir.RV:
        indices = key if isinstance(key, tuple) else (key,)
        for axis, (index, coordinate, carry) in enumerate(
            zip(indices, self.coordinates, self.carries)
        ):
            if index is coordinate:
                continue
            if not (
                isinstance(index, ir.RV)
                and isinstance(index.op, ir.Sub)
                and index.parents[0] is coordinate
                and index.parents[1].op == ir.Constant(1)
            ):
                raise ValueError("expected the current coordinate or coordinate - 1")
            if carry is None:
                raise ValueError(f"coordinate {axis} is not sequential")
            suffix = indices[axis + 1 :]
            return carry[suffix] if suffix else carry
        raise ValueError("indexing requires a coordinate - 1 selector")


def call_initializer(fun: Callable[..., ir.RV], *coordinates: ir.RV) -> ir.RV:
    """Call an initializer once and return a newly created RV with its value.

    Return a fresh output unchanged. If the output already existed, wrap it in
    Identity so that vmap can map it without copying its random choice. Arbitrary
    contributing random nodes are permitted. No values or interfaces are coerced.
    """
    output, generated, _ = trace_call(lambda: require_output(fun(*coordinates)))
    output = require_output(output)
    if not generated:
        return ir.RV(ir.Identity(), output, interface=output.interface)
    return output


def initialize_axis(
    initializer: Callable[..., ir.RV],
    coordinates: tuple[ir.RV, ...],
    axis: int,
) -> ir.RV:
    """Build an initialization array over every coordinate except axis.

    coordinates must contain nonempty one-dimensional RVs and axis must be a
    valid position. The initializer receives one scalar per remaining coordinate,
    in order, and returns one RV. Its output shape is prefixed by the lengths of
    those coordinates. It is called once, before any recurrence is constructed.
    Fresh random nodes are vectorized; captured RVs remain shared.
    """
    return map_grid(
        lambda *indices: call_initializer(initializer, *indices),
        coordinates[:axis] + coordinates[axis + 1 :],
    )


def loop_axes(
    fun: Callable[..., ir.RV],
    coordinates: tuple[ir.RV, ...],
    initials: tuple[ir.RV | None, ...],
    *,
    names: tuple[str, ...],
) -> ir.RV:
    """Construct nested vmaps and scans from existing initialization arrays.

    Each coordinate must be a nonempty one-dimensional RV. initials has one entry
    per coordinate: None marks a vectorized axis; an RV marks a sequential axis.
    That RV has a leading dimension for every other coordinate, in order, followed
    by the shape of one body output. fun accepts Previous first when any axis is
    sequential, followed by all scalar coordinates, and returns one RV.
    names supplies coordinate names in the same order for error messages.

    Recursion selects one scalar coordinate at each level. A vectorized level
    applies vmap to the remaining construction. A sequential level selects its
    initial carry using the already selected coordinates, retaining all later
    dimensions, then scans the remaining construction. With all coordinates
    selected, call fun. The result's leading dimensions follow coordinate order.

    No initializer callback is executed here. Initial random nodes are existing
    ancestors. The body retains the usual vmap and scan restrictions, including
    scan's carry-dependence requirement.
    """

    sequential = any(initial is not None for initial in initials)
    fun_name = getattr(fun, "__name__", type(fun).__name__)

    def build(indices: tuple[ir.RV, ...], carries: tuple[ir.RV | None, ...]) -> ir.RV:
        # Both tuples describe only the axes already selected; indices are scalar.
        axis = len(indices)
        if axis == len(coordinates):
            if not sequential:
                return require_output(fun(*indices))
            return require_output(fun(Previous(indices, carries), *indices))

        initial = initials[axis]
        try:
            if initial is None:
                return vmap(lambda i: build((*indices, i), (*carries, None)))(coordinates[axis])

            start = initial
            if indices:
                remaining = tuple(ir.RV(ir.Constant(range(size))) for size in initial.shape[axis:])
                start = index_orthogonal(initial, *indices, *remaining, interface=initial.interface)

            def step(carry: ir.RV, t: ir.RV) -> ir.RV:
                return build((*indices, t), (*carries, carry))

            return scan(step, coordinates[axis].shape[0])(start, coordinates[axis])
        except Exception as error:
            kind = "vectorized" if initial is None else "sequential"
            error.add_note(f"While constructing {kind} axis {names[axis]!r} of loop {fun_name!r}.")
            raise

    return build((), ())


class InitializationPrevious:
    """Reject previous-state reads while constructing initialization values."""

    def __getitem__(self, key: object) -> ir.RV:
        raise ValueError("prev cannot be read during initialization")


def initializer_for(
    fun: Callable[..., ir.RV], axis: int, initializer: Initializer
) -> Callable[..., ir.RV]:
    """Normalize an initializer to a callback accepting all other coordinates.

    An RV is returned by reference on each call.
    None calls fun with None inserted at axis and a prev that rejects indexing.
    No callback is executed here and no value is coerced.
    """
    if isinstance(initializer, ir.RV):
        return lambda *coordinates: initializer

    def initialize(*coordinates: ir.RV) -> ir.RV:
        args = (*coordinates[:axis], None, *coordinates[axis:])
        return fun(InitializationPrevious(), *args)

    return initialize


def loop(
    fun: Callable[..., ir.RV],
    /,
    *,
    interface: ir.RVInterface,
    **sizes: int | SequentialAxis,
) -> ir.RV:
    """Build an RV over named vectorized and sequential axes. ``fun`` receives scalar
    coordinate RVs in parameter order, preceded by ``prev`` for sequential loops.
    Initial states are excluded from the output.

    Parameters
    ----------
    fun : Callable[..., ir.RV]
        Function with a fixed positional signature returning one newly created RV.
        Returning a coordinate or captured RV directly is unsupported.
        With no axes, the function is called directly and may return an existing RV.
        The parameter name ``interface`` is reserved.
        Declare axes using parameter defaults or supply sizes explicitly, never both.
    interface : ir.RVInterface
        Interface attached to the coordinate RVs. The output retains the interface
        chosen by ``fun``.
    **sizes : int | list | tuple
        Positive integer size for each vectorized coordinate, or ``[size]`` or
        ``[size, initializer]`` for each sequential coordinate. An initializer is an RV.
        With ``[size]`` or ``[size, None]``, fun is called with that coordinate set to
        None and all other coordinates supplied as RVs; prev cannot be read in this call.
        Do not supply a size for ``prev``.

    Returns
    -------
    output : ir.RV
        The output RV, with its shape prefixed by the coordinate sizes.

    Examples
    --------
    Coordinates follow function-parameter order:

    >>> from pangolin import interface as pi
    >>> x = loop(lambda i, j: 10 * i + j, interface=pi.default, j=3, i=2)
    >>> x.shape
    (2, 3)

    A nested loop returns an array-valued RV:

    >>> def model(i):
    ...     return loop(lambda j: pi.normal(i + j, 1), interface=pi.default, j=3)
    >>> result = loop(model, interface=pi.default, i=4)
    >>> result.shape
    (4, 3)

    A sequential axis can sit between vectorized axes. The previous state contains
    the complete inner array, so each component can depend on all its entries:

    >>> def evolution(prev, i, t, j):
    ...     if t is None:
    ...         return pi.normal(i + j, 1)
    ...     return pi.normal(pi.sum(prev[i, t-1, :], 0) + j, 1)
    >>> x = loop(evolution, interface=pi.default, i=2, t=[4], j=3)
    >>> x.shape
    (2, 4, 3)

    Two sequential axes use separate initialization arrays:

    >>> def model(prev, t, s):
    ...     if t is None:
    ...         return s + 1
    ...     elif s is None:
    ...         return t + 1
    ...     return prev[t-1, s] + prev[t, s-1]
    >>> x = loop(model, interface=pi.default, t=[2], s=[3])
    >>> x.shape
    (2, 3)

    An existing RV initializer remains shared. Its array shape, if any, is the
    state shape at each coordinate:

    >>> start = pi.normal(0, 1)
    >>> x = loop(lambda prev, i, t: prev[i, t-1] + 1,
    ...          interface=pi.default, i=2, t=[3, start])
    >>> x.shape
    (2, 3)

    Axis specifications can instead come from parameter defaults:

    >>> def x(prev, t=[2], s=[3]):
    ...     if t is None:
    ...         return s + 1
    ...     elif s is None:
    ...         return t + 1
    ...     return prev[t-1, s] + prev[t, s-1]
    >>> x = loop(x, interface=pi.default)
    >>> x.shape
    (2, 3)
    """
    return loop_from_specifications(fun, loop_specifications(fun, sizes), interface=interface)


def loop_from_specifications(
    fun: Callable[..., ir.RV],
    specifications: Mapping[str, int | SequentialAxis],
    *,
    interface: ir.RVInterface,
) -> ir.RV:
    """Build a loop from resolved axis specifications and a single-RV function.

    Specifications have already been extracted from defaults or explicit sizes.
    Validate their values and parameter names, then build initializers and the body.
    This also lets the interface coerce initializers before graph construction.
    """
    lengths, initializers = parse_axes(specifications)
    names = loop_parameter_names(fun, tuple(lengths), tuple(initializers.values()))
    coordinates = tuple(
        ir.RV(ir.Constant(range(lengths[name])), interface=interface) for name in names
    )
    initials: list[ir.RV | None] = []
    for axis, name in enumerate(names):
        if name not in initializers:
            initials.append(None)
            continue
        try:
            initials.append(
                initialize_axis(initializer_for(fun, axis, initializers[name]), coordinates, axis)
            )
        except Exception as error:
            fun_name = getattr(fun, "__name__", type(fun).__name__)
            error.add_note(f"While initializing axis {name!r} of loop {fun_name!r}.")
            raise
    return loop_axes(fun, coordinates, tuple(initials), names=names)
