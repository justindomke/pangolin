"""Construct a named sequential loop containing any number of vectorized axes."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Protocol, overload

from pangolin import ir
from .scanning import scan
from .vfor_looping import axis_size, vfor_parameter_names, vfor_simple


class SForDecorator(Protocol):
    """Static type of a decorator returned by `sfor`."""

    @overload
    def __call__(self, fun: Callable[[ir.RV, ir.RV], ir.RV], /) -> ir.RV: ...

    @overload
    def __call__(self, fun: Callable[[ir.RV, ir.RV, ir.RV], ir.RV], /) -> ir.RV: ...

    @overload
    def __call__(
        self,
        fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
        /,
    ) -> ir.RV: ...

    @overload
    def __call__(
        self,
        fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
        /,
    ) -> ir.RV: ...

    @overload
    def __call__[*Rest](
        self,
        fun: Callable[
            [ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest],
            ir.RV,
        ],
        /,
    ) -> ir.RV: ...


def require_single_rv(output: object) -> ir.RV:
    """Require the deliberately narrow single-RV `sfor` output contract."""

    if not isinstance(output, ir.RV):
        raise TypeError("sfor function must return exactly one ir.RV")
    return output


@overload
def sfor(
    fun: Callable[[ir.RV, ir.RV], ir.RV],
    /,
    *,
    init: ir.RV,
    interface: ir.RVInterface,
    **sizes: int,
) -> ir.RV: ...


@overload
def sfor(
    fun: Callable[[ir.RV, ir.RV, ir.RV], ir.RV],
    /,
    *,
    init: ir.RV,
    interface: ir.RVInterface,
    **sizes: int,
) -> ir.RV: ...


@overload
def sfor(
    fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
    /,
    *,
    init: ir.RV,
    interface: ir.RVInterface,
    **sizes: int,
) -> ir.RV: ...


@overload
def sfor(
    fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
    /,
    *,
    init: ir.RV,
    interface: ir.RVInterface,
    **sizes: int,
) -> ir.RV: ...


@overload
def sfor[*Rest](
    fun: Callable[
        [ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest],
        ir.RV,
    ],
    /,
    *,
    init: ir.RV,
    interface: ir.RVInterface,
    **sizes: int,
) -> ir.RV: ...


@overload
def sfor(
    fun: None = None,
    /,
    *,
    init: ir.RV,
    interface: ir.RVInterface,
    **sizes: int,
) -> SForDecorator: ...


def sfor(
    fun: Callable[..., ir.RV] | None = None,
    /,
    *,
    init: ir.RV,
    interface: ir.RVInterface,
    **sizes: int,
) -> ir.RV | SForDecorator:
    """Build a state sequence with named sequential and parallel coordinates.

    This combinator constructs the result directly. A scalar call of the form::

        x = sfor(fun, init=init, interface=interface, t=T)

    has these semantics::

        previous = init
        states = []
        for t in range(T):
            t_rv = ir.RV(ir.Constant(t), interface=interface)
            next_state = fun(previous, t_rv)
            states.append(next_state)
            previous = next_state
        x = stack(states)

    For example, if::

        fun = lambda previous, t: normal(previous, 1)

    then::

        x[0] ~ normal(init, 1)
        x[t] ~ normal(x[t-1], 1), t > 0

    A call with parallel coordinates::

        x = sfor(fun, init=init, interface=interface, t=T, i=I, j=J)

    has these semantics::

        previous = init
        states = []
        for t in range(T):
            t_rv = ir.RV(ir.Constant(t), interface=interface)
            next_state = empty((I, J))
            for i in range(I):
                i_rv = ir.RV(ir.Constant(i), interface=interface)
                for j in range(J):
                    j_rv = ir.RV(ir.Constant(j), interface=interface)
                    next_state[i, j] = fun(previous, t_rv, i_rv, j_rv)
            states.append(next_state)
            previous = next_state
        x = stack(states)

    This pseudocode describes the meaning of the loops, not the graph constructed internally.
    Internally, each coordinate range is a constant RV array with the supplied interface, and
    the combinators trace ``fun`` with abstract scalar coordinates carrying that same interface.
    The loops are represented by one `Scan` containing nested `VMap` operations; they are not
    executed in Python. Function-parameter order determines the nesting, while keyword order has
    no effect.

    For example, if::

        fun = lambda previous, t, i, j: normal(a[i, j] * previous[i, j] + b[t, i, j], c[i, j])

    then::

        x[0, i, j] ~ normal(a[i, j] * init[i, j] + b[0, i, j], c[i, j])
        x[t, i, j] ~ normal(a[i, j] * x[t-1, i, j] + b[t, i, j], c[i, j]), t > 0

    Finally, decorator form::

        @sfor(init=init, interface=interface, t=T)
        def x(previous, t):
            return normal(previous, 1)

    is equivalent to::

        def fun(previous, t):
            return normal(previous, 1)

        x = sfor(fun, init=init, interface=interface, t=T)

    In both forms, ``init`` and ``interface`` must be supplied as keyword arguments.

    Each call to ``fun`` must return exactly one `ir.RV` created during that call. The result
    must depend on ``previous``. Any other RVs created during the call that contribute to the
    result must have deterministic Ops. The assembled state must have the same interface as
    ``init``.

    The shape of ``init`` must match one complete state assembled from ``fun``. For
    ``fun(previous, t, i, j)``, let ``S`` be the shape returned by one call to ``fun``. With
    ``t=T``, ``i=I``, and ``j=J``, ``init`` must have shape ``(I, J, *S)``, and the result has shape
    ``(T, I, J, *S)``. Equivalently, ``result.shape == (T,) + init.shape``. Note that the output
    axes appear in function-parameter order; keyword order is ignored.

    Each parallel call receives the complete previous state; ``sfor`` does not slice it along the
    parallel axes.

    Parameters
    ----------
    fun : Callable[..., ir.RV] | None
        Step function, or ``None`` in decorator form. It must have a fixed positional signature
        with at least two parameters and no defaults, variadic parameters, or keyword-only
        parameters. The first parameter receives the complete previous state; the second receives
        the scalar sequential coordinate; any later parameters receive scalar parallel
        coordinates. No parameter may be named ``init`` or ``interface``. The function must satisfy
        the return, carry-dependence, determinism, shape, and interface requirements above.
    init : ir.RV
        Keyword-only complete state before the first step. It must already be an RV; it is not
        coerced. Its shape and interface must match the state assembled by mapping ``fun`` over
        every parallel coordinate.
    interface : ir.RVInterface
        Required keyword-only interface attached to the generated sequential and parallel
        coordinate RVs. It need not match ``init.interface``; the operations used by ``fun``
        determine whether those interfaces can be used together.
    **sizes : int
        Exactly one positive integer for every coordinate parameter: the sequential parameter and
        all parallel parameters. Names must match the function parameters. Values determine axis
        lengths; function-parameter order, not keyword order, determines axis order.

    Returns
    -------
    result_or_decorator : ir.RV | decorator
        When ``fun`` is supplied, an `ir.RV` containing every state produced by the scan, not
        including ``init``. If the sequential size is ``T``, ``result[t]`` is the state produced at
        coordinate ``t`` and ``result.shape == (T,) + init.shape``. When ``fun`` is omitted, a
        decorator that accepts it and returns such an `ir.RV`. The result has ``init.interface``.

    Raises
    ------
    TypeError
        If ``fun`` is not callable or lacks the required signature, ``init`` is not an RV, a size
        is not an integer, or ``fun`` does not return exactly one `ir.RV`.
    ValueError
        If a parameter is named ``init`` or ``interface``, the function has no sequential coordinate,
        sizes are missing, unknown, or nonpositive, the result does not depend on ``previous``, an
        intermediate Op is random, or the mapped state does not match the shape or interface of
        ``init``.

    """

    if fun is None:

        def apply_sfor(actual_fun: Callable[..., ir.RV]) -> ir.RV:
            return sfor(actual_fun, init=init, interface=interface, **sizes)

        return apply_sfor

    if not callable(fun):
        raise TypeError(f"sfor expected a callable; got {type(fun).__name__}")

    parameter_names = vfor_parameter_names(fun)
    for name in ("init", "interface"):
        if name in parameter_names:
            raise ValueError(
                f"sfor function parameters cannot be named {name!r}; the name is reserved"
            )
    coordinate_names = parameter_names[1:]
    if not coordinate_names:
        raise ValueError("sfor function must accept a state and a sequential coordinate")

    sequential_name, *parallel_names = coordinate_names
    unknown_names = [name for name in sizes if name not in coordinate_names]
    if unknown_names:
        name = unknown_names[0]
        raise ValueError(f"sfor got size for unknown axis {name!r}; expected {coordinate_names}")

    missing_names = [name for name in coordinate_names if name not in sizes]
    if missing_names:
        name = missing_names[0]
        raise ValueError(f"sfor missing size for axis {name!r}")

    if not isinstance(init, ir.RV):
        raise TypeError(f"sfor initial state must be an RV; got {type(init).__name__}")
    sequential_size = axis_size(sequential_name, sizes[sequential_name])
    parallel_sizes = tuple(axis_size(name, sizes[name]) for name in parallel_names)
    sequential_coordinates = ir.RV(ir.Constant(range(sequential_size)), interface=interface)

    def step(previous: ir.RV, sequential_coordinate: ir.RV) -> ir.RV:
        if parallel_sizes:
            parallel_body = partial(fun, previous, sequential_coordinate)
            output = vfor_simple(parallel_body, *parallel_sizes, interface=interface)
        else:
            output = fun(previous, sequential_coordinate)
        return require_single_rv(output)

    return scan(step, sequential_size, 0)(init, sequential_coordinates)


__all__ = ["sfor"]
