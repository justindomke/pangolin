"""Broadcast scalar RV-producing functions using explicit broadcasting rules."""

from collections.abc import Callable
from functools import wraps
from typing import overload

import numpy as np

from pangolin import ir
from .indexing import index_orthogonal
from .vmapping import vmap

__all__ = ["broadcast_off", "broadcast_simple", "broadcast_rank", "broadcast_numpy"]


# --------------------------------------------------------------------------------------------------
# Preliminaries
# --------------------------------------------------------------------------------------------------


def broadcasting_error(shapes: tuple[ir.Shape, ...], mode: str) -> ValueError:
    """Create an error describing incompatible shapes and the broadcasting rule."""
    formatted_shapes = " and ".join(str(shape) for shape in shapes)
    return ValueError(f"Can't broadcast shapes {formatted_shapes} with broadcasting={mode}")


def wrap_broadcasting(
    scalar_fun: Callable[..., ir.RV],
    apply: Callable[[Callable[..., ir.RV], tuple[ir.RV, ...], tuple[ir.Shape, ...]], ir.RV],
) -> Callable[..., ir.RV]:
    """Broadcast supplied positional and keyword RVs without inspecting the function signature."""

    @wraps(scalar_fun)
    def broadcast_fun(*args: ir.RV, **kwargs: ir.RV) -> ir.RV:
        rv_args = (*args, *kwargs.values())
        for arg in rv_args:
            if not isinstance(arg, ir.RV):
                raise TypeError("broadcasting arguments must be RVs")
        shapes = tuple(arg.shape for arg in rv_args)

        def positional_fun(*values: ir.RV) -> ir.RV:
            return scalar_fun(
                *values[: len(args)],
                **dict(zip(kwargs, values[len(args) :], strict=True)),
            )

        return apply(positional_fun, rv_args, shapes)

    return broadcast_fun


# --------------------------------------------------------------------------------------------------
# Broadcasting: OFF
# --------------------------------------------------------------------------------------------------


@overload
def broadcast_off(
    scalar_fun: Callable[[], ir.RV],
) -> Callable[[], ir.RV]: ...


@overload
def broadcast_off(
    scalar_fun: Callable[[ir.RV], ir.RV],
) -> Callable[[ir.RV], ir.RV]: ...


@overload
def broadcast_off(
    scalar_fun: Callable[[ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_off(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_off(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_off(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_off[*Rest](
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest], ir.RV]: ...


def broadcast_off(scalar_fun: Callable[..., ir.RV]) -> Callable[..., ir.RV]:
    """Require scalar RV inputs when calling a scalar function.

    This combinator returns a function that calls ``scalar_fun`` with the supplied arguments.
    RV interfaces are preserved without interpreting them.

    Parameters
    ----------
    scalar_fun
        Function accepting scalar RV arguments and returning one scalar RV.

    Raises
    ------
    TypeError
        If a supplied argument is not an RV or the underlying function rejects the call.
    ValueError
        If any input RV is not scalar.

    Notes
    -----
    Only explicitly supplied arguments participate in broadcasting. Each must be an RV; the
    combinator performs no coercion. Positional and keyword arguments follow identical shape rules,
    including values supplied to keyword-only parameters or collected by ``**kwargs``.

    The combinator does not inspect the function signature or insert default arguments. Omitted
    defaults are handled normally by Python when ``scalar_fun`` is called, just like values captured
    inside the function. Their shapes do not determine the broadcast shape. Supplying a default
    value explicitly makes it a broadcast input, like any other supplied argument.

    Keyword calls are supported at runtime, but the static overloads describe positional calls only.

    Examples
    --------
    >>> def exp(x):
    ...     return ir.RV(ir.Exp(), x)
    >>> scalar_exp = broadcast_off(exp)
    >>> scalar_exp(ir.RV(ir.Constant(2))).shape
    ()
    """
    return wrap_broadcasting(scalar_fun, apply_without_broadcasting)


def apply_without_broadcasting(
    scalar_fun: Callable[..., ir.RV], args: tuple[ir.RV, ...], shapes: tuple[ir.Shape, ...]
) -> ir.RV:
    """Apply a scalar function without broadcasting."""
    if any(shapes):
        raise broadcasting_error(shapes, "OFF")
    return scalar_fun(*args)


# --------------------------------------------------------------------------------------------------
# Broadcasting: SIMPLE
# --------------------------------------------------------------------------------------------------


@overload
def broadcast_simple(
    scalar_fun: Callable[[], ir.RV],
) -> Callable[[], ir.RV]: ...


@overload
def broadcast_simple(
    scalar_fun: Callable[[ir.RV], ir.RV],
) -> Callable[[ir.RV], ir.RV]: ...


@overload
def broadcast_simple(
    scalar_fun: Callable[[ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_simple(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_simple(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_simple(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_simple[*Rest](
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest], ir.RV]: ...


def broadcast_simple(scalar_fun: Callable[..., ir.RV]) -> Callable[..., ir.RV]:
    """Broadcast a scalar function over scalars and equal-shaped arrays.

    This combinator returns a function applying ``scalar_fun`` elementwise using nested `vmap` calls.
    Scalars are broadcast to the common array shape. Tracing preserves input interfaces and the
    output interface selected by ``scalar_fun``.

    Parameters
    ----------
    scalar_fun
        Function accepting scalar RV arguments and returning one scalar RV.

    Raises
    ------
    TypeError
        If a supplied argument is not an RV or the underlying function rejects the call.
    ValueError
        If the non-scalar input shapes differ.

    Notes
    -----
    Only explicitly supplied arguments participate in broadcasting. Each must be an RV; the
    combinator performs no coercion. Positional and keyword arguments follow identical shape rules,
    including values supplied to keyword-only parameters or collected by ``**kwargs``.

    The combinator does not inspect the function signature or insert default arguments. Omitted
    defaults are handled normally by Python when ``scalar_fun`` is called, just like values captured
    inside the function. Their shapes do not determine the broadcast shape. Supplying a default
    value explicitly makes it a broadcast input, like any other supplied argument.

    Keyword calls are supported at runtime, but the static overloads describe positional calls only.

    Examples
    --------
    >>> def normal(mu, sigma):
    ...     return ir.RV(ir.Normal(), mu, sigma)
    >>> broadcasted_normal = broadcast_simple(normal)
    >>> mu = ir.RV(ir.Constant([0, 1, 2]))
    >>> sigma = ir.RV(ir.Constant(1))
    >>> broadcasted_normal(mu, sigma).shape
    (3,)
    """
    return wrap_broadcasting(scalar_fun, apply_with_simple_broadcasting)


def simple_broadcast_shape(*shapes: ir.Shape) -> ir.Shape:
    """Find the common shape under scalar-or-exact-shape broadcasting."""
    array_shapes = [shape for shape in shapes if shape]
    if not array_shapes:
        return ()
    if any(shape != array_shapes[0] for shape in array_shapes[1:]):
        raise broadcasting_error(shapes, "SIMPLE")
    return array_shapes[0]


def apply_with_simple_broadcasting(
    scalar_fun: Callable[..., ir.RV], args: tuple[ir.RV, ...], shapes: tuple[ir.Shape, ...]
) -> ir.RV:
    """Apply scalar-or-exact-shape broadcasting to a scalar function."""
    simple_broadcast_shape(*shapes)
    return apply_with_rank_broadcasting(scalar_fun, args, shapes)


# --------------------------------------------------------------------------------------------------
# Broadcasting: RANK
# --------------------------------------------------------------------------------------------------


@overload
def broadcast_rank(
    scalar_fun: Callable[[], ir.RV],
) -> Callable[[], ir.RV]: ...


@overload
def broadcast_rank(
    scalar_fun: Callable[[ir.RV], ir.RV],
) -> Callable[[ir.RV], ir.RV]: ...


@overload
def broadcast_rank(
    scalar_fun: Callable[[ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_rank(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_rank(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_rank(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_rank[*Rest](
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest], ir.RV]: ...


def broadcast_rank(scalar_fun: Callable[..., ir.RV]) -> Callable[..., ir.RV]:
    """Broadcast a scalar function using leading-rank promotion.

    Each input shape must be a suffix of the longest shape. This combinator returns a function applying
    ``scalar_fun`` elementwise using nested `vmap` calls, broadcasting missing leading axes.
    Existing singleton axes are not expanded. Tracing preserves input interfaces and the output
    interface selected by ``scalar_fun``.

    Parameters
    ----------
    scalar_fun
        Function accepting scalar RV arguments and returning one scalar RV.

    Raises
    ------
    TypeError
        If a supplied argument is not an RV or the underlying function rejects the call.
    ValueError
        If an input shape is not a suffix of the longest shape.

    Notes
    -----
    Only explicitly supplied arguments participate in broadcasting. Each must be an RV; the
    combinator performs no coercion. Positional and keyword arguments follow identical shape rules,
    including values supplied to keyword-only parameters or collected by ``**kwargs``.

    The combinator does not inspect the function signature or insert default arguments. Omitted
    defaults are handled normally by Python when ``scalar_fun`` is called, just like values captured
    inside the function. Their shapes do not determine the broadcast shape. Supplying a default
    value explicitly makes it a broadcast input, like any other supplied argument.

    Keyword calls are supported at runtime, but the static overloads describe positional calls only.

    Examples
    --------
    >>> def add(x, y):
    ...     return ir.RV(ir.Add(), x, y)
    >>> broadcasted_add = broadcast_rank(add)
    >>> x = ir.RV(ir.Constant([1, 2, 3]))
    >>> y = ir.RV(ir.Constant([[4, 5, 6], [7, 8, 9]]))
    >>> broadcasted_add(x, y).shape
    (2, 3)
    """
    return wrap_broadcasting(scalar_fun, apply_with_rank_broadcasting)


def rank_broadcast_shape(*shapes: ir.Shape) -> ir.Shape:
    """Find the common shape under leading-rank promotion."""
    result_shape = max(shapes, key=len, default=())
    for shape in shapes:
        if shape != result_shape[len(result_shape) - len(shape) :]:
            raise broadcasting_error(shapes, "RANK")
    return result_shape


def apply_with_rank_broadcasting(
    scalar_fun: Callable[..., ir.RV], args: tuple[ir.RV, ...], shapes: tuple[ir.Shape, ...]
) -> ir.RV:
    """Apply rank-promotion broadcasting to a scalar function."""
    shape = rank_broadcast_shape(*shapes)
    mapped_fun = scalar_fun
    for dimension, size in enumerate(reversed(shape)):
        axes = tuple(0 if len(arg_shape) > dimension else None for arg_shape in shapes)
        in_axes = axes[0] if len(axes) == 1 else axes
        mapped_fun = vmap(mapped_fun, in_axes, size)
    return mapped_fun(*args)


# --------------------------------------------------------------------------------------------------
# Broadcasting: NUMPY
# --------------------------------------------------------------------------------------------------


@overload
def broadcast_numpy(
    scalar_fun: Callable[[], ir.RV],
) -> Callable[[], ir.RV]: ...


@overload
def broadcast_numpy(
    scalar_fun: Callable[[ir.RV], ir.RV],
) -> Callable[[ir.RV], ir.RV]: ...


@overload
def broadcast_numpy(
    scalar_fun: Callable[[ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_numpy(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_numpy(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_numpy(
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]: ...


@overload
def broadcast_numpy[*Rest](
    scalar_fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest], ir.RV],
) -> Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest], ir.RV]: ...


def broadcast_numpy(scalar_fun: Callable[..., ir.RV]) -> Callable[..., ir.RV]:
    """Broadcast a scalar function using NumPy's broadcasting rules.

    Input shapes are aligned on the right; missing leading axes and singleton axes may be
    expanded. This combinator returns a function applying ``scalar_fun`` elementwise using nested
    `vmap` calls. Tracing preserves input interfaces and the output interface selected by
    ``scalar_fun``.

    Parameters
    ----------
    scalar_fun
        Function accepting scalar RV arguments and returning one scalar RV.

    Raises
    ------
    TypeError
        If a supplied argument is not an RV or the underlying function rejects the call.
    ValueError
        If the input shapes cannot be broadcast together.

    Notes
    -----
    Only explicitly supplied arguments participate in broadcasting. Each must be an RV; the
    combinator performs no coercion. Positional and keyword arguments follow identical shape rules,
    including values supplied to keyword-only parameters or collected by ``**kwargs``.

    The combinator does not inspect the function signature or insert default arguments. Omitted
    defaults are handled normally by Python when ``scalar_fun`` is called, just like values captured
    inside the function. Their shapes do not determine the broadcast shape. Supplying a default
    value explicitly makes it a broadcast input, like any other supplied argument.

    Keyword calls are supported at runtime, but the static overloads describe positional calls only.

    Examples
    --------
    >>> def add(x, y):
    ...     return ir.RV(ir.Add(), x, y)
    >>> broadcasted_add = broadcast_numpy(add)
    >>> x = ir.RV(ir.Constant([[1], [2]]))
    >>> y = ir.RV(ir.Constant([3, 4, 5]))
    >>> broadcasted_add(x, y).shape
    (2, 3)
    """
    return wrap_broadcasting(scalar_fun, apply_with_numpy_broadcasting)


def numpy_broadcast_shape(*shapes: ir.Shape) -> ir.Shape:
    """Find the common shape under NumPy broadcasting."""
    try:
        return np.broadcast_shapes(*shapes)
    except ValueError:
        raise broadcasting_error(shapes, "NUMPY") from None


def apply_with_numpy_broadcasting(
    scalar_fun: Callable[..., ir.RV], args: tuple[ir.RV, ...], shapes: tuple[ir.Shape, ...]
) -> ir.RV:
    """Apply full NumPy broadcasting to a scalar function."""
    shape = numpy_broadcast_shape(*shapes)
    if not shape:
        return scalar_fun(*args)

    mapped_fun = scalar_fun
    for dimension, size in enumerate(reversed(shape)):
        axes = []
        expanded_singletons = []
        for arg_shape in shapes:
            if len(arg_shape) <= dimension:
                axes.append(None)
                expanded_singletons.append(False)
            elif arg_shape[-dimension - 1] == size:
                axes.append(0)
                expanded_singletons.append(False)
            else:
                axes.append(None)
                expanded_singletons.append(True)

        in_axes = axes[0] if len(axes) == 1 else tuple(axes)
        mapped_fun = vmap(mapped_fun, in_axes, size)
        if any(expanded_singletons):
            mapped_fun = squeeze_singletons(mapped_fun, tuple(expanded_singletons))

    return mapped_fun(*args)


def squeeze_singletons(
    mapped_fun: Callable[..., ir.RV], expanded_singletons: tuple[bool, ...]
) -> Callable[..., ir.RV]:
    """Remove singleton axes immediately before a mapped scalar call."""

    def call(*args: ir.RV) -> ir.RV:
        squeezed_args = tuple(
            remove_leading_singleton(arg) if expand else arg
            for arg, expand in zip(args, expanded_singletons, strict=True)
        )
        return mapped_fun(*squeezed_args)

    return call


def remove_leading_singleton(arg: ir.RV) -> ir.RV:
    """Remove a leading singleton axis using orthogonal indexing."""
    if not arg.shape or arg.shape[0] != 1:
        raise ValueError(f"expected a leading singleton axis, got shape {arg.shape}")
    indices = (
        ir.RV(ir.Constant(0), interface=arg.interface),
        *(ir.RV(ir.Constant(np.arange(size)), interface=arg.interface) for size in arg.shape[1:]),
    )
    return index_orthogonal(arg, *indices, interface=arg.interface)
