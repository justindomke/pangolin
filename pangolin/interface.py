"""This module defines a friendly interface for creating models in the Pangolin IR.

Working with RVs
----------------

An RV represents a scalar or array-shaped value in a probabilistic model. Calling a
distribution function creates an RV; it does not draw a sample. Arithmetic expressions
create additional RVs:

>>> from pangolin import interface as pi
>>> slope = pi.normal(0, 1)
>>> mean = 2 * slope + 1
>>> observation = pi.normal(mean, 0.5)
>>> pi.print_upstream(observation, locals())
shape | statement
----- | ---------
()    | a = 2
()    | b = 0
()    | c = 1
()    | slope ~ normal(b, c)
()    | d = mul(a, slope)
()    | e = 1
()    | mean = add(d, e)
()    | f = 0.5
()    | observation ~ normal(mean, f)

Most graph-building functions accept :data:`RVLike` inputs: RVs, numbers, or numeric arrays.
Numbers and arrays become constant RVs automatically.

Every RV has a fixed shape, available as ``x.shape``. Scalars have shape ``()``:

>>> x = pi.normal(0, 1)
>>> x.shape
()
>>> y = pi.constant([1, 2, 3])
>>> y.shape
(3,)

You can use familiar Python notation instead of calling the corresponding functions:

.. list-table::
   :header-rows: 1

   * - Notation
     - Equivalent function call
   * - ``x + y``
     - ``pi.add(x, y)``
   * - ``x - y``
     - ``pi.sub(x, y)``
   * - ``x * y``
     - ``pi.mul(x, y)``
   * - ``x / y``
     - ``pi.div(x, y)``
   * - ``x ** y``
     - ``pi.pow(x, y)``
   * - ``abs(x)``
     - ``pi.abs(x)``
   * - ``x @ y``
     - ``pi.matmul(x, y)``
   * - ``x.T``
     - ``pi.transpose(x)``
   * - ``x[i, j]``
     - ``pi.oindex(x, i, j)``

For RVs created using the default interface, these are shorthand for the corresponding
function calls and follow the same rules.

Broadcasting is configurable. The default interface uses simple broadcasting: scalar
arguments can be combined with arrays, while all non-scalar arguments must have the same
shape. This applies to scalar distribution functions and scalar arithmetic:

>>> x = pi.normal([0, 1, 2], 1)
>>> x.shape
(3,)
>>> y = 2 * x + 1
>>> y.shape
(3,)

You can disable broadcasting, allow rank promotion, or use NumPy-style broadcasting.
Use ``pi.overload(...)`` for a temporary change or create a separately configured
:class:`Interface`. See :ref:`interface-broadcasting` for the rules and examples.

Combinators take ordinary Python graph-building functions and return new graph-building
functions. For example, ``vmap`` turns a function into one that operates across an array:

>>> def shifted_normal(mean):
...     return pi.normal(mean, 1) + 2
>>> mapped_normal = pi.vmap(shifted_normal)
>>> means = pi.constant([0, 1, 2])
>>> x = mapped_normal(means)
>>> x.shape
(3,)

Combinators can be combined: for example, ``vmap`` can map a function built using ``scan``
to create multiple time series. See :ref:`interface-combinators` for the available operations.

* See :doc:`interface.examples` for examples using arrays, time series, and nested combinators.
* See :doc:`interface.broadcasting` for broadcasting rules and configuration options.
* See :doc:`interface.indexing` for indexing rules and examples.

Reference card
--------------

The entire interface is based on methods that create :class:`pangolin.ir.RV` objects (or that create functions that create :class:`pangolin.ir.RV` objects).

Here are the main public tools:

============================ ======
To do...                     Use...
============================ ======
Constants                    :func:`constant`
Arithmetic                   :func:`add` :func:`sub` :func:`mul` :func:`div` (or infix ``+``, ``-``, ``*``, ``/``)
Trigonometry                 :func:`arccos` :func:`arccosh` :func:`arcsin` :func:`arcsinh` :func:`arctan` :func:`arctanh` :func:`cos` :func:`cosh` :func:`sin` :func:`sinh` :func:`tan` :func:`tanh`
Other scalar functions       :func:`pow` (or infix ``**``) :func:`sqrt` :func:`abs` :func:`minimum` :func:`maximum` :func:`exp` :func:`inv_logit` :func:`expit` :func:`sigmoid` :func:`log` :func:`loggamma` :func:`logit` :func:`step`
Linear algebra               :func:`matmul` (or infix ``@``) :func:`transpose` (or ``rv.T``) :func:`inv` :func:`cholesky`
Other multivariate functions :func:`concatenate` :func:`sum` :func:`softmax` :func:`diag` :func:`diag_matrix` :func:`fill_tril` :func:`extract_tril`
Indexing                     :func:`oindex` (or ``rv[...]``) :func:`vindex`
Scalar distributions         :func:`normal` :func:`normal_prec` :func:`lognormal` :func:`cauchy` :func:`bernoulli` :func:`bernoulli_logit` :func:`beta` :func:`binomial` :func:`categorical` :func:`uniform` :func:`beta_binomial` :func:`exponential` :func:`gamma` :func:`poisson` :func:`student_t`
Multivariate distributions   :func:`multi_normal` :func:`multinomial` :func:`dirichlet` :func:`wishart`
Bijected densities           :class:`Transform` :doc:`tforms <interface.tforms>` :func:`compose_transforms`
Functional combinators       :func:`broadcast` :func:`vmap` :func:`loop` :func:`seq` :func:`vfor` :func:`sfor` :func:`composite` :func:`scan` :func:`log_prob` :func:`cdf` :func:`quantile` :func:`truncated` :func:`grad` :func:`vjp`
Graph inspection             :func:`upstream_nodes` :func:`print_upstream` :func:`graph_upstream` :func:`ir_code` :func:`signature`
Configuration                :class:`Interface` :func:`overload`
============================ ======

"""

from __future__ import annotations

import builtins
import inspect
import typing
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import EllipsisType
from functools import partial, wraps
from typing import Any, Concatenate, Final, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from pangolin import functional, ir
from pangolin.dag import upstream_nodes
from pangolin.functional import (
    composite,
    ir_code,
    signature,
    vmap,
    scan,
    Transform,
    compose_transforms,
    vjp,
)
from pangolin.functional.sfor_looping import SForDecorator
from pangolin.functional.looping import loop_specifications, loop_from_specifications
from pangolin.functional.vfor_looping import VForDecorator
from pangolin.ir import print_upstream, graph_upstream
from pangolin.util import PyTree

__all__ = [
    "Interface",
    "RVLike",
    "default",
    "overload",
    "constant",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sqrt",
    "neg",
    "abs",
    "minimum",
    "maximum",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "exp",
    "inv_logit",
    "expit",
    "sigmoid",
    "log",
    "logit",
    "step",
    "loggamma",
    "normal",
    "normal_prec",
    "lognormal",
    "bernoulli",
    "bernoulli_logit",
    "binomial",
    "cauchy",
    "uniform",
    "beta",
    "exponential",
    "gamma",
    "poisson",
    "beta_binomial",
    "student_t",
    "matmul",
    "transpose",
    "diag",
    "diag_matrix",
    "inv",
    "cholesky",
    "softmax",
    "sum",
    "concatenate",
    "fill_tril",
    "extract_tril",
    "multi_normal",
    "categorical",
    "multinomial",
    "dirichlet",
    "wishart",
    "oindex",
    "vindex",
    "loop",
    "seq",
    "vfor",
    "sfor",
    "broadcast",
    "composite",
    "log_prob",
    "cdf",
    "quantile",
    "truncated",
    "grad",
    "vjp",
    "vmap",
    "scan",
    "Transform",
    "compose_transforms",
    "tforms",
    "upstream_nodes",
    "print_upstream",
    "ir_code",
    "signature",
    "graph_upstream",
]


type RVLike = ir.RV | ArrayLike
type Index = RVLike | slice | EllipsisType
type LoopInitializer = RVLike | None
type LoopAxis = int | tuple[int] | tuple[int, LoopInitializer] | list[int | LoopInitializer]


def seq(size: int, initializer: LoopInitializer = None) -> tuple[int, LoopInitializer]:
    """Specify a sequential axis for `loop`, returning ``(size, initializer)``.

    Parameters
    ----------
    size
        Number of sequential steps.
    initializer
        Initial RV or numeric value. None uses the loop function's initialization branch.

    Examples
    --------
    >>> from pangolin import interface as pi
    >>> pi.seq(5)
    (5, None)
    >>> pi.seq(5, 0.0)
    (5, 0.0)
    >>> @pi.loop
    ... def x(prev, i=3, t=pi.seq(5)):
    ...     if t is None:
    ...         return pi.normal(i, 1)
    ...     return pi.normal(prev[i, t-1], 1)
    >>> x.shape
    (3, 5)
    """
    return size, initializer


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Store the immutable configuration of a Pangolin interface.

    Parameters
    ----------
    broadcasting
        Broadcasting policy for scalar graph-building functions. ``"off"`` requires scalar
        inputs, ``"simple"`` additionally permits scalars mixed with equal-shaped arrays,
        ``"rank"`` additionally permits rank promotion, and ``"numpy"`` uses NumPy broadcasting.
    """

    broadcasting: Literal["off", "simple", "rank", "numpy"] = "simple"

    def __post_init__(self) -> None:
        if not isinstance(self.broadcasting, str):
            raise TypeError("broadcasting must be a string")
        if self.broadcasting not in ("off", "simple", "rank", "numpy"):
            raise ValueError(
                "broadcasting must be one of ['off', 'simple', 'rank', 'numpy']; "
                f"got {self.broadcasting!r}"
            )


# --------------------------------------------------------------------------------------------------
# Scalar functions
# --------------------------------------------------------------------------------------------------


broadcasting_combinators = {
    "off": functional.broadcast_off,
    "simple": functional.broadcast_simple,
    "rank": functional.broadcast_rank,
    "numpy": functional.broadcast_numpy,
}


def scalar_op_class(name: str) -> type[ir.ScalarOp]:
    """Find the built-in scalar Op whose interface name is ``name``."""
    matches = [
        op_class
        for op_class in ir.ScalarOp.__subclasses__()
        if op_class.__module__ == ir.__name__ and str(op_class()) == name
    ]
    if len(matches) != 1:
        raise TypeError(f"cannot identify one built-in ScalarOp for {name!r}")
    return matches[0]


def scalar_parent_descriptions(
    op_class: type[ir.ScalarOp],
    parent_names: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Return the parent names and descriptions declared by a scalar Op."""
    expected_parents = op_class._expected_parents
    if isinstance(expected_parents, int):
        if parent_names is None:
            parent_names = tuple(
                chr(ord("a") + position) for position in builtins.range(expected_parents)
            )
        if len(parent_names) != expected_parents:
            raise TypeError(
                f"{op_class.__name__} expects {expected_parents} parents, got {len(parent_names)}"
            )
        return {name: f"parent {position}" for position, name in enumerate(parent_names)}

    expected_names = tuple(expected_parents)
    if parent_names is not None and parent_names != expected_names:
        raise TypeError(
            f"{op_class.__name__} declares parent names {expected_names}, got {parent_names}"
        )
    return dict(expected_parents)


def scalar_doc(
    op_class: type[ir.ScalarOp],
    parent_names: tuple[str, ...] | None = None,
    module_example: bool = False,
) -> str:
    """Create an interface-function docstring from a scalar Op declaration."""
    op = op_class()
    parent_descriptions = scalar_parent_descriptions(op_class, parent_names)
    if op.random:
        wikipedia = ir._get_scalar_op_wikipedia(op_class)
        reference = f" (`Wikipedia <{wikipedia}>`__)" if wikipedia else ""
        summary = f"Create a `{op.name}` distributed RV{reference}."
    else:
        summary = f"Create an RV by applying `pangolin.ir.{op.name}` to its parents."

    broadcasting = (
        "This function belongs to the default interface. Thus, all arguments must either be scalar "
        "or mutually broadcastable using :ref:`simple broadcasting <interface-broadcasting>`, "
        "unless changed within an :func:`~pangolin.interface.overload` block."
        if module_example
        else "All arguments must either be scalar or mutually broadcastable according to\n"
        "the broadcasting policy selected when constructing the interface or within an overload block."
    )
    lines = [
        summary,
        "",
        broadcasting,
        "",
        "Parameters",
        "----------",
    ]
    for name, description in parent_descriptions.items():
        lines.extend([name, f"    {description}."])

    num_parents = len(parent_descriptions)
    parent_word = "parent" if num_parents == 1 else "parents"
    lines.extend(
        [
            "",
            "Returns",
            "-------",
            "ir.RV",
            f"    RV whose operation is `pangolin.ir.{op.name}` when all parents are scalar, or",
            "    one or more nested `pangolin.ir.VMap` operations around that type when",
            f"    broadcasting is required. The operation has {num_parents} {parent_word}.",
            "",
            "Examples",
            "--------",
        ]
    )
    if module_example:
        lines.append(">>> import pangolin.interface as pi")
    else:
        lines.append(">>> pi = Interface()")
    arguments = [0.1 * 2**position for position in builtins.range(num_parents)]
    argument_text = ", ".join(map(str, arguments))
    parent_text = ", ".join(f"RV(Constant({argument}))" for argument in arguments)
    lines.extend(
        [
            f">>> x = pi.{op}({argument_text})",
            ">>> x",
            f"RV({op!r}, {parent_text})",
            ">>> x.shape",
            "()",
        ]
    )

    broadcast_argument_text = ", ".join(
        [f"[{arguments[0]}, {arguments[0]}]", *map(str, arguments[1:])]
    )
    broadcast_parent_text = ", ".join(
        [f"RV(Constant([{arguments[0]},{arguments[0]}]))"]
        + [f"RV(Constant({argument}))" for argument in arguments[1:]]
    )
    in_axes = [0] + [None] * (num_parents - 1)
    lines.extend(
        [
            f">>> y = pi.{op}({broadcast_argument_text})",
            ">>> y",
            f"RV(VMap({op!r}, {in_axes}, 2), {broadcast_parent_text})",
            ">>> y.shape",
            "(2,)",
        ]
    )

    if op._notes:
        lines.extend(["", "Notes", "-----"])
        lines.extend(op._notes)

    lines.extend(["", "See Also", "--------", f":class:`pangolin.ir.{op.name}`"])
    return "\n".join(lines)


def scalar[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Validate a scalar method's parameter names against the IR and attach its documentation."""
    function_name = getattr(function, "__name__")
    function_qualname = getattr(function, "__qualname__", function_name)
    parameters = tuple(inspect.signature(function).parameters.values())
    if not parameters or parameters[0].name != "self":
        raise TypeError(f"{function_qualname} must have self as its first parameter")

    parent_parameters = parameters[1:]
    if any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameter.default is not inspect.Parameter.empty
        for parameter in parent_parameters
    ):
        raise TypeError(
            f"{function_qualname} parents must be required positional-or-keyword parameters"
        )

    parent_names = tuple(parameter.name for parameter in parent_parameters)
    op_class = scalar_op_class(function_name)
    scalar_parent_descriptions(op_class, parent_names)
    function.__doc__ = scalar_doc(op_class, parent_names)
    return function


def default_function[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Wrap a default scalar method with module-level documentation and preserve its signature."""
    name = getattr(function, "__name__")

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        return function(*args, **kwargs)

    wrapped.__qualname__ = name
    wrapped.__doc__ = scalar_doc(
        scalar_op_class(name), tuple(inspect.signature(function).parameters), module_example=True
    )
    return wrapped


# --------------------------------------------------------------------------------------------------
# Index normalization
# --------------------------------------------------------------------------------------------------


def eliminate_ellipses(ndim: int, indices: tuple[Index, ...]) -> tuple[RVLike | slice, ...]:
    """Replace one ellipsis with enough full slices to index every dimension.

    Without an ellipsis, ``indices`` must contain exactly one index per dimension. With one
    ellipsis, the ellipsis is replaced by the number of full slices needed to produce that many
    indices.

    Parameters
    ----------
    ndim
        Number of dimensions in the value being indexed.
    indices
        Indices containing at most one ellipsis.

    Returns
    -------
    normalized_indices
        Exactly ``ndim`` indices with any ellipsis expanded into full slices.

    Raises
    ------
    ValueError
        If there is more than one ellipsis or the indices do not describe exactly ``ndim``
        dimensions.

    Examples
    --------
    >>> eliminate_ellipses(3, (0, 1, 2))
    (0, 1, 2)
    >>> eliminate_ellipses(4, (0, ..., 2))
    (0, slice(None, None, None), slice(None, None, None), 2)
    """
    ellipsis_count = builtins.sum(item is ... for item in indices)
    if ellipsis_count > 1:
        raise ValueError("an index can contain at most one ellipsis")

    slices_needed = ndim - len(indices) + ellipsis_count
    normalized_indices: list[RVLike | slice] = []
    for item in indices:
        if item is ...:
            normalized_indices.extend([slice(None)] * slices_needed)
        else:
            normalized_indices.append(item)

    if len(normalized_indices) != ndim:
        raise ValueError(
            f"number of indices ({len(normalized_indices)}) does not match ndim ({ndim})"
        )
    return tuple(normalized_indices)


def convert_index(interface: Interface, size: int, index: RVLike | slice) -> ir.RV:
    """Convert one array-like index or slice into an `ir.RV`.

    Existing `ir.RV` objects are preserved. Array-like indices become constant RVs, while a
    slice becomes a constant vector containing the selected positions in ``range(size)``.

    Parameters
    ----------
    interface
        Interface used to create new constant RVs.
    size
        Length of the dimension being indexed. This is used only to expand a slice.
    index
        Existing interface RV, array-like index, or slice.

    Returns
    -------
    index_rv
        Converted index RV.

    Examples
    --------
    >>> convert_index(default, 10, 5)
    RV(Constant(5))
    >>> convert_index(default, 5, slice(None, None, 2))
    RV(Constant([0,2,4]))
    >>> existing = constant([3, 1])
    >>> convert_index(default, 5, existing) is existing
    True
    """
    if isinstance(index, slice):
        return interface.constant(np.arange(size)[index])
    return interface.coerce(index)


def convert_indices(
    interface: Interface, shape: ir.Shape, *indices: RVLike | slice
) -> tuple[ir.RV, ...]:
    """Convert exactly one index for each dimension of a shape.

    Parameters
    ----------
    interface
        Interface used to create new constant RVs.
    shape
        Shape of the value being indexed.
    *indices
        Existing interface RV, array-like index, or slice for each dimension.

    Returns
    -------
    index_rvs
        Converted indices in their original order.

    Raises
    ------
    ValueError
        If the number of indices differs from the number of dimensions.

    Examples
    --------
    >>> convert_indices(default, (5, 3), slice(None), [2, 0])
    (RV(Constant([0,1,2,3,4])), RV(Constant([2,0])))
    """
    if len(indices) != len(shape):
        raise ValueError(f"number of indices ({len(indices)}) does not match ndim ({len(shape)})")
    return tuple(
        convert_index(interface, size, index) for size, index in zip(shape, indices, strict=True)
    )


# --------------------------------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------------------------------


def apply_op(interface: Interface, op: ir.Op, *parents: RVLike) -> ir.RV:
    """Coerce parents and construct an RV owned by the interface without broadcasting."""
    return ir.RV(op, *(interface.coerce(parent) for parent in parents), interface=interface)


class VectorIndexProxy:
    """Defer vectorized indexing until indices are supplied."""

    def __init__(self, interface: Interface, rv: ir.RV) -> None:
        self.interface = interface
        self.rv = rv

    def __getitem__(self, index: Index | tuple[Index, ...]) -> ir.RV:
        indices = index if isinstance(index, tuple) else (index,)
        return self.interface.vindex(self.rv, *indices)


class TransformLibrary:
    """Provide shared fixed transforms and factories that coerce through one interface.

    Only newly coerced factory parameters are tagged with this interface. Existing RVs retain
    their identity and interface, and the returned generic Transform keeps its usual output
    interface behavior.
    """

    def __init__(self, interface: Interface) -> None:
        self.interface: Final = interface

    exp = functional.tforms.exp
    log = functional.tforms.log
    logit = functional.tforms.logit
    inv_logit = functional.tforms.inv_logit
    stick_breaking = functional.tforms.stick_breaking
    softmax_centered = functional.tforms.softmax_centered
    softmax_centered_ilr = functional.tforms.softmax_centered_ilr
    cholesky = functional.tforms.cholesky
    fill_tril = functional.tforms.fill_tril
    extract_tril = functional.tforms.extract_tril
    exp_diagonal = functional.tforms.exp_diagonal
    log_diagonal = functional.tforms.log_diagonal
    unconstrain_spd = functional.tforms.unconstrain_spd

    def scale_by(self, factor: RVLike) -> Transform:
        """Return the scalar bijection ``y = factor * x``.

        ``factor`` is coerced to an RV when the Transform is created and becomes a captured bijector
        parameter.

        Coercion uses the interface owning this library; the returned Transform is a generic
        combinator, not bound to that interface.

        Parameters
        ----------
        factor
            Scalar scale factor. It is coerced into an `ir.RV` immediately.

        Returns
        -------
        transform
            Scalar `pangolin.functional.Transform` that multiplies by ``factor`` in its forward
            direction.

        Raises
        ------
        ValueError
            If ``factor`` is not scalar or the Transform is later applied to a non-scalar RV.

        Examples
        --------
        >>> from pangolin import interface as pi
        >>> y = pi.tforms.scale_by(2)(pi.normal)(0, 1)
        >>> y.shape
        ()
        """
        return functional.tforms.scale_by(self.interface.coerce(factor))

    def shift_by(self, offset: RVLike) -> Transform:
        """Return the bijection ``y = x + offset``.

        ``offset`` is coerced to an RV when the Transform is created and becomes a captured bijector
        parameter.

        Coercion uses the interface owning this library; the returned Transform is a generic
        combinator, not bound to that interface.

        The value and offset must have identical shapes; no broadcasting is performed.

        Parameters
        ----------
        offset
            Offset to add in the forward direction. It is coerced into an `ir.RV` immediately.

        Returns
        -------
        transform
            `pangolin.functional.Transform` that adds ``offset`` in its forward direction.

        Examples
        --------
        >>> from pangolin import interface as pi
        >>> y = pi.tforms.shift_by(3)(pi.normal)(0, 1)
        >>> y.shape
        ()
        """
        return functional.tforms.shift_by(self.interface.coerce(offset))

    def scaled_logit(self, lower: RVLike, upper: RVLike) -> Transform:
        """Return the scalar bijection from ``(lower, upper)`` to the real line.

        The forward mapping is ``logit((x - lower) / (upper - lower))``. Both bounds are coerced to RVs
        when the Transform is created and become captured bijector parameters.

        Coercion uses the interface owning this library; the returned Transform is a generic
        combinator, not bound to that interface.

        Parameters
        ----------
        lower
            Scalar lower endpoint of the open interval.
        upper
            Scalar upper endpoint of the open interval.

        Returns
        -------
        transform
            Scalar `pangolin.functional.Transform` mapping ``(lower, upper)`` to the real line.

        Raises
        ------
        ValueError
            If either endpoint is not scalar or the Transform is later applied to a non-scalar RV.

        Examples
        --------
        >>> from pangolin import interface as pi
        >>> unconstrained = pi.tforms.scaled_logit(1, 5)(pi.uniform)(1, 5)
        >>> unconstrained.shape
        ()
        """
        return functional.tforms.scaled_logit(
            self.interface.coerce(lower), self.interface.coerce(upper)
        )


class Interface(ir.RVInterface):
    """Own one configuration and the graph-building functions that use it.

    Parameters
    ----------
    broadcasting
        Broadcasting policy used outside temporary overrides: ``"off"``, ``"simple"``
        (the default), ``"rank"``, or ``"numpy"``.

    Notes
    -----
    Scalar functions follow the configured broadcasting policy. Non-scalar primitive operations
    use the IR's shape rules without implicit broadcasting. Both preserve existing parent RVs and attach
    this interface to newly created outputs.

    """

    # ----------------------------------------------------------------------------------------------
    # Shared combinators and graph inspection
    # ----------------------------------------------------------------------------------------------

    composite = staticmethod(composite)
    vjp = staticmethod(vjp)
    vmap = staticmethod(vmap)
    scan = staticmethod(scan)
    seq = staticmethod(seq)
    Transform = Transform
    compose_transforms = staticmethod(compose_transforms)
    upstream_nodes = staticmethod(upstream_nodes)
    print_upstream = staticmethod(print_upstream)
    ir_code = staticmethod(ir_code)
    signature = staticmethod(signature)
    graph_upstream = staticmethod(graph_upstream)

    def grad[**P](self, fun: Callable[P, ir.RV], argnum: int = 0) -> Callable[P, Any]:
        """A gradient combinator that also accepts a selected numeric scalar.

        If the selected argument is a Python or NumPy integer or floating-point scalar,
        it becomes a constant RV before differentiation. Booleans, complex numbers, arrays,
        and numeric leaves inside containers are not coerced. All other arguments are supplied
        unchanged. Otherwise, the contract is the same as `pangolin.functional.grad`.

        Parameters
        ----------
        fun
            Function returning one scalar RV, with no contributing newly created random nodes.
            The selected argument is supplied to it as an RV or an RV-only PyTree.
        argnum
            Nonnegative index of the supplied positional argument to differentiate.

        Returns
        -------
        grad_fun
            Function returning one gradient RV for a scalar or RV input, or the selected
            RV-only PyTree's structure with one gradient RV per leaf.

        Notes
        -----
        Newly coerced scalars use this interface; existing RVs retain their interfaces.
        Gradient outputs use the supplied function's output interface.

        Static typing preserves ``fun``'s parameter signature and uses ``Any`` for the
        gradient's return type. Scalar coercion does not widen the advertised input types.

        Examples
        --------
        >>> from pangolin import interface as pi
        >>> pi.grad(pi.exp)(0.0)
        RV(VJP(Exp(), 0), RV(Constant(1.)), RV(Constant(0.)))

        RV-only trees continue to produce gradients with the same structure:

        >>> result = pi.grad(lambda pair: pair[0] * pair[1])([pi.constant(2), pi.constant(3)])
        >>> [x.shape for x in result]
        [(), ()]
        """
        grad_fun: Callable[..., Any] = functional.grad(fun, argnum)

        @wraps(grad_fun)
        def call(*args: P.args, **kwargs: P.kwargs) -> Any:
            if argnum < len(args):
                argument = args[argnum]
                if isinstance(argument, (int, float, np.integer, np.floating)) and not isinstance(
                    argument, (bool, np.bool_)
                ):
                    return grad_fun(
                        *args[:argnum], self.constant(argument), *args[argnum + 1 :], **kwargs
                    )
            return grad_fun(*args, **kwargs)

        return call

    def cdf[**P](self, fun: Callable[P, ir.RV]) -> Callable[Concatenate[RVLike, P], ir.RV]:
        """A combinator building CDFs, accepting an RV-like evaluation value.

        Calling ``cdf(fun)(value, *args, **kwargs)`` coerces ``value`` to an RV and
        supplies all remaining arguments unchanged to ``fun``. No broadcasting is added.
        Otherwise, the contract is the same as `pangolin.functional.cdf`.

        :rtype: Callable[..., RV]

        Parameters
        ----------
        fun
            Function creating one random RV.

        Returns
        -------
        cdf_fun
            Function taking an RV-like evaluation value followed by ``fun``'s arguments,
            and returning a scalar CDF RV.

        Notes
        -----
        Newly coerced evaluation values use this interface; existing RVs are preserved.
        The result retains the interface of the random RV returned by ``fun``.

        Examples
        --------
        >>> from pangolin import interface as pi
        >>> normal_cdf = pi.cdf(pi.normal)
        >>> normal_cdf(0.0, 0.0, 1.0)
        RV(CDF(Normal()), RV(Constant(0.)), RV(Constant(0.)), RV(Constant(1.)))

        The supplied function can include ordinary parameter calculations:

        >>> def normal_with_log_scale(mean, log_scale):
        ...     return pi.normal(mean, pi.exp(log_scale))
        >>> model_cdf = pi.cdf(normal_with_log_scale)
        >>> model_cdf(0.0, 0.0, 0.0)
        RV(CDF(Normal()), RV(Constant(0.)), RV(Constant(0.)), RV(Exp(), RV(Constant(0.))))
        """
        cdf_fun = functional.cdf(fun)

        @wraps(cdf_fun)
        def call(value: RVLike, /, *args: P.args, **kwargs: P.kwargs) -> ir.RV:
            return cdf_fun(self.coerce(value), *args, **kwargs)

        return call

    def quantile[**P](self, fun: Callable[P, ir.RV]) -> Callable[Concatenate[RVLike, P], ir.RV]:
        """A combinator building quantiles, accepting an RV-like probability.

        Calling ``quantile(fun)(probability, *args, **kwargs)`` coerces ``probability`` to
        a scalar RV and supplies all remaining arguments unchanged to ``fun``. No broadcasting
        is added. Otherwise, the contract is the same as `pangolin.functional.quantile`.

        :rtype: Callable[..., RV]

        Parameters
        ----------
        fun
            Function creating one scalar random RV.

        Returns
        -------
        quantile_fun
            Function taking a scalar RV-like probability followed by ``fun``'s arguments,
            and returning a scalar quantile RV. Numerical support depends on the backend;
            see `pangolin.jax_backend.lower_quantile` for JAX's supported distributions.

        Notes
        -----
        Newly coerced probabilities use this interface; existing RVs are preserved.
        The result retains the interface of the random RV returned by ``fun``.

        Examples
        --------
        >>> from pangolin import interface as pi
        >>> pi.quantile(pi.normal)(0.5, 0.0, 1.0)
        RV(Quantile(Normal()), RV(Constant(0.5)), RV(Constant(0.)), RV(Constant(1.)))

        The supplied function can include ordinary parameter calculations:

        >>> def normal_with_log_scale(mean, log_scale):
        ...     return pi.normal(mean, pi.exp(log_scale))
        >>> pi.quantile(normal_with_log_scale)(0.5, 0.0, 0.0)
        RV(Quantile(Normal()), RV(Constant(0.5)), RV(Constant(0.)), RV(Exp(), RV(Constant(0.))))

        Use explicit vectorization for several probabilities; the distribution inside
        ``quantile`` must remain scalar:

        >>> probabilities = pi.constant([0.1, 0.5, 0.9])
        >>> y = pi.vmap(pi.quantile(pi.normal), (0, None, None))(probabilities, 0.0, 1.0)
        >>> y.op
        VMap(Quantile(Normal()), [0, None, None], 3)
        >>> y.shape
        (3,)
        """
        quantile_fun = functional.quantile(fun)

        @wraps(quantile_fun)
        def call(probability: RVLike, /, *args: P.args, **kwargs: P.kwargs) -> ir.RV:
            return quantile_fun(self.coerce(probability), *args, **kwargs)

        return call

    def log_prob[**P](self, fun: Callable[P, ir.RV]) -> Callable[Concatenate[RVLike, P], ir.RV]:
        """A combinator building log probabilities, accepting an RV-like evaluation value.

        Calling ``log_prob(fun)(value, *args, **kwargs)`` coerces ``value`` to an RV and
        supplies all remaining arguments unchanged to ``fun``. No broadcasting is added.
        Otherwise, the contract is the same as `pangolin.functional.log_prob`.

        :rtype: Callable[..., RV]

        Parameters
        ----------
        fun
            Function creating one random RV.

        Returns
        -------
        log_prob_fun
            Function taking an RV-like evaluation value followed by ``fun``'s arguments,
            and returning a scalar log-probability RV.

        Notes
        -----
        Newly coerced evaluation values use this interface; existing RVs are preserved.
        The result retains the interface of the random RV returned by ``fun``.

        Examples
        --------
        >>> from pangolin import interface as pi
        >>> normal_log_prob = pi.log_prob(pi.normal)
        >>> normal_log_prob(0.0, 0.0, 1.0)
        RV(LogProb(Normal()), RV(Constant(0.)), RV(Constant(0.)), RV(Constant(1.)))

        The supplied function can include ordinary parameter calculations:

        >>> def normal_with_log_scale(mean, log_scale):
        ...     return pi.normal(mean, pi.exp(log_scale))
        >>> model_log_prob = pi.log_prob(normal_with_log_scale)
        >>> model_log_prob(0.0, 0.0, 0.0)
        RV(LogProb(Normal()), RV(Constant(0.)), RV(Constant(0.)), RV(Exp(), RV(Constant(0.))))
        """
        log_prob_fun = functional.log_prob(fun)

        @wraps(log_prob_fun)
        def call(value: RVLike, /, *args: P.args, **kwargs: P.kwargs) -> ir.RV:
            return log_prob_fun(self.coerce(value), *args, **kwargs)

        return call

    def truncated[**P](
        self, fun: Callable[P, ir.RV], lo: RVLike | None = None, hi: RVLike | None = None
    ) -> Callable[P, ir.RV]:
        """A combinator restricting continuous scalar distributions to a shared interval.

        Calling ``truncated(fun, lo, hi)(*args, **kwargs)`` constructs an RV with the
        distribution defined by ``fun(*args, **kwargs)``, restricted to
        the supplied bounds and renormalized. Use ``None`` to leave either end unrestricted.
        Supplied bounds must be finite and strictly inside the base distribution's support,
        with ``lo < hi`` when both are supplied. Backends enforce these value-dependent
        requirements during evaluation, not when the graph is constructed.

        The function may produce a scalar distribution or independent scalar draws expressed
        through nested VMap and Composite Ops. Every draw uses the same scalar bounds.
        The result has the same shape as the original output. The full contract is given in
        `pangolin.functional.truncated`.
        With both bounds omitted, the original output RV is returned, subject to the same requirements.

        Numerical support depends on the backend. See `pangolin.jax_backend.lower_truncated`
        for JAX's sampling requirements and numerical limitations.

        :rtype: Callable[..., RV]

        Parameters
        ----------
        fun
            Function creating one continuous random RV, with no contributing newly created
            random intermediate RVs.
        lo
            Scalar lower bound, or ``None`` (the default) to omit it.
        hi
            Scalar upper bound, or ``None`` (the default) to omit it.

        Returns
        -------
        truncated_fun
            Function accepting ``fun``'s arguments unchanged and returning a truncated RV
            with the same shape as its original output.

        Raises
        ------
        ValueError
            If a bound is not scalar, or the supplied function's output violates the
            requirements of `pangolin.functional.truncated`.

        Notes
        -----
        Arguments are supplied to ``fun`` unchanged; any coercion or broadcasting performed
        by ``fun`` is preserved. Bounds are scalar and are never broadcast.

        Supplied bounds are coerced when the combinator is constructed. Existing RV bounds are
        preserved; newly coerced bounds use this interface. The result retains the interface
        chosen by ``fun``, not necessarily this interface. Numerical support and validation of
        bound values are left to backends.

        Examples
        --------
        Create a lower-truncated Normal with no upper bound:

        >>> from pangolin import interface as pi
        >>> positive_normal = pi.truncated(pi.normal, 0.0)
        >>> x = positive_normal(1.1, 2.2)
        >>> x.op
        Truncated(Normal(), upper=False)

        Normal's usual broadcasting is preserved, with scalar bounds shared across draws:

        >>> y = positive_normal([1.1, 1.2, 1.3], 2.2)
        >>> y.op
        VMap(Truncated(Normal(), upper=False), [None, 0, None], 3)
        >>> y.shape
        (3,)

        Supply only an upper bound, or both bounds:

        >>> x = pi.truncated(pi.normal, hi=0.0)(1.1, 2.2)
        >>> x.op
        Truncated(Normal(), lower=False)
        >>> x = pi.truncated(pi.normal, 0.0, 1.0)(1.1, 2.2)
        >>> x.op
        Truncated(Normal())

        With neither bound, no truncation is added:

        >>> x = pi.truncated(pi.normal)(1.1, 2.2)
        >>> x.op
        Normal()

        Even length-one arrays are rejected as bounds:

        >>> pi.truncated(pi.normal, [0.0])
        Traceback (most recent call last):
            ...
        ValueError: truncation bounds must be scalar
        """
        return functional.truncated(
            fun,
            None if lo is None else self.coerce(lo),
            None if hi is None else self.coerce(hi),
        )

    @typing.overload
    def broadcast(self, fun: Callable[[], ir.RV]) -> Callable[[], ir.RV]: ...

    @typing.overload
    def broadcast(self, fun: Callable[[ir.RV], ir.RV]) -> Callable[[RVLike], ir.RV]: ...

    @typing.overload
    def broadcast(
        self, fun: Callable[[ir.RV, ir.RV], ir.RV]
    ) -> Callable[[RVLike, RVLike], ir.RV]: ...

    @typing.overload
    def broadcast(
        self, fun: Callable[[ir.RV, ir.RV, ir.RV], ir.RV]
    ) -> Callable[[RVLike, RVLike, RVLike], ir.RV]: ...

    @typing.overload
    def broadcast(
        self, fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]
    ) -> Callable[[RVLike, RVLike, RVLike, RVLike], ir.RV]: ...

    @typing.overload
    def broadcast(
        self, fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]
    ) -> Callable[[RVLike, RVLike, RVLike, RVLike, RVLike], ir.RV]: ...

    @typing.overload
    def broadcast(
        self, fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV]
    ) -> Callable[[RVLike, RVLike, RVLike, RVLike, RVLike, RVLike], ir.RV]: ...

    @typing.overload
    def broadcast(self, fun: Callable[..., ir.RV]) -> Callable[..., ir.RV]: ...

    def broadcast(self, fun: Callable[..., ir.RV]) -> Callable[..., ir.RV]:
        """A combinator that broadcasts a scalar graph-building function.

        The returned function coerces supplied arguments to RVs and applies ``fun`` to scalar
        arguments according to this interface's broadcasting policy. The policy is looked up
        when the returned function is called, so temporary `overload` settings apply.

        Parameters
        ----------
        fun
            Function accepting scalar RV arguments and returning one scalar RV. Its internal
            calculations may use vectors and matrices; only its inputs and output must be scalar.

        Returns
        -------
        broadcast_fun
            Function accepting RVLike arguments and returning an RV with their broadcast shape.
            Supplied positional and keyword arguments follow the same broadcasting rules.

        Raises
        ------
        TypeError
            If ``fun`` does not return an RV.
        ValueError
            If input shapes cannot be broadcast under the current policy or ``fun`` returns a
            nonscalar RV.

        Notes
        -----
        Existing RV inputs retain their interfaces; coerced constants use this interface. The
        result retains the interface chosen by ``fun``.

        Only explicitly supplied arguments are coerced and broadcast. Omitted defaults and
        captured values are used normally by ``fun`` and do not determine the broadcast shape.
        Static overloads describe positional calls with up to six arguments. Other signatures
        retain the RV return type but leave arguments unchecked. Runtime coercion applies to
        every supplied argument, including keyword arguments.

        Examples
        --------
        Broadcast a function that accepts only scalar RVs:

        >>> from pangolin import interface as pi, ir
        >>> def squared_difference(x, y):
        ...     assert isinstance(x, ir.RV) and x.shape == ()
        ...     assert isinstance(y, ir.RV) and y.shape == ()
        ...     return (x - y) * (x - y)
        >>> broadcast_squared_difference = pi.broadcast(squared_difference)
        >>> broadcast_squared_difference([1.0, 2.0, 3.0], 1.0).shape
        (3,)
        """

        def scalar_fun(*args: ir.RV, **kwargs: ir.RV) -> ir.RV:
            output = fun(*args, **kwargs)
            if not isinstance(output, ir.RV):
                raise TypeError("broadcast function must return an RV")
            if output.shape != ():
                raise ValueError("broadcast function must return a scalar RV")
            return output

        @wraps(fun)
        def broadcast_fun(*args: RVLike, **kwargs: RVLike) -> ir.RV:
            args_rv = tuple(self.coerce(arg) for arg in args)
            kwargs_rv = {name: self.coerce(arg) for name, arg in kwargs.items()}
            broadcast = broadcasting_combinators[self._effective_config.broadcasting]
            return broadcast(scalar_fun)(*args_rv, **kwargs_rv)

        return broadcast_fun

    # ----------------------------------------------------------------------------------------------
    # Configuration and syntax
    # ----------------------------------------------------------------------------------------------

    def __init__(
        self, *, broadcasting: Literal["off", "simple", "rank", "numpy"] = "simple"
    ) -> None:
        self._config: Final = Config(broadcasting=broadcasting)
        self.tforms: Final = TransformLibrary(self)
        self._config_override: Final[ContextVar[Config | None]] = ContextVar(
            "pangolin.interface.config", default=None
        )

    def get_attribute(self, rv: ir.RV, name: str, /) -> Any:
        """Provide vectorized indexing and summation as additional RV conveniences."""
        if name == "vindex":
            return VectorIndexProxy(self, rv)
        if name == "sum":
            return partial(self.sum, rv)
        return super().get_attribute(rv, name)

    @property
    def _effective_config(self) -> Config:
        """Return the effective, read-only configuration in the current context."""
        override = self._config_override.get()
        return self._config if override is None else override

    @contextmanager
    def overload(
        self, *, broadcasting: Literal["off", "simple", "rank", "numpy"] | None = None
    ) -> Iterator[None]:
        """Temporarily change this interface's configuration in the current context.

        Unspecified options retain their current values. Overrides may be nested, and the previous
        configuration is restored on exit, including when an exception occurs. The base configuration
        and other interfaces are unchanged.

        Functions read the effective configuration when called. Saved function references and
        indexing proxies therefore observe the active override. RVs created inside the context keep
        this interface, so their syntax uses its restored configuration after the context ends.

        Parameters
        ----------
        broadcasting
            Temporary scalar and vectorized-index broadcasting policy. Omit it or pass ``None``
            to leave the current policy unchanged.

        Raises
        ------
        TypeError
            If an unknown option is supplied or broadcasting is neither a string nor ``None``.
        ValueError
            If broadcasting is not a supported mode.

        Notes
        -----
        Overrides are context-local, not global mutations. Async tasks inherit the active context
        when created; subsequent overrides and restoration in one task do not change another task's
        configuration. The same context-local mechanism isolates overrides in separate threads.

        Examples
        --------
        >>> from pangolin.interface import Interface
        >>> pi = Interface(broadcasting="off")
        >>> normal = pi.normal
        >>> with pi.overload(broadcasting="simple"):
        ...     y = normal([0, 1], 1)
        ...     print(y.shape)
        (2,)
        >>> normal([0, 1], 1)
        Traceback (most recent call last):
            ...
        ValueError: Can't broadcast shapes (2,) and () with broadcasting=OFF
        >>> y.interface is pi
        True
        """
        config = (
            self._effective_config
            if broadcasting is None
            else replace(self._effective_config, broadcasting=broadcasting)
        )
        token = self._config_override.set(config)
        try:
            yield
        finally:
            self._config_override.reset(token)

    def neg(self, x: RVLike, /) -> ir.RV:
        """Negate a scalar value."""
        return self.mul(x, -1)

    @scalar
    def add(self, a: RVLike, b: RVLike) -> ir.RV:
        def scalar_add(a: ir.RV, b: ir.RV) -> ir.RV:
            return ir.RV(ir.Add(), a, b, interface=self)

        return self.broadcast(scalar_add)(a, b)

    @scalar
    def sub(self, a: RVLike, b: RVLike) -> ir.RV:
        def scalar_sub(a: ir.RV, b: ir.RV) -> ir.RV:
            return ir.RV(ir.Sub(), a, b, interface=self)

        return self.broadcast(scalar_sub)(a, b)

    @scalar
    def mul(self, a: RVLike, b: RVLike) -> ir.RV:
        def scalar_mul(a: ir.RV, b: ir.RV) -> ir.RV:
            return ir.RV(ir.Mul(), a, b, interface=self)

        return self.broadcast(scalar_mul)(a, b)

    @scalar
    def div(self, a: RVLike, b: RVLike) -> ir.RV:
        def scalar_div(a: ir.RV, b: ir.RV) -> ir.RV:
            return ir.RV(ir.Div(), a, b, interface=self)

        return self.broadcast(scalar_div)(a, b)

    @scalar
    def pow(self, a: RVLike, b: RVLike) -> ir.RV:
        def scalar_pow(a: ir.RV, b: ir.RV) -> ir.RV:
            return ir.RV(ir.Pow(), a, b, interface=self)

        return self.broadcast(scalar_pow)(a, b)

    @scalar
    def arccos(self, a: RVLike) -> ir.RV:
        def scalar_arccos(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Arccos(), a, interface=self)

        return self.broadcast(scalar_arccos)(a)

    @scalar
    def arccosh(self, a: RVLike) -> ir.RV:
        def scalar_arccosh(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Arccosh(), a, interface=self)

        return self.broadcast(scalar_arccosh)(a)

    @scalar
    def arcsin(self, a: RVLike) -> ir.RV:
        def scalar_arcsin(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Arcsin(), a, interface=self)

        return self.broadcast(scalar_arcsin)(a)

    @scalar
    def arcsinh(self, a: RVLike) -> ir.RV:
        def scalar_arcsinh(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Arcsinh(), a, interface=self)

        return self.broadcast(scalar_arcsinh)(a)

    @scalar
    def arctan(self, a: RVLike) -> ir.RV:
        def scalar_arctan(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Arctan(), a, interface=self)

        return self.broadcast(scalar_arctan)(a)

    @scalar
    def arctanh(self, a: RVLike) -> ir.RV:
        def scalar_arctanh(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Arctanh(), a, interface=self)

        return self.broadcast(scalar_arctanh)(a)

    @scalar
    def cos(self, a: RVLike) -> ir.RV:
        def scalar_cos(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Cos(), a, interface=self)

        return self.broadcast(scalar_cos)(a)

    @scalar
    def cosh(self, a: RVLike) -> ir.RV:
        def scalar_cosh(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Cosh(), a, interface=self)

        return self.broadcast(scalar_cosh)(a)

    @scalar
    def sin(self, a: RVLike) -> ir.RV:
        def scalar_sin(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Sin(), a, interface=self)

        return self.broadcast(scalar_sin)(a)

    @scalar
    def sinh(self, a: RVLike) -> ir.RV:
        def scalar_sinh(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Sinh(), a, interface=self)

        return self.broadcast(scalar_sinh)(a)

    @scalar
    def tan(self, a: RVLike) -> ir.RV:
        def scalar_tan(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Tan(), a, interface=self)

        return self.broadcast(scalar_tan)(a)

    @scalar
    def tanh(self, a: RVLike) -> ir.RV:
        def scalar_tanh(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Tanh(), a, interface=self)

        return self.broadcast(scalar_tanh)(a)

    @scalar
    def abs(self, a: RVLike) -> ir.RV:
        def scalar_abs(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Abs(), a, interface=self)

        return self.broadcast(scalar_abs)(a)

    @scalar
    def minimum(self, a: RVLike, b: RVLike) -> ir.RV:
        def scalar_minimum(a: ir.RV, b: ir.RV) -> ir.RV:
            return ir.RV(ir.Minimum(), a, b, interface=self)

        return self.broadcast(scalar_minimum)(a, b)

    @scalar
    def maximum(self, a: RVLike, b: RVLike) -> ir.RV:
        def scalar_maximum(a: ir.RV, b: ir.RV) -> ir.RV:
            return ir.RV(ir.Maximum(), a, b, interface=self)

        return self.broadcast(scalar_maximum)(a, b)

    @scalar
    def exp(self, a: RVLike) -> ir.RV:
        def scalar_exp(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Exp(), a, interface=self)

        return self.broadcast(scalar_exp)(a)

    @scalar
    def inv_logit(self, a: RVLike) -> ir.RV:
        def scalar_inv_logit(a: ir.RV) -> ir.RV:
            return ir.RV(ir.InvLogit(), a, interface=self)

        return self.broadcast(scalar_inv_logit)(a)

    @scalar
    def log(self, a: RVLike) -> ir.RV:
        def scalar_log(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Log(), a, interface=self)

        return self.broadcast(scalar_log)(a)

    @scalar
    def logit(self, a: RVLike) -> ir.RV:
        def scalar_logit(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Logit(), a, interface=self)

        return self.broadcast(scalar_logit)(a)

    @scalar
    def step(self, a: RVLike) -> ir.RV:
        def scalar_step(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Step(), a, interface=self)

        return self.broadcast(scalar_step)(a)

    @scalar
    def loggamma(self, a: RVLike) -> ir.RV:
        def scalar_loggamma(a: ir.RV) -> ir.RV:
            return ir.RV(ir.Loggamma(), a, interface=self)

        return self.broadcast(scalar_loggamma)(a)

    @scalar
    def normal(self, mu: RVLike, sigma: RVLike) -> ir.RV:
        def scalar_normal(mu: ir.RV, sigma: ir.RV) -> ir.RV:
            return ir.RV(ir.Normal(), mu, sigma, interface=self)

        return self.broadcast(scalar_normal)(mu, sigma)

    @scalar
    def normal_prec(self, mu: RVLike, tau: RVLike) -> ir.RV:
        def scalar_normal_prec(mu: ir.RV, tau: ir.RV) -> ir.RV:
            return ir.RV(ir.NormalPrec(), mu, tau, interface=self)

        return self.broadcast(scalar_normal_prec)(mu, tau)

    @scalar
    def lognormal(self, mu: RVLike, sigma: RVLike) -> ir.RV:
        def scalar_lognormal(mu: ir.RV, sigma: ir.RV) -> ir.RV:
            return ir.RV(ir.Lognormal(), mu, sigma, interface=self)

        return self.broadcast(scalar_lognormal)(mu, sigma)

    @scalar
    def bernoulli(self, theta: RVLike) -> ir.RV:
        def scalar_bernoulli(theta: ir.RV) -> ir.RV:
            return ir.RV(ir.Bernoulli(), theta, interface=self)

        return self.broadcast(scalar_bernoulli)(theta)

    @scalar
    def bernoulli_logit(self, theta: RVLike) -> ir.RV:
        def scalar_bernoulli_logit(theta: ir.RV) -> ir.RV:
            return ir.RV(ir.BernoulliLogit(), theta, interface=self)

        return self.broadcast(scalar_bernoulli_logit)(theta)

    @scalar
    def binomial(self, N: RVLike, theta: RVLike) -> ir.RV:
        def scalar_binomial(N: ir.RV, theta: ir.RV) -> ir.RV:
            return ir.RV(ir.Binomial(), N, theta, interface=self)

        return self.broadcast(scalar_binomial)(N, theta)

    @scalar
    def cauchy(self, mu: RVLike, sigma: RVLike) -> ir.RV:
        def scalar_cauchy(mu: ir.RV, sigma: ir.RV) -> ir.RV:
            return ir.RV(ir.Cauchy(), mu, sigma, interface=self)

        return self.broadcast(scalar_cauchy)(mu, sigma)

    @scalar
    def uniform(self, alpha: RVLike, beta: RVLike) -> ir.RV:
        def scalar_uniform(alpha: ir.RV, beta: ir.RV) -> ir.RV:
            return ir.RV(ir.Uniform(), alpha, beta, interface=self)

        return self.broadcast(scalar_uniform)(alpha, beta)

    @scalar
    def beta(self, alpha: RVLike, beta: RVLike) -> ir.RV:
        def scalar_beta(alpha: ir.RV, beta: ir.RV) -> ir.RV:
            return ir.RV(ir.Beta(), alpha, beta, interface=self)

        return self.broadcast(scalar_beta)(alpha, beta)

    @scalar
    def exponential(self, beta: RVLike) -> ir.RV:
        def scalar_exponential(beta: ir.RV) -> ir.RV:
            return ir.RV(ir.Exponential(), beta, interface=self)

        return self.broadcast(scalar_exponential)(beta)

    @scalar
    def gamma(self, alpha: RVLike, beta: RVLike) -> ir.RV:
        def scalar_gamma(alpha: ir.RV, beta: ir.RV) -> ir.RV:
            return ir.RV(ir.Gamma(), alpha, beta, interface=self)

        return self.broadcast(scalar_gamma)(alpha, beta)

    @scalar
    def poisson(self, lambd: RVLike) -> ir.RV:
        def scalar_poisson(lambd: ir.RV) -> ir.RV:
            return ir.RV(ir.Poisson(), lambd, interface=self)

        return self.broadcast(scalar_poisson)(lambd)

    @scalar
    def beta_binomial(self, N: RVLike, alpha: RVLike, beta: RVLike) -> ir.RV:
        def scalar_beta_binomial(N: ir.RV, alpha: ir.RV, beta: ir.RV) -> ir.RV:
            return ir.RV(ir.BetaBinomial(), N, alpha, beta, interface=self)

        return self.broadcast(scalar_beta_binomial)(N, alpha, beta)

    @scalar
    def student_t(self, nu: RVLike, mu: RVLike, sigma: RVLike) -> ir.RV:
        def scalar_student_t(nu: ir.RV, mu: ir.RV, sigma: ir.RV) -> ir.RV:
            return ir.RV(ir.StudentT(), nu, mu, sigma, interface=self)

        return self.broadcast(scalar_student_t)(nu, mu, sigma)

    # ----------------------------------------------------------------------------------------------
    # Scalar conveniences
    # ----------------------------------------------------------------------------------------------

    def expit(self, a: RVLike) -> ir.RV:
        "An alias for :func:`pangolin.interface.inv_logit`."
        return self.inv_logit(a)

    def sigmoid(self, a: RVLike) -> ir.RV:
        "An alias for :func:`pangolin.interface.inv_logit`."
        return self.inv_logit(a)

    def sqrt(self, x: RVLike) -> ir.RV:
        """
        Take a square root.

        sqrt(x) is an alias for pow(x, 0.5)
        """
        return self.pow(x, 0.5)

    # ----------------------------------------------------------------------------------------------
    # Non-scalar operations
    # ----------------------------------------------------------------------------------------------

    def matmul(self, a: RVLike, b: RVLike) -> ir.RV:
        """Multiply two vectors or matrices.

        The behavior follows
        `NumPy's matmul <https://numpy.org/doc/stable/reference/generated/numpy.matmul.html>`_ except
        that both inputs must be one- or two-dimensional:

        * Two vectors produce their scalar inner product.
        * A vector and a matrix produce a vector.
        * A matrix and a vector produce a vector.
        * Two matrices produce a matrix.

        The contracted dimensions must match. Higher-dimensional batched matrix multiplication is not
        currently supported.

        Parameters
        ----------
        a
            Left vector or matrix.
        b
            Right vector or matrix.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.matmul([1, 2, 3], [4, 5, 6]).shape
        ()
        >>> pi.matmul([[1, 2, 3], [4, 5, 6]], [7, 8, 9]).shape
        (2,)
        """
        return apply_op(self, ir.Matmul(), a, b)

    def transpose(self, a: RVLike) -> ir.RV:
        """Transpose a matrix.

        The input must be two-dimensional. An input with shape ``(m, n)`` produces an output with
        shape ``(n, m)``.

        Parameters
        ----------
        a
            Input matrix.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.transpose([[1, 2, 3], [4, 5, 6]]).shape
        (3, 2)
        """
        return apply_op(self, ir.Transpose(), a)

    def diag(self, a: RVLike) -> ir.RV:
        """Extract the diagonal of a square matrix as a vector.

        Parameters
        ----------
        a
            Input square matrix.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.diag([[1, 2], [3, 4]]).shape
        (2,)

        See Also
        --------
        diag_matrix : Construct a square diagonal matrix from a vector.
        """
        return apply_op(self, ir.Diag(), a)

    def diag_matrix(self, a: RVLike) -> ir.RV:
        """Construct a square diagonal matrix from a vector.

        Parameters
        ----------
        a
            Vector of diagonal entries.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.diag_matrix([1, 2, 3]).shape
        (3, 3)

        See Also
        --------
        diag : Extract the diagonal of a square matrix as a vector.
        """
        return apply_op(self, ir.DiagMatrix(), a)

    def inv(self, a: RVLike) -> ir.RV:
        """Invert a square matrix.

        The IR verifies that the input is a square matrix. Invertibility depends on the matrix's value
        and therefore remains a backend-level value constraint.

        Parameters
        ----------
        a
            Input square matrix.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.inv([[1, 0], [0, 2]]).shape
        (2, 2)
        """
        return apply_op(self, ir.Inv(), a)

    def cholesky(self, a: RVLike) -> ir.RV:
        """Compute the lower-triangular Cholesky factor of a square matrix.

        The IR verifies that the input is a square matrix. Positive-definiteness depends on the
        matrix's value and therefore remains a backend-level value constraint.

        Parameters
        ----------
        a
            Input positive-definite matrix.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.cholesky([[2, 0], [0, 3]]).shape
        (2, 2)
        """
        return apply_op(self, ir.Cholesky(), a)

    def softmax(self, a: RVLike) -> ir.RV:
        """Apply the `softmax function <https://en.wikipedia.org/wiki/Softmax_function>`_ to a vector.

        The input must be one-dimensional, and the result has the same shape. The current operation
        does not provide a configurable axis.

        Parameters
        ----------
        a
            Input vector.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.softmax([1, 2, 3]).shape
        (3,)
        """
        return apply_op(self, ir.Softmax(), a)

    def sum(self, x: RVLike, axis: int) -> ir.RV:
        """Sum an array along one axis, removing that axis from the result.

        Negative axes count from the end. The input must be nonscalar, and ``axis`` must refer to an
        existing axis.

        Parameters
        ----------
        x
            Array to sum. Non-RV values are converted to constant RVs.
        axis
            Axis to remove.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.sum([[1, 2, 3], [4, 5, 6]], axis=0).shape
        (3,)
        >>> pi.sum([[1, 2, 3], [4, 5, 6]], axis=-1).shape
        (2,)
        """
        return apply_op(self, ir.Sum(axis), x)

    def concatenate(self, *arrays: RVLike, axis: int = 0) -> ir.RV:
        """Concatenate arrays along an existing axis without broadcasting.

        Parameters
        ----------
        *arrays
            One or more arrays with the same rank and equal dimensions except along ``axis``.
            Non-RV values are converted to constant RVs.
        axis
            Existing axis along which to concatenate. Negative axes count from the end.

        Returns
        -------
        ir.RV
            Concatenated RV owned by this interface.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> A = pi.constant([[1, 2], [3, 4]])
        >>> B = pi.constant([[5, 6], [7, 8]])
        >>> A.shape
        (2, 2)
        >>> B.shape
        (2, 2)
        >>> pi.concatenate(A, B).shape
        (4, 2)
        >>> pi.concatenate(A, B, axis=1).shape
        (2, 4)

        Random and deterministic RVs can be concatenated in the same way.

        >>> x = pi.multi_normal([0, 0], [[1, 0], [0, 1]])
        >>> pi.concatenate(x, pi.softmax(x)).shape
        (4,)
        """
        return apply_op(self, ir.Concatenate(axis), *arrays)

    def fill_tril(self, params: RVLike) -> ir.RV:
        """Fill a lower-triangular matrix row by row from a vector.

        This function creates no specialized Op; it lowers the operation to indexing and
        multiplication in the ordinary Pangolin IR.

        Parameters
        ----------
        params
            One-dimensional RV or array-like value of length ``N * (N + 1) / 2``.

        Returns
        -------
        matrix
            Lower-triangular RV with shape ``(N, N)``.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> params = pi.constant([1.1,2.2,3.3])
        >>> out = pi.fill_tril(params)
        >>> out.shape
        (2, 2)
        >>> pi.print_upstream(out)
        shape  | statement
        ------ | ---------
        (3,)   | a = [1.1 2.2 3.3]
        (2, 2) | b = [[0 0] [1 2]]
        (2, 2) | c = vmap(vmap(index, [None, 0], 2), [None, 0], 2)(a, b)
        (2, 2) | d = [[1. 0.] [1. 1.]]
        (2, 2) | e = vmap(vmap(mul, [0, 0], 2), [0, 0], 2)(c, d)
        """
        return functional.bind_interface(self, functional.tforms.fill_tril.forward)(
            self.coerce(params)
        )

    def extract_tril(self, L: RVLike) -> ir.RV:
        """Extract the lower-triangular entries of a square matrix row by row.

        This function creates no specialized Op; it lowers the operation to vectorized indexing in the
        ordinary Pangolin IR.

        Parameters
        ----------
        L
            Square two-dimensional RV or array-like value with shape ``(N, N)``.

        Returns
        -------
        entries
            One-dimensional RV of length ``N * (N + 1) / 2``.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> L = pi.constant([[1.1,0],[2.2,3.3]])
        >>> l = pi.extract_tril(L)
        >>> l.shape
        (3,)
        >>> pi.print_upstream(l)
        shape  | statement
        ------ | ---------
        (2, 2) | a = [[1.1 0. ] [2.2 3.3]]
        (3,)   | b = [0 1 1]
        (3,)   | c = [0 0 1]
        (3,)   | d = vmap(index, [None, 0, 0], 3)(a, b, c)
        """
        return functional.bind_interface(self, functional.tforms.extract_tril.forward)(
            self.coerce(L)
        )

    # ----------------------------------------------------------------------------------------------
    # Multivariate distributions
    # ----------------------------------------------------------------------------------------------

    def multi_normal(self, mean: RVLike, cov: RVLike) -> ir.RV:
        """Create a multivariate normal RV.

        The mean must have shape ``(n,)`` and the covariance must have shape ``(n, n)``. The result has
        shape ``(n,)``. Positive-definiteness of the covariance is value-dependent and therefore
        remains a backend-level constraint.

        Parameters
        ----------
        mean
            Mean vector.
        cov
            Covariance matrix.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.multi_normal([0, 0], [[1, 0], [0, 1]]).shape
        (2,)
        """
        return apply_op(self, ir.MultiNormal(), mean, cov)

    def categorical(self, theta: RVLike) -> ir.RV:
        """Create a scalar categorical RV (`Wikipedia <https://en.wikipedia.org/wiki/Categorical_distribution>`__).

        ``theta`` must be a one-dimensional vector of nonnegative probabilities that sum to one. The
        IR validates its shape; the probability constraints remain backend-level. The result is a
        scalar category index.

        Parameters
        ----------
        theta
            Category probabilities.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.categorical([0.2, 0.3, 0.5]).shape
        ()
        """
        return apply_op(self, ir.Categorical(), theta)

    def multinomial(self, n: RVLike, p: RVLike) -> ir.RV:
        """Create a multinomial RV (`Wikipedia <https://en.wikipedia.org/wiki/Multinomial_distribution>`__).

        ``n`` must be a scalar number of trials and ``p`` must be a one-dimensional probability
        vector. The result is a count vector with the same shape as ``p``. The IR validates shapes;
        constraints on the values remain backend-level.

        Parameters
        ----------
        n
            Number of trials.
        p
            Category probabilities.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.multinomial(10, [0.2, 0.3, 0.5]).shape
        (3,)
        """
        return apply_op(self, ir.Multinomial(), n, p)

    def dirichlet(self, alpha: RVLike) -> ir.RV:
        """Create a simplex-valued Dirichlet RV (`Wikipedia <https://en.wikipedia.org/wiki/Dirichlet_distribution>`__).

        ``alpha`` must be a one-dimensional vector of positive concentrations. The result has the same
        shape and lies on the corresponding simplex. The IR validates the shape; positivity remains a
        backend-level constraint.

        Parameters
        ----------
        alpha
            Concentration vector.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.dirichlet([1, 2, 3]).shape
        (3,)
        """
        return apply_op(self, ir.Dirichlet(), alpha)

    def wishart(self, nu: RVLike, scale: RVLike) -> ir.RV:
        """Create a matrix-valued Wishart RV (`Wikipedia <https://en.wikipedia.org/wiki/Wishart_distribution>`__).

        ``nu`` must be scalar and ``scale`` must be a square positive-definite matrix. The result has
        the same shape as ``scale``. The IR validates the shapes; constraints on their values remain
        backend-level.

        Parameters
        ----------
        nu
            Degrees of freedom.
        scale
            Scale matrix.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> pi.wishart(5, [[1, 0], [0, 1]]).shape
        (2, 2)
        """
        return apply_op(self, ir.Wishart(), nu, scale)

    # ----------------------------------------------------------------------------------------------
    # Indexing
    # ----------------------------------------------------------------------------------------------

    def getitem(self, x: RVLike, /, *indices: Index) -> ir.RV:
        """Implement bracket indexing using this interface's orthogonal indexing rules."""
        return self.oindex(x, *indices)

    def oindex(self, x: RVLike, *indices: Index) -> ir.RV:
        """Index an RV using fully orthogonal indexing.

        Every nonscalar index contributes all of its dimensions independently to the output. Slices
        are treated as one-dimensional integer arrays. This is intentionally simpler than NumPy, JAX,
        or PyTorch indexing and resembles ``oindex`` from `NEP 21
        <https://numpy.org/neps/nep-0021-advanced-indexing.html>`_.

        Every dimension must be explicitly indexed, or covered by a single ellipsis. Missing
        trailing indices are not filled with full slices. For a matrix ``x``, ``x[0]`` raises
        ``ValueError``; use ``x[0, :]`` or ``x[0, ...]`` instead. The same requirement applies to
        explicit ``oindex`` calls.

        Parameters
        ----------
        x
            Interface RV or array-like value to index.
        *indices
            One scalar, array-like value, interface RV, or slice per dimension of ``x``. A single
            ellipsis may stand for the required number of full slices.

        Returns
        -------
        indexed
            Interface RV containing an `ir.Index` Op wrapped in `ir.VMap` for each index
            dimension. Its shape is the concatenation of the index shapes.

        Raises
        ------
        ValueError
            If the number of indices does not match the rank of ``x`` or more than one ellipsis is
            supplied.

        Examples
        --------
        >>> import pangolin.interface as pi
        >>> A = pi.constant([[3, 0, 2], [4, 4, 4]])
        >>> pi.oindex(A, slice(None), [2, 2])
        RV(VMap(VMap(Index(), [None, None, 0], 2), [None, 0, None], 2), RV(Constant([[3,0,2],[4,4,4]])), RV(Constant([0,1])), RV(Constant([2,2])))
        >>> pi.oindex(A, 0, ...).shape
        (3,)

        A scalar can be indexed with no indices:

        >>> pi.oindex(pi.constant(12.0)).shape
        ()
        """
        var_rv = self.coerce(x)
        normalized_indices = eliminate_ellipses(var_rv.ndim, indices)
        index_rvs = convert_indices(self, var_rv.shape, *normalized_indices)
        return functional.index_orthogonal(var_rv, *index_rvs, interface=self)

    def vindex(self, var: RVLike, *indices: Index) -> ir.RV:
        """Index an RV using fully vectorized indexing.

        ``var`` is treated as a scalar function of its indices, which are broadcast according to the
        called interface's current ``config.broadcasting`` mode. Slices are converted to
        one-dimensional integer arrays before broadcasting. The result therefore uses Pangolin's
        explicit scalar broadcasting rules rather than NumPy's mixed basic/advanced indexing rules.

        The name honors ``vindex`` from `NEP 21
        <https://numpy.org/neps/nep-0021-advanced-indexing.html>`_, although Pangolin's precise
        semantics differ.

        Every dimension must be explicitly indexed, or covered by a single ellipsis. Missing
        trailing indices are not filled with full slices. For a matrix ``x``, ``x.vindex[0]`` raises
        ``ValueError``; use ``x.vindex[0, :]`` or ``x.vindex[0, ...]`` instead. The same requirement
        applies to explicit ``vindex`` calls. Expanded slices still follow the broadcasting rule.

        Parameters
        ----------
        var
            RV or array-like value to index.
        *indices
            One scalar, array-like, RV, or slice per dimension of ``var``. A single ellipsis may stand
            for the required number of full slices.

        Returns
        -------
        indexed
            Indexed RV whose shape is the common broadcast shape of the converted indices.

        Raises
        ------
        TypeError
            If no indices are supplied. Use an ellipsis to index a scalar RV.
        ValueError
            If the number of indices does not match the rank of ``var``, an ellipsis is invalid, or the
            index shapes cannot be broadcast under the configured mode.

        Examples
        --------
        Index a 2D array with two matching 2D arrays:

        >>> from pangolin import ir
        >>> import pangolin.interface as pi
        >>> A = pi.constant([[0., 0, 0], [1, 1, 1], [2, 2, 2]])
        >>> B = pi.constant([[0, 1, 2], [2, 1, 0]])
        >>> C = pi.constant([[2, 1, 1], [0, 0, 0]])
        >>> D = pi.vindex(A, B, C)
        >>> ir.print_upstream(D)
        shape  | statement
        ------ | ---------
        (3, 3) | a = [[0. 0. 0.] [1. 1. 1.] [2. 2. 2.]]
        (2, 3) | b = [[0 1 2] [2 1 0]]
        (2, 3) | c = [[2 1 1] [0 0 0]]
        (2, 3) | d = vmap(vmap(index, [None, 0, 0], 3), [None, 0, 0], 2)(a, b, c)

        With a square matrix, matching slices extract the diagonal:

        >>> A = pi.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> B = pi.vindex(A, slice(None), slice(None))
        >>> ir.print_upstream(B)
        shape  | statement
        ------ | ---------
        (3, 3) | a = [[1 2 3] [4 5 6] [7 8 9]]
        (3,)   | b = [0 1 2]
        (3,)   | c = [0 1 2]
        (3,)   | d = vmap(index, [None, 0, 0], 3)(a, b, c)

        Reversing one slice extracts the anti-diagonal:

        >>> C = pi.vindex(A, slice(None), slice(None, None, -1))
        >>> ir.print_upstream(C)
        shape  | statement
        ------ | ---------
        (3, 3) | a = [[1 2 3] [4 5 6] [7 8 9]]
        (3,)   | b = [0 1 2]
        (3,)   | c = [2 1 0]
        (3,)   | d = vmap(index, [None, 0, 0], 3)(a, b, c)
        """
        if not indices:
            raise TypeError("vindex requires at least one index; use an ellipsis to index a scalar")
        var_rv = self.coerce(var)
        normalized_indices = eliminate_ellipses(var_rv.ndim, indices)
        index_rvs = convert_indices(self, var_rv.shape, *normalized_indices)

        def scalar_index(*scalar_indices: ir.RV) -> ir.RV:
            if any(index_rv.shape != () for index_rv in scalar_indices):
                raise ValueError("non-scalar index")
            if len(scalar_indices) != var_rv.ndim:
                raise TypeError(f"got {len(scalar_indices)} indices for RV with {var_rv.ndim} dims")
            return apply_op(self, ir.Index(), var_rv, *scalar_indices)

        broadcast = broadcasting_combinators[self._effective_config.broadcasting]
        return broadcast(scalar_index)(*index_rvs)

    def __repr__(self) -> str:
        return f"Interface(broadcasting={self._effective_config.broadcasting!r})"

    # ----------------------------------------------------------------------------------------------
    # Named-loop combinators
    # ----------------------------------------------------------------------------------------------

    def loop(
        self,
        fun: Callable[..., RVLike],
        /,
        **sizes: LoopAxis,
    ) -> ir.RV:
        """Build an RV over named vectorized and sequential axes, in function-parameter order.

        Sequential loops support four calling styles:

        1. Decorator with a supplied axis size and initializer:

           >>> from pangolin import interface as pi
           >>> # x[i, 0] ~ normal(0, 1),          0 ≤ i < 3
           >>> # x[i, t] ~ normal(x[i, t-1], 1),  0 ≤ i < 3, 0 < t < 5
           >>> @pi.loop
           ... def x(prev, i=3, t=pi.seq(5, 0.0)):
           ...     return pi.normal(prev[i, t-1], 1)

        2. Direct call with a supplied initializer:

           >>> # x[i, 0] ~ normal(0, 1),          0 ≤ i < 3
           >>> # x[i, t] ~ normal(x[i, t-1], 1),  0 ≤ i < 3, 0 < t < 5
           >>> def model(prev, i, t):
           ...     return pi.normal(prev[i, t-1], 1)
           >>> x = pi.loop(model, i=3, t=pi.seq(5, 0.0))

        3. Decorator with an initialization branch:

           >>> # a[i] ~ normal(i, 1),             0 ≤ i < 3
           >>> # x[i, 0] ~ normal(a[i], 1),       0 ≤ i < 3
           >>> # x[i, t] ~ normal(x[i, t-1], 1),  0 ≤ i < 3, 0 < t < 5
           >>> @pi.loop
           ... def x(prev, i=3, t=pi.seq(5)):
           ...     if t is None:
           ...         return pi.normal(i, 1)
           ...     return pi.normal(prev[i, t-1], 1)

        4. Direct call with an initialization branch:

           >>> # a[i] ~ normal(i, 1),             0 ≤ i < 3
           >>> # x[i, 0] ~ normal(a[i], 1),       0 ≤ i < 3
           >>> # x[i, t] ~ normal(x[i, t-1], 1),  0 ≤ i < 3, 0 < t < 5
           >>> def model(prev, i, t):
           ...     if t is None:
           ...         return pi.normal(i, 1)
           ...     return pi.normal(prev[i, t-1], 1)
           >>> x = pi.loop(model, i=3, t=pi.seq(5))

        All axis specifications must come from parameter defaults or all from explicit
        call arguments. For vectorized axes, use an integer size instead of ``seq(...)``.
        Each access to ``prev`` must select one sequential axis using ``coordinate - 1``.
        Preceding indices must be the unchanged coordinates; remaining indices index
        the previous state normally. (Omit ``prev`` when no axis is sequential.)

        :rtype: RV

        Parameters
        ----------
        fun : callable
            Function taking scalar coordinates, preceded by ``prev`` when any axis
            is sequential. Returns one RV or numeric value.
        **sizes : int | list | tuple
            Positive integer for a vectorized axis; ``seq(T, initializer)`` for a sequential
            axis. An initializer is an RV or numeric value. ``seq(T)`` or ``seq(T, None)``
            calls fun with that coordinate set to None and the others supplied as RVs
            for initialization; prev cannot be read in this call.
            Equivalent lists and tuples are also accepted. Initial states are excluded
            from the output.

        Returns
        -------
        output : RV
            The output RV.

        Notes
        -----
        ``fun`` is traced once per required initialization branch and once for the body,
        not per iteration. In ``prev[...]``, preceding selectors must be the exact
        coordinate RVs in function-parameter order; the first differing selector must be
        ``coordinate - 1`` for a sequential coordinate. Remaining selectors index its
        previous state normally. Initialization cannot read ``prev``. In a sequential
        step, no newly created random RV may contribute to the result except the returned
        RV itself; initialization has no such restriction.

        Coordinates and coerced values use this interface. Existing RVs retain their
        interfaces. Body results and initializer values are coerced to RVs.

        Examples
        --------
        Vectorized axes, using a direct call:

        >>> from pangolin import interface as pi
        >>> # x[i, j] = 10 * i + j,  0 ≤ i < 2, 0 ≤ j < 3
        >>> x = pi.loop(lambda i, j: 10 * i + j, i=2, j=3)
        >>> x.shape
        (2, 3)
        >>> x.op
        VMap(VMap(Add(), [None, 0], 3), [0, None], 2)

        The same calculation declared with parameter defaults used to indicate axis sizes:

        >>> # y[i, j] = 10 * i + j,  0 ≤ i < 2, 0 ≤ j < 3
        >>> @pi.loop
        ... def y(i=2, j=3):
        ...     return 10 * i + j
        >>> y.shape
        (2, 3)
        >>> ir.rv_equal(x, y)
        True

        One sequential axis, initialized at zero:

        >>> # x[0] ~ normal(0, 1)
        >>> # x[t] ~ normal(x[t-1], 1),  0 < t < 3
        >>> @pi.loop
        ... def x(prev, t=pi.seq(3, 0.0)):
        ...     return pi.normal(prev[t-1], 1)
        >>> x.shape
        (3,)
        >>> x.op
        Scan(Composite(1, [Constant(1), Normal()], [[], [0, 1]]), 3, [])

        The same model using a None branch for initialization:

        >>> # x[0] ~ normal(0, 1)
        >>> # x[t] ~ normal(x[t-1], 1),  0 < t < 3
        >>> @pi.loop
        ... def x(prev, t=pi.seq(3)):
        ...     if t is None:
        ...         return 0.0
        ...     return pi.normal(prev[t-1], 1)
        >>> x.shape
        (3,)

        Add a vectorized axis for independent trajectories:

        >>> # x[i, 0] ~ normal(0, 1),          0 ≤ i < 2
        >>> # x[i, t] ~ normal(x[i, t-1], 1),  0 ≤ i < 2, 0 < t < 3
        >>> @pi.loop
        ... def x(prev, i=2, t=pi.seq(3)):
        ...     if t is None:
        ...         return 0.0
        ...     return pi.normal(prev[i, t-1], 1)
        >>> x.shape
        (2, 3)

        Initialization can depend on the other coordinates inside the function:

        >>> # a[i] ~ normal(i, 1),             0 ≤ i < 2
        >>> # x[i, 0] ~ normal(a[i], 1),       0 ≤ i < 2
        >>> # x[i, t] ~ normal(x[i, t-1], 1),  0 ≤ i < 2, 0 < t < 3
        >>> @pi.loop
        ... def x(prev, i=2, t=pi.seq(3)):
        ...     if t is None:
        ...         return pi.normal(i, 1)
        ...     return pi.normal(prev[i, t-1], 1)
        >>> x.shape
        (2, 3)
        >>> pi.print_upstream(x, {"x": x})
        shape  | statement
        ------ | ---------
        (2,)   | a = [0 1]
        ()     | b = 1
        (2,)   | c ~ vmap(normal, [0, None], 2)(a, b)
        (2, 3) | x ~ vmap(scan(composite(1, [1, normal], [[], [0, 1]]), 3, []), [0], 2)(c)

        Two sequential axes, each with its own initialization branch:

        >>> # x[0, 0] ~ normal(1, 1)
        >>> # x[0, s] ~ normal(x[0, s-1], 1),              0 < s < 3
        >>> # x[t, 0] ~ normal(x[t-1, 0] + 1, 1),          0 < t < 2
        >>> # x[t, s] ~ normal(x[t-1, s] + x[t, s-1], 1),  0 < t < 2, 0 < s < 3
        >>> @pi.loop
        ... def x(prev, t=pi.seq(2), s=pi.seq(3)):
        ...     if t is None:
        ...         return 0.0
        ...     elif s is None:
        ...         return 1.0
        ...     return pi.normal(prev[t-1, s] + prev[t, s-1], 1)
        >>> x.shape
        (2, 3)

        Nest loops to give each candidate its own drift, volatility, and random walk,
        with shared population parameters:

        >>> N, T = 3, 4
        >>> # drift_mean ~ normal(0, 0.2); drift_sd ~ lognormal(-2, 0.5)
        >>> # log_volatility_mean ~ normal(-2, 0.5)
        >>> drift_mean = pi.normal(0, 0.2)
        >>> drift_sd = pi.lognormal(-2, 0.5)
        >>> log_volatility_mean = pi.normal(-2, 0.5)
        >>> # For 0 ≤ i < N:
        >>> # drift[i] ~ normal(drift_mean, drift_sd)
        >>> # volatility[i] ~ lognormal(log_volatility_mean, 0.3)
        >>> # a[i] ~ normal(0, 1)
        >>> # popularity[i, 0] ~ normal(a[i] + drift[i], volatility[i])
        >>> # popularity[i, t] ~ normal(popularity[i, t-1] + drift[i], volatility[i]),  0 < t < T
        >>> @pi.loop
        ... def popularity(i=N):
        ...     drift = pi.normal(drift_mean, drift_sd)
        ...     volatility = pi.lognormal(log_volatility_mean, 0.3)
        ...     @pi.loop
        ...     def trajectory(prev, t=pi.seq(T)):
        ...         if t is None:
        ...             return pi.normal(0, 1)
        ...         return pi.normal(prev[t-1] + drift, volatility)
        ...     return trajectory
        >>> popularity.shape
        (3, 4)

        Use separate loops to keep drift and volatility directly accessible:

        >>> # For 0 ≤ i < N:
        >>> # drift[i] ~ normal(drift_mean, drift_sd)
        >>> @pi.loop
        ... def drift(i=N):
        ...     return pi.normal(drift_mean, drift_sd)
        >>> # For 0 ≤ i < N:
        >>> # volatility[i] ~ lognormal(log_volatility_mean, 0.3)
        >>> @pi.loop
        ... def volatility(i=N):
        ...     return pi.lognormal(log_volatility_mean, 0.3)
        >>> # For 0 ≤ i < N:
        >>> # a[i] ~ normal(0, 1)
        >>> # popularity[i, 0] ~ normal(a[i] + drift[i], volatility[i])
        >>> # popularity[i, t] ~ normal(popularity[i, t-1] + drift[i], volatility[i]),  0 < t < T
        >>> @pi.loop
        ... def popularity(prev, i=N, t=pi.seq(T)):
        ...     if t is None:
        ...         return pi.normal(0, 1)
        ...     return pi.normal(prev[i, t-1] + drift[i], volatility[i])
        >>> drift.shape, volatility.shape, popularity.shape
        ((3,), (3,), (3, 4))

        Put time before candidates to let each update depend on the previous
        population. A sequential axis carries the complete array of later axes:

        >>> # For 0 ≤ i < N:
        >>> # a[i] ~ normal(0, 1)
        >>> # popularity[0, i] ~ normal(0.9 * a[i] + 0.1 * mean(a), 1)
        >>> # m[t] = sum(popularity[t, j] for j in range(N)) / N,                     0 ≤ t < T
        >>> # popularity[t, i] ~ normal(0.9 * popularity[t-1, i] + 0.1 * m[t-1], 1),  0 < t < T
        >>> @pi.loop
        ... def popularity(prev, t=pi.seq(T), i=N):
        ...     if t is None:
        ...         return pi.normal(0, 1)
        ...     population_mean = pi.sum(prev[t-1, :], 0) / N
        ...     return pi.normal(
        ...         0.9 * prev[t-1, i] + 0.1 * population_mean,
        ...         1,
        ...     )
        >>> popularity.shape
        (4, 3)
        """
        axes = loop_specifications(fun, sizes)

        @wraps(fun)
        def call(*args: Any, **kwargs: Any) -> ir.RV:
            return self.coerce(fun(*args, **kwargs))

        for name, specification in axes.items():
            if isinstance(specification, (list, tuple)) and len(specification) == 2:
                size, initializer = specification
                if initializer is not None:
                    if callable(initializer):
                        raise TypeError(
                            f"axis {name!r}: initializer must be an RV, a numeric value, or None"
                        )
                    axes[name] = (size, self.coerce(initializer))
        return loop_from_specifications(call, axes, interface=self)

    @typing.overload
    def vfor[Output: PyTree[ir.RV]](
        self,
        fun: Callable[[], Output],
        /,
        **sizes: int,
    ) -> Output: ...

    @typing.overload
    def vfor[Output: PyTree[ir.RV]](
        self,
        fun: Callable[[ir.RV], Output],
        /,
        **sizes: int,
    ) -> Output: ...

    @typing.overload
    def vfor[Output: PyTree[ir.RV]](
        self,
        fun: Callable[[ir.RV, ir.RV], Output],
        /,
        **sizes: int,
    ) -> Output: ...

    @typing.overload
    def vfor[Output: PyTree[ir.RV]](
        self,
        fun: Callable[[ir.RV, ir.RV, ir.RV], Output],
        /,
        **sizes: int,
    ) -> Output: ...

    @typing.overload
    def vfor[Output: PyTree[ir.RV]](
        self,
        fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], Output],
        /,
        **sizes: int,
    ) -> Output: ...

    @typing.overload
    def vfor[Output: PyTree[ir.RV]](
        self,
        fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], Output],
        /,
        **sizes: int,
    ) -> Output: ...

    @typing.overload
    def vfor[Output: PyTree[ir.RV], *Rest](
        self,
        fun: Callable[
            [ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest],
            Output,
        ],
        /,
        **sizes: int,
    ) -> Output: ...

    @typing.overload
    def vfor(
        self,
        fun: None = None,
        /,
        **sizes: int,
    ) -> VForDecorator: ...

    def vfor[Output: PyTree[ir.RV]](
        self,
        fun: Callable[..., Output] | None = None,
        /,
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

        Coordinate RVs carry this interface. Output RVs retain the interfaces chosen by ``fun``;
        they are not required to carry this interface.

        This combinator supports direct calls, decorator factories, and multiline
        decorator syntax.

        Parameters
        ----------
        fun : Callable[..., PyTree[ir.RV]] | None
            Function to vectorize, or ``None`` when creating a decorator. It must have a fixed
            positional signature, accept one scalar `ir.RV` coordinate per parameter, and return a
            PyTree of newly created `ir.RV` objects. No parameter may be named ``interface``.
        **sizes : int
            Positive integer size for every parameter of ``fun``, supplied under the corresponding
            parameter name.

        Returns
        -------
        output_or_decorator : PyTree[ir.RV] | decorator
            The vectorized output PyTree when ``fun`` is supplied, otherwise a decorator that
            accepts ``fun`` and produces that output.

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
        of the output, and requires the first six arguments to be `ir.RV` objects. Runtime
        validation still applies to every argument.

        Examples
        --------
        Direct call:

        >>> from pangolin import interface as pi
        >>> x = pi.vfor(lambda i, j: 10 * i + j, i=2, j=3)
        >>> x.shape
        (2, 3)

        Keyword order does not affect axis order:

        >>> y = pi.vfor(lambda i, j: 10 * i + j, j=3, i=2)
        >>> y.shape
        (2, 3)
        >>> ir.rv_equal(x, y)
        True

        Decorator factory:

        >>> make_x = pi.vfor(i=2, j=3)
        >>> x = make_x(lambda i, j: 10 * i + j)
        >>> x.shape
        (2, 3)

        Multiline decorator form:

        >>> @pi.vfor(i=2, j=3)
        ... def x(i, j):
        ...     return 10 * i + j
        >>> x.shape
        (2, 3)

        Every function parameter must have a named size:

        >>> pi.vfor(lambda i, j: i + j, i=2)
        Traceback (most recent call last):
        ...
        ValueError: vfor missing size for axis 'j'

        Unknown axis names are rejected:

        >>> pi.vfor(lambda i, j: i + j, i=2, j=3, k=4)
        Traceback (most recent call last):
        ...
        ValueError: vfor got size for unknown axis 'k'; expected ('i', 'j')

        Python defaults are not interpreted as axis sizes:

        >>> pi.vfor(lambda i=3: i + 1)
        Traceback (most recent call last):
        ...
        TypeError: <lambda> has no fixed positional signature: 'i' has a default
        """

        if fun is None:
            return functional.vfor(interface=self, **sizes)
        return functional.vfor(fun, interface=self, **sizes)

    @typing.overload
    def sfor(
        self,
        fun: Callable[[ir.RV, ir.RV], ir.RV],
        /,
        *,
        init: RVLike,
        **sizes: int,
    ) -> ir.RV: ...

    @typing.overload
    def sfor(
        self,
        fun: Callable[[ir.RV, ir.RV, ir.RV], ir.RV],
        /,
        *,
        init: RVLike,
        **sizes: int,
    ) -> ir.RV: ...

    @typing.overload
    def sfor(
        self,
        fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
        /,
        *,
        init: RVLike,
        **sizes: int,
    ) -> ir.RV: ...

    @typing.overload
    def sfor(
        self,
        fun: Callable[[ir.RV, ir.RV, ir.RV, ir.RV, ir.RV], ir.RV],
        /,
        *,
        init: RVLike,
        **sizes: int,
    ) -> ir.RV: ...

    @typing.overload
    def sfor[*Rest](
        self,
        fun: Callable[
            [ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, ir.RV, *Rest],
            ir.RV,
        ],
        /,
        *,
        init: RVLike,
        **sizes: int,
    ) -> ir.RV: ...

    @typing.overload
    def sfor(
        self,
        fun: None = None,
        /,
        *,
        init: RVLike,
        **sizes: int,
    ) -> SForDecorator: ...

    def sfor(
        self,
        fun: Callable[..., ir.RV] | None = None,
        /,
        *,
        init: RVLike,
        **sizes: int,
    ) -> ir.RV | SForDecorator:
        """Build a state sequence with named sequential and parallel coordinates.

        In the examples below, ``pi`` is this interface. This combinator constructs the result
        directly. A scalar call of the form::

            x = pi.sfor(fun, init=init, t=T)

        has these semantics::

            previous = init if isinstance(init, ir.RV) else pi.constant(init)
            states = []
            for t in range(T):
                t_rv = pi.constant(t)
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

            x = pi.sfor(fun, init=init, t=T, i=I, j=J)

        has these semantics::

            previous = init if isinstance(init, ir.RV) else pi.constant(init)
            states = []
            for t in range(T):
                t_rv = pi.constant(t)
                next_state = empty((I, J))
                for i in range(I):
                    i_rv = pi.constant(i)
                    for j in range(J):
                        j_rv = pi.constant(j)
                        next_state[i, j] = fun(previous, t_rv, i_rv, j_rv)
                states.append(next_state)
                previous = next_state
            x = stack(states)

        This pseudocode describes the meaning of the loops, not the graph constructed internally.
        Internally, each coordinate range is a constant RV array with this interface, and
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

            @pi.sfor(init=init, t=T)
            def x(previous, t):
                return normal(previous, 1)

        is equivalent to::

            def fun(previous, t):
                return normal(previous, 1)

            x = pi.sfor(fun, init=init, t=T)

        In both forms, ``init`` must be supplied as a keyword argument.

        Each call to ``fun`` must return exactly one `ir.RV` created during that call. The result
        must depend on ``previous``. Any other RVs created during the call that contribute to the
        result must have deterministic Ops. The assembled state must have the same interface as
        the coerced ``init``, which need not be this interface when ``init`` is already an RV.

        After coercion, the shape of ``init`` must match one complete state assembled from ``fun``.
        For ``fun(previous, t, i, j)``, let ``S`` be the shape returned by one call to ``fun``.
        With ``t=T``, ``i=I``, and ``j=J``, ``init`` must have shape ``(I, J, *S)``, and the result has
        shape ``(T, I, J, *S)``. Equivalently, ``result.shape == (T,) + init.shape``. The output
        axes appear in function-parameter order; keyword order is ignored.

        Each parallel call receives the complete previous state; ``sfor`` does not slice it along
        the parallel axes.

        Parameters
        ----------
        fun : Callable[..., ir.RV] | None
            Step function, or ``None`` in decorator form. It must have a fixed positional signature
            with at least two parameters and no defaults, variadic parameters, or keyword-only
            parameters. The first parameter receives the complete previous state; the second
            receives the scalar sequential coordinate; any later parameters receive scalar parallel
            coordinates. No parameter may be named ``init`` or ``interface``. The function must
            satisfy the return, carry-dependence, determinism, shape, and interface requirements
            above.
        init : RVLike
            Keyword-only complete state before the first step. An existing RV is preserved,
            including its interface; other values become constant RVs owned by this interface.
            The resulting RV's shape and interface must match the state assembled by mapping
            ``fun`` over every parallel coordinate.
        **sizes : int
            Exactly one positive integer for every coordinate parameter: the sequential parameter
            and all parallel parameters. Names must match the function parameters. Values determine
            axis lengths; function-parameter order, not keyword order, determines axis order.

        Returns
        -------
        result_or_decorator : ir.RV | decorator
            When ``fun`` is supplied, an `ir.RV` containing every state produced by the scan, not
            including the coerced ``init``. If the sequential size is ``T``, ``result[t]`` is the
            state produced at coordinate ``t`` and ``result.shape == (T,) + init.shape``. When
            ``fun`` is omitted, a decorator that accepts it and returns such an `ir.RV`. The
            result has the coerced initial state's interface; it is not required to carry this
            interface.

        Raises
        ------
        TypeError
            If ``fun`` is not callable or lacks the required signature, a size is not an integer,
            or ``fun`` does not return exactly one `ir.RV`.
        ValueError
            If a parameter is named ``init`` or ``interface``, the function has no sequential
            coordinate, sizes are missing, unknown, or nonpositive, the result does not depend on
            ``previous``, an intermediate Op is random, or the mapped state does not match the shape
            or interface of the coerced ``init``, or a non-RV initial value is not a valid constant.

        Examples
        --------
        >>> from pangolin.interface import Interface
        >>> pi = Interface()
        >>> walk = pi.sfor(lambda previous, t: pi.normal(previous, 1), init=0, t=4)
        >>> walk.shape
        (4,)
        >>> walk.interface is pi
        True

        The initial RV and output can belong to a different interface:

        >>> other = Interface()
        >>> start = other.constant(0)
        >>> walk = pi.sfor(lambda previous, t: other.normal(previous, 1), init=start, t=3)
        >>> walk.interface is other
        True
        """

        initial_state = init if isinstance(init, ir.RV) else self.constant(init)
        if fun is None:
            return functional.sfor(init=initial_state, interface=self, **sizes)
        return functional.sfor(fun, init=initial_state, interface=self, **sizes)

    # ----------------------------------------------------------------------------------------------
    # Constants and coercion
    # ----------------------------------------------------------------------------------------------

    def constant(self, value: ArrayLike, /) -> ir.RV:
        """Create an interface RV representing a fixed numeric value.

        ``value`` is normalized by `pangolin.ir.Constant` to an immutable NumPy array. Scalars remain
        scalar-shaped, while nested sequences and arrays retain their array shape.

        Parameters
        ----------
        value
            A numeric scalar, sequence, NumPy array, JAX array, or other numeric array-like value.

        Raises
        ------
        ValueError
            If ``value`` is empty or does not have a numeric dtype.

        Examples
        --------
        >>> from pangolin import interface as pi
        >>> pi.constant(7)
        RV(Constant(7))
        >>> x = pi.constant([[1, 2], [3, 4]])
        >>> x.shape
        (2, 2)
        """
        return ir.RV(ir.Constant(value), interface=self)

    def coerce(self, value: RVLike, /) -> ir.RV:
        """Preserve an RV or create a constant owned by this interface."""
        if isinstance(value, ir.RV):
            return value
        return self.constant(value)


# --------------------------------------------------------------------------------------------------
# Default interface
# --------------------------------------------------------------------------------------------------


default: Final = Interface()
tforms = default.tforms

constant = default.constant
overload = default.overload
broadcast = default.broadcast
cdf = default.cdf
quantile = default.quantile
log_prob = default.log_prob
truncated = default.truncated
grad = default.grad

add = default_function(default.add)
sub = default_function(default.sub)
mul = default_function(default.mul)
div = default_function(default.div)
pow = default_function(default.pow)
sqrt = default.sqrt
neg = default.neg

abs = default_function(default.abs)
minimum = default_function(default.minimum)
maximum = default_function(default.maximum)
arccos = default_function(default.arccos)
arccosh = default_function(default.arccosh)
arcsin = default_function(default.arcsin)
arcsinh = default_function(default.arcsinh)
arctan = default_function(default.arctan)
arctanh = default_function(default.arctanh)
cos = default_function(default.cos)
cosh = default_function(default.cosh)
sin = default_function(default.sin)
sinh = default_function(default.sinh)
tan = default_function(default.tan)
tanh = default_function(default.tanh)
exp = default_function(default.exp)
inv_logit = default_function(default.inv_logit)
expit = default.expit
sigmoid = default.sigmoid
log = default_function(default.log)
logit = default_function(default.logit)
step = default_function(default.step)
loggamma = default_function(default.loggamma)

normal = default_function(default.normal)
normal_prec = default_function(default.normal_prec)
lognormal = default_function(default.lognormal)
bernoulli = default_function(default.bernoulli)
bernoulli_logit = default_function(default.bernoulli_logit)
binomial = default_function(default.binomial)
cauchy = default_function(default.cauchy)
uniform = default_function(default.uniform)
beta = default_function(default.beta)
exponential = default_function(default.exponential)
gamma = default_function(default.gamma)
poisson = default_function(default.poisson)
beta_binomial = default_function(default.beta_binomial)
student_t = default_function(default.student_t)

matmul = default.matmul
transpose = default.transpose
diag = default.diag
diag_matrix = default.diag_matrix
inv = default.inv
cholesky = default.cholesky
softmax = default.softmax
sum = default.sum
concatenate = default.concatenate
fill_tril = default.fill_tril
extract_tril = default.extract_tril
multi_normal = default.multi_normal
categorical = default.categorical
multinomial = default.multinomial
dirichlet = default.dirichlet
wishart = default.wishart

oindex = default.oindex
vindex = default.vindex
loop = default.loop
vfor = default.vfor
sfor = default.sfor
