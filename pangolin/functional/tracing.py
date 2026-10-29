"""Trace calls that create Pangolin IR graphs."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, Final, SupportsIndex, cast

import jax

from pangolin import dag, ir, util
from pangolin.dag import upstream_nodes_flat
from pangolin.util import PyTree, tree_flatten_with_none_as_leaf

__all__ = ["bind_interface", "graph_scan"]


def require_rv_leaves(tree: Any, context: str) -> list[ir.RV]:
    """Return the leaves of ``tree`` after requiring every leaf to be an IR RV."""
    leaves, _ = tree_flatten_with_none_as_leaf(tree)
    for leaf in leaves:
        if not isinstance(leaf, ir.RV):
            raise TypeError(
                f"{context} must be a PyTree whose leaves are RV; found {type(leaf).__name__}"
            )
    # The loop above established every leaf's runtime type, while PyTree flattening exposes Any.
    return cast(list[ir.RV], leaves)


def flatten_rv_inputs(
    tree: PyTree[object],
    in_axes: PyTree[int | None],
    context: str,
) -> tuple[list[object], list[ir.RV], list[int | None], list[int]]:
    """Flatten inputs while validating axes and identifying every RV leaf."""
    axis_leaves, leaves = util.dual_flatten(in_axes, tree)

    axes: list[int | None] = []
    for axis in axis_leaves:
        if axis is None:
            axes.append(None)
        elif isinstance(axis, bool) or not isinstance(axis, SupportsIndex):
            raise TypeError(f"each {context} axis must be an integer or None")
        else:
            axes.append(operator.index(axis))

    rvs: list[ir.RV] = []
    rv_axes: list[int | None] = []
    rv_positions: list[int] = []
    for position, (axis, leaf) in enumerate(zip(axes, leaves, strict=True)):
        if isinstance(leaf, ir.RV):
            rvs.append(leaf)
            rv_axes.append(axis)
            rv_positions.append(position)
        elif axis is not None:
            raise TypeError(
                f"{context} axis for non-RV argument must be None; got axis {axis} for "
                f"{type(leaf).__name__}"
            )

    return leaves, rvs, rv_axes, rv_positions


def graph_scan[Nodes: PyTree[ir.RV], Emission](
    nodes: Nodes,
    step: Callable[[ir.RV, tuple[ir.RV, ...]], tuple[ir.RV, Emission]],
    *,
    boundary: Mapping[ir.RV, ir.RV] | None = None,
) -> tuple[Nodes, dict[ir.RV, Emission]]:
    """Scan an IR graph with RV carries and preserve the requested PyTree structure.

    This is the IR-specialized form of `pangolin.dag.graph_scan`. Each node and propagated carry
    is an `ir.RV`; ``step`` may rebuild or otherwise replace each visited node while emitting
    arbitrary auxiliary information.

    Parameters
    ----------
    nodes
        RV or PyTree of RVs whose upstream graph should be scanned.
    step
        Function returning a replacement RV and an emission from one input RV and its replacement
        parents.
    boundary
        Optional mapping from input RVs to replacement RVs at which traversal stops.

    Returns
    -------
    replacements
        Replacement RVs for ``nodes`` with the same runtime PyTree structure.
        The generic return type describes this structure, not preservation of RV subclasses.
    emissions
        Emission from each visited input RV, keyed by that input RV.

    Raises
    ------
    TypeError
        If ``nodes`` has a non-RV leaf or ``step`` returns a non-RV carry.
    """
    require_rv_leaves(nodes, "functional.graph_scan nodes")

    def checked_step(node: ir.RV, parent_replacements: tuple[ir.RV, ...]) -> tuple[ir.RV, Emission]:
        replacement, emission = step(node, parent_replacements)
        if not isinstance(replacement, ir.RV):
            raise TypeError(
                "functional.graph_scan step must return an RV carry; "
                f"got {type(replacement).__name__}"
            )
        return replacement, emission

    # ty incorrectly binds dag.graph_scan's node type to the complete recursive PyTree union
    # rather than to its RV leaves; this wrapper fixes every node and carry type to ir.RV.
    replacements, emissions = dag.graph_scan(
        nodes,  # ty: ignore[invalid-argument-type]
        checked_step,
        boundary=boundary,
    )
    # graph_scan preserves the input PyTree structure, which dag.graph_scan's return type erases;
    # Nodes describes that structure with plain RV leaves, not specialized RV subclasses.
    return cast(Nodes, replacements), emissions


def trace_call(
    fun: Callable[..., PyTree[ir.RV]],
    /,
    *args: Any,
    **kwargs: Any,
) -> tuple[PyTree[ir.RV], list[ir.RV], list[ir.RV]]:
    """Call ``fun`` once and describe the new IR graph contributing to its output.

    ``output`` is the exact PyTree returned by ``fun``. Every leaf must be an `ir.RV`, but an
    output RV may either be created during the call or have existed before it.

    ``generated`` contains every RV created during the call that contributes to an output,
    including intermediate RVs and newly created output RVs. Each distinct RV appears exactly once
    in creation order. New RVs that do not contribute to an output are omitted.

    ``boundary`` contains every pre-existing output RV and every pre-existing parent of a generated
    RV. Each distinct RV appears exactly once. Contributing RVs found in ``args`` or ``kwargs``
    appear first in flattened argument order; captured RVs follow in first-encounter order.

    Parameters
    ----------
    fun
        Function to call. Its output must be a PyTree whose leaves are `ir.RV` objects.
    *args
        Positional arguments passed to ``fun`` without modification.
    **kwargs
        Keyword arguments passed to ``fun`` without modification.

    Returns
    -------
    output
        Exact output returned by ``fun``, including its PyTree structure and repeated references.
    generated
        Newly created RVs contributing to ``output``, once each in creation order.
    boundary
        Pre-existing output RVs and external parents of generated RVs, once each in boundary order.

    Raises
    ------
    TypeError
        If any leaf of ``output`` is not an `ir.RV`.

    Examples
    --------
    A newly created output appears in ``generated``, while its existing input appears in
    ``boundary``:

    >>> x = ir.RV(ir.Constant(2))
    >>> output, generated, boundary = trace_call(lambda value: ir.RV(ir.Exp(), value), x)
    >>> output is generated[0]
    True
    >>> boundary == [x]
    True

    Returning an existing RV creates no new nodes:

    >>> output, generated, boundary = trace_call(lambda value: value, x)
    >>> output is x
    True
    >>> generated, boundary == [x]
    ([], True)

    RVs captured by the function are also included in the boundary:

    >>> offset = ir.RV(ir.Constant(3))
    >>> output, generated, boundary = trace_call(lambda value: ir.RV(ir.Add(), value, offset), x)
    >>> generated == [output]
    True
    >>> boundary == [x, offset]
    True

    Tracing returns the exact RV objects created by the function, including their interfaces:

    >>> interface = ir.RVInterface()
    >>> special = ir.RV(ir.Constant(2), interface=interface)
    >>> output, generated, boundary = trace_call(
    ...     lambda value: ir.RV(ir.Exp(), value, interface=value.interface), special
    ... )
    >>> output.interface is interface
    True
    >>> boundary == [special]
    True

    Arguments may be arbitrary Python objects. Only pre-existing RV arguments can appear in
    ``boundary``; RVs constructed during the call appear in ``generated``:

    >>> def build(value, take_log):
    ...     base = ir.RV(ir.Constant(value))
    ...     op = ir.Log() if take_log else ir.Exp()
    ...     return ir.RV(op, base)
    >>> output, generated, boundary = trace_call(build, 3, True)
    >>> [rv.op for rv in generated]
    [Constant(3), Log()]
    >>> generated[-1] is output
    True
    >>> boundary
    []
    """
    creation_boundary = ir.RV._n

    def is_existing(rv: ir.RV) -> bool:
        return rv._n < creation_boundary

    output = fun(*args, **kwargs)
    output_leaves = require_rv_leaves(output, "trace_call output")
    output_rvs: list[ir.RV] = []
    for leaf in output_leaves:
        if leaf not in output_rvs:
            output_rvs.append(leaf)

    generated = upstream_nodes_flat(output_rvs, node_block=is_existing)
    generated.sort(key=lambda rv: rv._n)

    discovered_boundary = [rv for rv in output_rvs if is_existing(rv)]
    for rv in generated:
        for parent in rv.parents:
            if parent not in generated and parent not in discovered_boundary:
                discovered_boundary.append(parent)

    argument_rvs: list[ir.RV] = []
    argument_leaves, _ = tree_flatten_with_none_as_leaf((args, kwargs))
    for leaf in argument_leaves:
        if isinstance(leaf, ir.RV) and leaf not in argument_rvs:
            argument_rvs.append(leaf)

    boundary = [rv for rv in argument_rvs if rv in discovered_boundary] + [
        rv for rv in discovered_boundary if rv not in argument_rvs
    ]
    return output, generated, boundary


def bind_interface[Output: PyTree[ir.RV], **P](
    interface: ir.RVInterface,
    fun: Callable[P, Output],
    /,
) -> Callable[P, Output]:
    """Bind the RV inputs seen by ``fun`` to an interface while preserving its signature.

    The returned function accepts RVs carrying any interfaces. On each call, ``fun`` is traced
    with abstract replacements carrying ``interface`` for every RV input. Non-RV inputs are passed
    through unchanged. The traced graph is then replayed with the exact original RVs replacing the
    abstract inputs.

    Captured RVs remain unchanged, while every generated RV carries the interface selected by its
    corresponding traced RV. Repeated input and output RV identities are preserved, and returning
    an input directly returns that exact input object.

    Parameters
    ----------
    interface
        Interface attached to the abstract RV inputs supplied to ``fun``.
    fun
        RV-producing function to bind. Its output must be a PyTree of RVs.

    Returns
    -------
    bound_fun
        Function with the same signature and output PyTree type as ``fun``.

    Raises
    ------
    TypeError
        If an output leaf is not an `ir.RV`.

    Examples
    --------
    Bind infix addition to a small interface without changing the actual input RV:

    >>> class AddInterface(ir.RVInterface):
    ...     def add(self, x, y):
    ...         if not isinstance(y, ir.RV):
    ...             y = ir.RV(ir.Constant(y), interface=self)
    ...         return ir.RV(ir.Add(), x, y, interface=self)
    >>> interface = AddInterface()
    >>> add_one = bind_interface(interface, lambda x: x + 1)
    >>> x = ir.RV(ir.Constant(2))
    >>> result = add_one(x)
    >>> result.parents[0] is x
    True
    >>> result.interface is interface
    True
    """

    @wraps(fun)
    def bound_fun(*args: P.args, **kwargs: P.kwargs) -> Output:
        input_leaves, input_treedef = tree_flatten_with_none_as_leaf((args, kwargs))
        abstract_by_actual: dict[ir.RV, ir.RV] = {}
        abstract_leaves: list[object] = []
        for leaf in input_leaves:
            if isinstance(leaf, ir.RV):
                if leaf not in abstract_by_actual:
                    abstract_by_actual[leaf] = ir.RV(AbstractOp(leaf.shape), interface=interface)
                abstract_leaves.append(abstract_by_actual[leaf])
            else:
                abstract_leaves.append(leaf)

        abstract_args, abstract_kwargs = jax.tree_util.tree_unflatten(
            input_treedef, abstract_leaves
        )
        traced_output, _, boundary = trace_call(fun, *abstract_args, **abstract_kwargs)

        actual_by_traced = {abstract: actual for actual, abstract in abstract_by_actual.items()}
        for captured in boundary:
            if captured not in actual_by_traced:
                actual_by_traced[captured] = captured

        def replay(node: ir.RV, parents: tuple[ir.RV, ...]) -> tuple[ir.RV, None]:
            return ir.RV(node.op, *parents, interface=node.interface), None

        actual_output, _ = graph_scan(traced_output, replay, boundary=actual_by_traced)
        # graph_scan preserves the exact output PyTree structure declared by fun, but its general
        # PyTree annotation cannot express that concrete structure.
        return cast(Output, actual_output)

    return bound_fun


class AbstractOp(ir.Op):
    """Represent a distinct, deterministic tracing input with a fixed shape."""

    random: bool = False
    """False"""

    def __init__(self, shape: ir.Shape = ()):
        if not isinstance(shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) for size in shape
        ):
            raise TypeError("AbstractOp shape must be a tuple of integers")
        if any(size <= 0 for size in shape):
            raise ValueError("AbstractOp dimensions must be positive")

        self.shape: Final = shape
        super().__init__()

    def get_shape(self) -> ir.Shape:
        """Return the fixed placeholder shape."""
        return self.shape

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is this exact placeholder."""
        return self is other

    def __hash__(self) -> int:
        """Return the identity-based hash of this placeholder."""
        return object.__hash__(self)

    def __repr__(self) -> str:
        """Return a constructor-shaped representation."""
        if self.shape == ():
            return "AbstractOp()"
        return f"AbstractOp({self.shape})"

    def __str__(self) -> str:
        """Return a compact representation."""
        return "abstract()"
