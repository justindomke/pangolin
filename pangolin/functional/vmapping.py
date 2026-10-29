"""Provide the `vmap` functional combinator."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import cast

import jax

from pangolin import ir
from pangolin.util import PyTree, tree_flatten_with_none_as_leaf

from .simplifying import permutation_view, unpermuted_mapped_axis
from .tracing import AbstractOp, flatten_rv_inputs, trace_call

__all__ = ["vmap"]


def normalize_in_axes(in_axes: PyTree[int | None], num_args: int) -> PyTree[int | None]:
    """Normalize the top level of ``in_axes`` to match the positional arguments.

    Parameters
    ----------
    in_axes
        Axis specification for the mapped function's positional arguments.
    num_args
        Number of positional arguments.

    Returns
    -------
    normalized_in_axes
        Axis specification whose top level represents all positional arguments.
    """
    if num_args == 1:
        return (in_axes,)
    if isinstance(in_axes, list):
        return tuple(in_axes)
    return in_axes


def make_abstract_rvs(
    actual_rvs: list[ir.RV],
    in_axes: list[int | None],
    axis_size: int | None,
) -> tuple[list[ir.RV], int]:
    """Create abstract RVs with mapped axes removed and resolve the mapped axis size.

    Parameters
    ----------
    actual_rvs
        Flat list of actual RV arguments.
    in_axes
        Mapped axis of each RV, or ``None`` for an unmapped RV.
    axis_size
        Explicit mapped size, or ``None`` to infer it from a mapped RV.

    Returns
    -------
    abstract_rvs
        Abstract RVs carrying the interfaces of the corresponding ``actual_rvs``.
    resolved_axis_size
        Positive common mapped-axis size.

    Raises
    ------
    ValueError
        If the argument lists differ in length, mapped dimensions disagree, or the axis size is
        unavailable or non-positive.
    """
    abstract_rvs = []
    for input_number, (actual_rv, axis) in enumerate(zip(actual_rvs, in_axes, strict=True)):
        shape, mapped_size = ir.split_shape(actual_rv.shape, axis)
        abstract_rvs.append(ir.RV(AbstractOp(shape), interface=actual_rv.interface))

        if axis_size is None:
            axis_size = mapped_size
        elif mapped_size is not None and mapped_size != axis_size:
            raise ValueError(
                f"input {input_number} has mapped axis size {mapped_size}, expected {axis_size}"
            )

    if axis_size is None:
        raise ValueError("axis_size is required when no inputs are mapped")
    if axis_size <= 0:
        raise ValueError("axis_size must be positive")
    return abstract_rvs, axis_size


def lift_graph(
    abstract_rvs: list[ir.RV],
    rv_axes: list[int | None],
    generated: list[ir.RV],
    outputs: list[ir.RV],
    actual_rvs: list[ir.RV],
    axis_size: int,
    *,
    simplify: bool,
) -> list[ir.RV]:
    """Replay a traced scalar graph using actual arguments and VMap operations.

    Each replayed node carries the interface of its corresponding traced node. A simplification may
    reuse an existing base RV only when doing so preserves that interface.

    Parameters
    ----------
    abstract_rvs
        Abstract input RVs used to trace the scalar graph.
    rv_axes
        Mapped axis of each abstract RV, or ``None`` when it is unmapped.
    generated
        Newly generated traced RVs in creation order.
    outputs
        Traced RVs whose lifted counterparts should be returned.
    actual_rvs
        Actual RVs replacing ``abstract_rvs``.
    axis_size
        Size of the mapped axis.
    simplify
        Whether to eliminate recognized indexing and permutation views.

    Returns
    -------
    lifted_outputs
        Lifted RVs corresponding positionally to ``outputs``.
    """
    real = dict(zip(abstract_rvs, actual_rvs, strict=True))
    mapped_axes = dict(zip(abstract_rvs, rv_axes, strict=True))
    output_set = set(outputs)

    for node in generated:
        if isinstance(node.op, AbstractOp):
            raise ValueError("generated graph cannot contain an abstract input operation")
        parents = tuple(real.get(parent, parent) for parent in node.parents)
        in_axes = tuple(mapped_axes.get(parent) for parent in node.parents)
        is_output = node in output_set

        if all(axis is None for axis in in_axes) and not node.op.random and not is_output:
            real_node = ir.RV(node.op, *parents, interface=node.interface)
            mapped_axis = None

            if simplify:
                view = permutation_view(real_node)
                if view is not None:
                    base, permutation = view
                    if (
                        permutation == tuple(range(base.ndim))
                        and base.interface is real_node.interface
                    ):
                        real_node = base
        else:
            real_node = ir.RV(
                ir.VMap(node.op, in_axes, axis_size),
                *parents,
                interface=node.interface,
            )
            mapped_axis = 0

            if simplify:
                replacement = unpermuted_mapped_axis(real_node, is_output=is_output)
                if replacement is not None:
                    base, replacement_axis = replacement
                    if base.interface is real_node.interface:
                        real_node = base
                        mapped_axis = replacement_axis

        real[node] = real_node
        mapped_axes[node] = mapped_axis

    return [real[output] for output in outputs]


def vmap[Output: PyTree[ir.RV], *Args](
    fun: Callable[[*Args], Output],
    in_axes: PyTree[int | None] = 0,
    axis_size: int | None = None,
    *,
    simplify: bool = True,
) -> Callable[[*Args], Output]:
    """A combinator that returns a vectorized version of an IR-producing function.

    Pangolin calls ``fun`` once with abstract RVs and lifts the resulting scalar graph. Each
    abstract input carries the actual input's interface, and each lifted node carries the interface
    selected by the corresponding traced node.

    An input leaf assigned an integer axis must be an RV. A leaf assigned ``None`` may be either an
    RV or any Python value; non-RV values are passed to ``fun`` unchanged and never become VMap
    parents. Outputs must be a PyTree of newly created RVs.

    Parameters
    ----------
    fun
        Function whose output is a PyTree of newly created `ir.RV` objects.
    in_axes
        Integer, ``None``, or PyTree selecting an axis for each RV input. The specification may be
        a prefix of the positional argument PyTree. ``None`` leaves an RV unmapped or passes a
        non-RV value unchanged.
    axis_size
        Explicit mapped-axis size. It is inferred from the first mapped RV when omitted and is
        required when no RV input is mapped.
    simplify
        Whether to eliminate recognized indexing and axis-permutation patterns.

    Returns
    -------
    vectorized_fun
        Function accepting the original positional argument structures and returning the original
        output PyTree structure with a leading mapped axis on each RV.

    Raises
    ------
    TypeError
        If an integer axis is assigned to a non-RV argument or an output leaf is not an `ir.RV`.
    ValueError
        If mapped sizes disagree, no axis size is available, or an output RV was not created by
        ``fun``.

    Examples
    --------
    The combinator returns a function that maps a scalar exponential over an RV vector:

    >>> def scalar_exp(x):
    ...     return ir.RV(ir.Exp(), x)
    >>> mapped_exp = vmap(scalar_exp)
    >>> mapped_exp  # doctest: +ELLIPSIS
    <function vmap_scalar_exp at 0x...>
    >>> values = ir.RV(ir.Constant([1.0, 2.0, 3.0]))
    >>> result = mapped_exp(values)
    >>> result.op
    VMap(Exp(), [0], 3)
    >>> result.parents == (values,)
    True

    An unmapped Python value is passed through without becoming a parent:

    >>> def choose(x, operation):
    ...     op = ir.Log() if operation == "log" else ir.Exp()
    ...     return ir.RV(op, x)
    >>> mapped_choose = vmap(choose, in_axes=(0, None))
    >>> result = mapped_choose(values, "log")
    >>> result.op
    VMap(Log(), [0], 3)
    >>> result.parents == (values,)
    True

    Each lifted output carries the interface selected by the mapped function:

    >>> interface = ir.RVInterface()
    >>> special_exp = vmap(lambda x: ir.RV(ir.Exp(), x, interface=interface))
    >>> special_result = special_exp(values)
    >>> special_result.interface is interface
    True
    """

    @wraps(fun)
    def vmap_fun(*args: object) -> PyTree[ir.RV]:
        normalized_in_axes = normalize_in_axes(in_axes, len(args))
        argument_leaves, actual_rvs, rv_axes, rv_positions = flatten_rv_inputs(
            args, normalized_in_axes, "vmap"
        )

        abstract_rvs, resolved_axis_size = make_abstract_rvs(actual_rvs, rv_axes, axis_size)
        args_treedef = jax.tree_util.tree_structure(args)
        abstract_leaves = list(argument_leaves)
        for position, abstract_rv in zip(rv_positions, abstract_rvs, strict=True):
            abstract_leaves[position] = abstract_rv
        abstract_args = jax.tree_util.tree_unflatten(args_treedef, abstract_leaves)

        output, generated, _ = trace_call(fun, *abstract_args)
        output_leaves, output_treedef = tree_flatten_with_none_as_leaf(output)
        # trace_call has already validated every output leaf as an RV, but PyTree flattening erases
        # the leaf type from its return annotation.
        outputs = cast(list[ir.RV], output_leaves)
        if any(output_rv not in generated for output_rv in outputs):
            raise ValueError("every output must be created during the function call")

        real_outputs = lift_graph(
            abstract_rvs,
            rv_axes,
            generated,
            outputs,
            actual_rvs,
            resolved_axis_size,
            simplify=simplify,
        )
        return jax.tree_util.tree_unflatten(output_treedef, real_outputs)

    original_name = getattr(fun, "__name__", None)
    if not original_name or original_name == "<lambda>":
        original_name = "function"
    vmap_fun.__name__ = f"vmap_{original_name}"
    vmap_fun.__qualname__ = vmap_fun.__name__

    # The runtime implementation erases argument and output-tree types, but preserves both the
    # positional signature and the exact traced output PyTree structure.
    typed_vmap_fun = cast(Callable[[*Args], Output], vmap_fun)
    return typed_vmap_fun
