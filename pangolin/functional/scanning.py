"""Provide the `scan` functional combinator."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import cast

import jax

from pangolin import ir
from pangolin.util import PyTree

from .compositing import composite
from .simplifying import simplify_scan
from .tracing import AbstractOp, flatten_rv_inputs

__all__ = ["scan"]


def normalize_in_axes(in_axes: PyTree[int | None], num_args: int) -> PyTree[int | None]:
    """Normalize the top level of ``in_axes`` to match the additional arguments.

    A specification for one additional argument is wrapped in a tuple. For multiple arguments, a
    top-level list is converted to a tuple and every other structure is returned unchanged.

    Parameters
    ----------
    in_axes
        Axis specification for the arguments after the carry.
    num_args
        Number of additional positional arguments.

    Returns
    -------
    normalized_in_axes
        Axis specification whose top level represents all additional arguments.
    """
    if num_args == 1:
        return (in_axes,)
    if isinstance(in_axes, list):
        return tuple(in_axes)
    return in_axes


def make_abstract_rvs(
    actual_rvs: list[ir.RV],
    in_axes: list[int | None],
    length: int,
) -> list[ir.RV]:
    """Create abstract RVs with scanned axes removed.

    Parameters
    ----------
    actual_rvs
        Flat list of actual RV arguments, excluding the carry.
    in_axes
        Scanned axis of each argument, or ``None`` for an unscanned argument.
    length
        Positive scan length required along every scanned axis.

    Returns
    -------
    abstract_rvs
        Abstract RVs carrying the interfaces of the corresponding ``actual_rvs``.

    Raises
    ------
    ValueError
        If ``length`` is not positive, the lists have different lengths, or a scanned axis has the
        wrong length.
    """
    if length <= 0:
        raise ValueError("length must be positive")

    abstract_rvs = []
    for input_number, (actual_rv, axis) in enumerate(zip(actual_rvs, in_axes, strict=True)):
        shape, mapped_length = ir.split_shape(actual_rv.shape, axis)
        abstract_rvs.append(ir.RV(AbstractOp(shape), interface=actual_rv.interface))

        if mapped_length is not None and mapped_length != length:
            raise ValueError(
                f"input {input_number} has mapped axis length {mapped_length}, expected {length}"
            )

    return abstract_rvs


def scan[Carry: ir.RV, *Args](
    fun: Callable[[Carry, *Args], ir.RV],
    length: int,
    in_axes: PyTree[int | None] = 0,
    *,
    simplify: bool = True,
) -> Callable[[Carry, *Args], ir.RV]:
    """A combinator that returns a function constructing a Scan RV from a step function.

    The step function has the conceptual signature ``fun(carry, *step_args) -> next_carry``. It
    must create and return exactly one RV that depends on ``carry`` and has the same shape. Calling
    the returned function constructs one RV representing all ``length`` successive carries.

    Pangolin calls ``fun`` once; it does not execute a Python loop. Each RV argument is replaced by
    an abstract RV carrying the same interface. A non-RV argument is permitted when its axis is
    ``None`` and is passed to ``fun`` unchanged. Non-RV values never become Scan parents. The
    traced step output must have the initial carry's interface, which the resulting Scan RV also
    carries.

    Parameters
    ----------
    fun
        Step function with the conceptual signature ``fun(carry, *step_args) -> next_carry``.
    length
        Positive number of scan steps.
    in_axes
        Axis specification for the arguments after the carry. An integer selects an axis of an RV;
        ``None`` reuses an RV or passes a non-RV value unchanged at every step. A PyTree may match
        or be a prefix of the additional argument structure. The carry is not included.
    simplify
        If true, replace full-range indexing in the step with direct scanned
        inputs and remove unused coordinates (the default).

    Returns
    -------
    scan_fun
        Function accepting an initial carry followed by additional RV PyTrees and returning one RV
        containing an `ir.Scan` Op.

    Raises
    ------
    TypeError
        If the initial carry is not an `ir.RV`, an integer axis is assigned to a non-RV argument,
        or ``fun`` does not return exactly one RV.
    ValueError
        If ``length`` is invalid, a scanned axis has the wrong length, the output does not depend
        on the carry, the output existed before the call, its shape or interface differs from the
        carry, or a contributing intermediate RV has a random Op.

    Examples
    --------
    The combinator returns a function that repeatedly exponentiates a scalar carry:

    >>> start = ir.RV(ir.Constant(1.0))
    >>> def exp_step(carry):
    ...     return ir.RV(ir.Exp(), carry)
    >>> repeated_exp = scan(exp_step, length=3)
    >>> repeated_exp  # doctest: +ELLIPSIS
    <function scan_exp_step at 0x...>
    >>> result = repeated_exp(start)
    >>> result.op
    Scan(Exp(), 3, [])
    >>> result.parents == (start,)
    True

    Scan an additional argument along axis zero:

    >>> increments = ir.RV(ir.Constant([1.0, 2.0, 3.0]))
    >>> add_steps = scan(lambda carry, x: ir.RV(ir.Add(), carry, x), length=3)
    >>> result = add_steps(start, increments)
    >>> result.op
    Scan(Add(), 3, [0])
    >>> result.parents == (start, increments)
    True

    A non-RV configuration argument can be passed unchanged with ``in_axes=None``:

    >>> def configured_step(carry, config):
    ...     op = ir.Exp() if config["exponential"] else ir.Identity()
    ...     return ir.RV(op, carry)
    >>> configured = scan(configured_step, length=2, in_axes=None)
    >>> result = configured(start, {"exponential": True})
    >>> result.op
    Scan(Exp(), 2, [])
    >>> result.parents == (start,)
    True

    The Scan RV carries the initial carry's interface:

    >>> interface = ir.RVInterface()
    >>> special_start = ir.RV(ir.Constant(1.0), interface=interface)
    >>> special_scan = scan(lambda carry: ir.RV(ir.Exp(), carry, interface=interface), length=3)
    >>> special_result = special_scan(special_start)
    >>> special_result.interface is interface
    True
    """

    @wraps(fun)
    def scan_fun(init: Carry, *args: *Args) -> ir.RV:
        if not isinstance(init, ir.RV):
            raise TypeError(f"scan initial carry must be an RV; got {type(init).__name__}")

        normalized_in_axes = normalize_in_axes(in_axes, len(args))
        argument_leaves, actual_rvs, rv_axes, rv_positions = flatten_rv_inputs(
            args, normalized_in_axes, "scan"
        )

        abstract_rvs = make_abstract_rvs(actual_rvs, rv_axes, length)
        abstract_carry = ir.RV(AbstractOp(init.shape), interface=init.interface)

        args_treedef = jax.tree_util.tree_structure(args)
        abstract_leaves = list(argument_leaves)
        for position, abstract_rv in zip(rv_positions, abstract_rvs, strict=True):
            abstract_leaves[position] = abstract_rv
        abstract_args = jax.tree_util.tree_unflatten(args_treedef, abstract_leaves)
        # The symbolic values preserve the interfaces and PyTree structure of the original
        # arguments, while tree reconstruction erases the argument structure from its type.
        abstract_call_args = cast(tuple[Carry, *Args], (abstract_carry, *abstract_args))
        base_rv = composite(fun)(*abstract_call_args)
        if base_rv.interface is not abstract_carry.interface:
            raise ValueError(
                "scan function output interface must match the initial carry interface"
            )
        boundary = base_rv.parents
        if not boundary or boundary[0] is not abstract_carry:
            raise ValueError("scan function output must depend on its carry")

        input_replacements = dict(zip(abstract_rvs, actual_rvs, strict=True))
        input_axes = dict(zip(abstract_rvs, rv_axes, strict=True))
        parents = [init]
        axes = []
        for parent in boundary[1:]:
            parents.append(input_replacements.get(parent, parent))
            axes.append(input_axes.get(parent))

        op = ir.Scan(base_rv.op, length, axes)
        final_parents = tuple(parents)
        if simplify:
            op, final_parents = simplify_scan(op, final_parents)
        return ir.RV(op, *final_parents, interface=base_rv.interface)

    original_name = getattr(fun, "__name__", None)
    if not original_name or original_name == "<lambda>":
        original_name = "function"
    scan_fun.__name__ = f"scan_{original_name}"
    scan_fun.__qualname__ = scan_fun.__name__

    return scan_fun
