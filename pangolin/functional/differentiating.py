"""Build derivative graphs from deterministic graph-building functions."""

from collections.abc import Callable
from typing import Any, Concatenate

import jax.tree_util
import numpy as np

from pangolin import ir
from pangolin.util import PyTree, tree_flatten_with_none_as_leaf

from .tracing import AbstractOp, graph_scan, require_rv_leaves, trace_call

__all__ = ["vjp", "grad"]


def reverse_pass(
    output: ir.RV,
    generated: list[ir.RV],
    boundary: list[ir.RV],
    inputs: dict[ir.RV, ir.RV],
    cotangent: ir.RV,
) -> list[ir.RV]:
    """Build and reconnect cotangents for abstract inputs in their insertion order."""
    if any(node.op.random for node in generated):
        raise ValueError("function must not create contributing random RVs")

    dependent = set(inputs)
    for node in generated:
        if any(parent in dependent for parent in node.parents):
            dependent.add(node)

    cotangents = {output: cotangent}
    for node in reversed(generated):
        if node not in cotangents:
            continue
        for index, parent in enumerate(node.parents):
            if parent not in dependent:
                continue
            contribution = ir.RV(
                ir.VJP(node.op, index),
                cotangents[node],
                *node.parents,
                interface=output.interface,
            )
            if parent in cotangents:
                add: ir.Op = ir.Add()
                for size in reversed(parent.shape):
                    add = ir.VMap(add, (0, 0), size)
                contribution = ir.RV(
                    add, cotangents[parent], contribution, interface=output.interface
                )
            cotangents[parent] = contribution

    gradients = [
        cotangents[abstract] if abstract in cotangents
        else ir.RV(ir.Constant(np.zeros(argument.shape)), interface=output.interface)
        for abstract, argument in inputs.items()
    ]

    def reconnect(node: ir.RV, parents: tuple[ir.RV, ...]) -> tuple[ir.RV, None]:
        if parents == node.parents:
            return node, None
        return ir.RV(node.op, *parents, interface=node.interface), None

    replacements = {node: node for node in boundary}
    replacements.update(inputs)
    replacements[cotangent] = cotangent
    results, _ = graph_scan(gradients, reconnect, boundary=replacements)
    return [
        result if result.interface is output.interface
        else ir.RV(ir.Identity(), result, interface=output.interface)
        for result in results
    ]


def vjp[**P](fun: Callable[P, ir.RV], argnum: int = 0) -> Callable[Concatenate[ir.RV, P], ir.RV]:
    """A combinator that returns a function building a vector-Jacobian product.

    ``vjp(fun, argnum)(cotangent, *args, **kwargs)`` differentiates with respect to one positional
    argument of ``fun``, which must be an RV. For real-valued inputs and outputs, the result is
    the gradient of the sum of the elementwise product of ``fun``'s output and ``cotangent``.
    The cotangent's own dependencies do not contribute to this differentiation.
    The function must return one RV and must not create
    contributing random nodes. Arguments and captures may themselves be random RVs: their
    upstream graphs are not differentiated. Other arguments and captures are independent of
    the selected argument for differentiation, even when they refer to the same RV.

    The function is traced once. Reverse traversal builds
    per-Op `ir.VJP` nodes and adds contributions where paths meet, without introducing a
    Composite. The resulting graph is connected back to the original arguments. Newly created
    nodes that do not contribute to the output are ignored.

    Parameters
    ----------
    fun
        Function returning one RV. Contributing newly created nodes must be deterministic.
        Returning the selected argument or another existing RV is allowed.
    argnum
        Nonnegative index of the positional RV argument to differentiate. Other arguments may
        have any Python or PyTree structure accepted by ``fun``; no inputs are coerced.

    Returns
    -------
    vjp_fun
        Function taking a positional-only cotangent RV followed by ``fun``'s arguments. The
        cotangent must have exactly the output's shape, without coercion or broadcasting.
        Returns one RV with the selected argument's shape.
        An unused argument gives zeros; an identity function gives the cotangent's value.

    Raises
    ------
    TypeError
        If ``argnum`` is not an integer, the cotangent or selected argument is not an RV,
        or the output is not a single RV.
    ValueError
        If ``argnum`` is negative or missing from ``fun``'s positional arguments, the cotangent
        has the wrong shape, or the contributing computation contains newly created random RVs.

    Notes
    -----
    RVs representing calculations in ``fun`` retain their interfaces. New VJP, accumulation,
    and zero nodes use the function output's interface. Existing inputs, captures, and the
    supplied cotangent retain their interfaces. The returned
    RV always has the function output's interface; an Identity node adapts it when necessary.

    Examples
    --------
    >>> def square(x):
    ...     return ir.RV(ir.Mul(), x, x)
    >>> derivative = vjp(square)
    >>> x = ir.RV(ir.Constant(3.0))
    >>> cotangent = ir.RV(ir.Constant(2.0))
    >>> result = derivative(cotangent, x)
    >>> result.op
    Add()
    >>> [parent.op for parent in result.parents]
    [VJP(Mul(), 0), VJP(Mul(), 1)]
    >>> result.parents[0].parents[1:] == (x, x)
    True
    """
    if isinstance(argnum, bool) or not isinstance(argnum, (int, np.integer)):
        raise TypeError("vjp argnum must be an integer")
    if argnum < 0:
        raise ValueError("vjp argnum must be nonnegative")

    def call(cotangent: ir.RV, /, *args: P.args, **kwargs: P.kwargs) -> ir.RV:
        if not isinstance(cotangent, ir.RV):
            raise TypeError("vjp cotangent must be an RV")
        if argnum >= len(args):
            raise ValueError("vjp argnum must index a supplied positional argument")
        argument = args[argnum]
        if not isinstance(argument, ir.RV):
            raise TypeError("vjp selected argument must be an RV")
        abstract = ir.RV(AbstractOp(argument.shape), interface=argument.interface)
        abstract_args = (*args[:argnum], abstract, *args[argnum + 1 :])
        output, generated, boundary = trace_call(fun, *abstract_args, **kwargs)
        if not isinstance(output, ir.RV):
            raise TypeError("vjp function must return exactly one RV")
        if cotangent.shape != output.shape:
            raise ValueError("vjp cotangent shape must match the function output shape")
        return reverse_pass(output, generated, boundary, {abstract: argument}, cotangent)[0]

    original_name = getattr(fun, "__name__", None)
    if not original_name or original_name == "<lambda>":
        original_name = "function"
    call.__name__ = f"vjp_{original_name}"
    call.__qualname__ = call.__name__
    return call


def grad[**P](fun: Callable[P, ir.RV], argnum: int = 0) -> Callable[P, Any]:
    """A combinator that returns a function building a scalar output's gradient.

    The selected positional argument must be an RV or a PyTree whose leaves are RVs. The
    function is traced once and must return one scalar RV. A single reverse traversal builds
    gradients for all selected leaves, without introducing a Composite.

    Each leaf occurrence is an independent differentiation input, even when the same RV appears
    in multiple positions. Reusing a leaf within the function accumulates its contributions.
    Other arguments and captures are independent of every selected leaf for differentiation,
    even when they refer to the same RV or depend on it in the existing graph.

    Parameters
    ----------
    fun
        Function returning one scalar RV, with no contributing newly created random nodes.
        Existing random inputs and captures are allowed; their upstream graphs are not
        differentiated.
    argnum
        Nonnegative index of the positional RV or RV-only PyTree argument to differentiate.
        Other arguments may have any Python or PyTree structure accepted by ``fun``; no inputs
        are coerced.

    Returns
    -------
    grad_fun
        Function accepting ``fun``'s arguments and returning the selected argument's PyTree
        structure, with one gradient RV of matching shape per leaf. An unused leaf gives zeros;
        returning a selected scalar leaf gives one for that leaf and zeros for the others.
        Empty trees return empty trees, but ``fun`` is still called and its output validated.
        A single RV argument still returns a single RV. Container types follow JAX's PyTree
        registration; dictionary keys and custom-node metadata are not differentiated.

    Raises
    ------
    TypeError
        If ``argnum`` is not an integer, any selected leaf is not an RV (including None),
        or the output is not a single RV.
    ValueError
        If ``argnum`` is negative or missing from the positional arguments, the output is not
        scalar, or the contributing computation contains newly created random RVs.

    Notes
    -----
    Each selected leaf's interface is preserved during tracing. New derivative nodes and every
    returned gradient RV use the function output's interface. Existing inputs and captures
    retain their interfaces. The scalar seed uses the default interface; an Identity node adapts
    it when necessary. Numeric leaves are not coerced into RVs.

    Static typing preserves ``fun``'s parameter signature but uses ``Any`` for the gradient's
    return type to avoid rejecting valid lambdas and compositions. The selected argument's
    RV-only restriction and the returned tree structure are enforced at runtime.

    Examples
    --------
    >>> def square(x):
    ...     return ir.RV(ir.Mul(), x, x)
    >>> derivative = grad(square)
    >>> result = derivative(ir.RV(ir.Constant(3.0)))
    >>> result.op
    Add()
    >>> [parent.op for parent in result.parents]
    [VJP(Mul(), 0), VJP(Mul(), 1)]

    The gradient has the same tree structure as the selected argument:

    >>> def product(pair):
    ...     return ir.RV(ir.Mul(), pair['x'], pair['y'][0])
    >>> x = ir.RV(ir.Constant(2.0))
    >>> y = ir.RV(ir.Constant(3.0))
    >>> result = grad(product)({'x': x, 'y': [y]})
    >>> result['x'].op, result['y'][0].op
    (VJP(Mul(), 0), VJP(Mul(), 1))
    >>> result['x'].parents[1:] == (x, y)
    True
    """

    if isinstance(argnum, bool) or not isinstance(argnum, (int, np.integer)):
        raise TypeError("grad argnum must be an integer")
    if argnum < 0:
        raise ValueError("grad argnum must be nonnegative")

    def call(*args: P.args, **kwargs: P.kwargs) -> PyTree[ir.RV]:
        if argnum >= len(args):
            raise ValueError("grad argnum must index a supplied positional argument")
        argument = args[argnum]
        leaves = require_rv_leaves(argument, "grad selected argument")
        _, treedef = tree_flatten_with_none_as_leaf(argument)
        abstracts = [ir.RV(AbstractOp(leaf.shape), interface=leaf.interface) for leaf in leaves]
        abstract_argument = jax.tree_util.tree_unflatten(treedef, abstracts)
        abstract_args = (*args[:argnum], abstract_argument, *args[argnum + 1 :])
        output, generated, boundary = trace_call(fun, *abstract_args, **kwargs)
        if not isinstance(output, ir.RV):
            raise TypeError("grad function must return exactly one RV")
        if output.shape != ():
            raise ValueError("grad function output must be scalar")
        inputs = dict(zip(abstracts, leaves, strict=True))
        gradients = reverse_pass(output, generated, boundary, inputs, ir.RV(ir.Constant(1.0)))
        return jax.tree_util.tree_unflatten(treedef, gradients)

    original_name = getattr(fun, "__name__", None)
    if not original_name or original_name == "<lambda>":
        original_name = "function"
    call.__name__ = f"grad_{original_name}"
    call.__qualname__ = call.__name__
    return call
