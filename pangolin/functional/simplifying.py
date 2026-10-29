"""Recognize canonical axis-permutation graphs and simplify scan inputs."""

from __future__ import annotations

from pangolin import dag, ir
from .compositing import composite_from_trace
from .tracing import AbstractOp

__all__: list[str] = []


def simplify_scan(op: ir.Scan, parents: tuple[ir.RV, ...]) -> tuple[ir.Scan, tuple[ir.RV, ...]]:
    """Replace scanning full index ranges with scanning the indexed arrays.

    ``op`` and ``parents`` must describe a valid Scan, with the initial carry
    first. Recognize Index operations (including vectorized Index operations)
    whose full-range coordinate inputs select successive slices of an existing
    array. Supply those slices directly as scan inputs and remove inputs and
    intermediate operations that are no longer used. Coordinates used elsewhere
    remain inputs. Never move carry-dependent computations or random draws out
    of the step.

    Preserve the initial carry, scan length, and output shape. Return the original
    arguments when no indexing can be removed. Currently only bodies represented
    by Composite are inspected; nested operations are not recursively rewritten.

    >>> start = ir.RV(ir.Constant(0))
    >>> indices = ir.RV(ir.Constant([0, 1]))
    >>> values = ir.RV(ir.Constant([10, 20]))
    >>> body = ir.Composite(3, [ir.Index(), ir.Add()], [[2, 1], [0, 3]])
    >>> op, parents = simplify_scan(ir.Scan(body, 2, [0, None]),
    ...                            (start, indices, values))
    >>> op
    Scan(Add(), 2, [0])
    >>> parents == (start, values)
    True
    """
    if not isinstance(op.base_op, ir.Composite):
        return op, parents
    op.get_shape(*(parent.shape for parent in parents))

    axes = (None, *op.in_axes)
    inputs = [
        ir.RV(AbstractOp(ir.split_shape(parent.shape, axis)[0]), interface=parent.interface)
        for parent, axis in zip(parents, axes, strict=True)
    ]
    # The carry changes on every step, so it must never be treated as a fixed input.
    known = dict(zip(inputs[1:], zip(parents[1:], op.in_axes, strict=True), strict=True))
    replacements = {(parent, axis): node for node, (parent, axis) in known.items()}
    nodes = list(inputs)
    changed = False

    for component, numbers in zip(op.base_op.ops, op.base_op.par_nums, strict=True):
        arguments = tuple(nodes[number] for number in numbers)
        node = ir.RV(component, *arguments)
        primitive = component
        while isinstance(primitive, ir.VMap):
            primitive = primitive.base_op
        if isinstance(primitive, ir.Index) and all(arg in known for arg in arguments):
            actual, mapped = zip(*(known[arg] for arg in arguments))
            if any(axis is not None for axis in mapped):
                view = ir.RV(ir.VMap(component, mapped, op.length), *actual)
                replacement = unpermuted_mapped_axis(view, is_output=False)
                if replacement is not None:
                    if replacement not in replacements:
                        source, axis = replacement
                        new_input = ir.RV(AbstractOp(ir.split_shape(source.shape, axis)[0]))
                        replacements[replacement] = new_input
                        inputs.append(new_input)
                        known[new_input] = replacement
                    node = replacements[replacement]
                    changed = True
        if isinstance(component, ir.Constant):
            known[node] = (node, None)
        nodes.append(node)

    if not changed:
        return op, parents

    output = nodes[-1]
    upstream = dag.upstream_nodes_flat([output])
    used = set(upstream)
    boundary = [inputs[0], *(node for node in inputs[1:] if node in used)]
    boundary_set = set(boundary)
    generated = [node for node in upstream if node not in boundary_set]
    body = composite_from_trace(output, generated, boundary, allow_existing_output=True)
    new_parents = (parents[0], *(known[node][0] for node in boundary[1:]))
    new_axes = tuple(known[node][1] for node in boundary[1:])
    return ir.Scan(body, op.length, new_axes), new_parents


def permutation_view(x: ir.RV) -> tuple[ir.RV, tuple[int, ...]] | None:
    """Recognize when a vmapped Index with constant parents amounts to a dimension permutation.

    If this returns ``(base, permutation)``, then ``permutation[j] == i`` means dimension ``j`` of
    ``x`` comes from dimension ``i`` of ``base``.

    Parameters
    ----------
    x
        Candidate RV to inspect.

    Returns
    -------
    view
        Base RV and output-to-input axis permutation, or ``None`` when ``x`` is not a canonical
        permutation view.

    Examples
    --------
    Mapping over complete columns creates a transpose view:

    >>> from pangolin import ir
    >>> matrix = ir.RV(ir.Constant([[1, 2, 3], [4, 5, 6]]))
    >>> rows = ir.RV(ir.Constant(range(2)))
    >>> columns = ir.RV(ir.Constant(range(3)))
    >>> column_view = ir.RV(
    ...     ir.VMap(ir.VMap(ir.Index(), (None, 0, None), 2), (None, None, 0), 3),
    ...     matrix,
    ...     rows,
    ...     columns,
    ... )
    >>> column_view.shape
    (3, 2)
    >>> permutation_view(column_view) == (matrix, (1, 0))
    True
    """
    levels: list[ir.VMap] = []
    op = x.op

    while isinstance(op, ir.VMap):
        levels.append(op)
        op = op.base_op

    if not isinstance(op, ir.Index) or not x.parents:
        return None

    base, *indices = x.parents
    ndim = base.ndim

    if len(indices) != ndim or len(levels) != ndim:
        return None

    for index_number, index_rv in enumerate(indices):
        if index_rv.op != ir.Constant(range(base.shape[index_number])):
            return None

    num_parents = 1 + ndim
    permutation: list[int] = []

    for level in levels:
        if len(level.in_axes) != num_parents or level.in_axes[0] is not None:
            return None

        mapped_here = [
            index_number for index_number, axis in enumerate(level.in_axes[1:]) if axis is not None
        ]
        if len(mapped_here) != 1:
            return None

        index_number = mapped_here[0]
        if level.in_axes[index_number + 1] != 0:
            return None

        permutation.append(index_number)

    return base, tuple(permutation)


def canonical_permutation_view(x: ir.RV) -> tuple[ir.RV, tuple[int, ...]] | None:
    """Recognize a chain of permutation views and compose their dimension permutations.

    If this returns ``(base, permutation)``, then ``base`` precedes every view in the chain and
    ``permutation[j] == i`` means dimension ``j`` of ``x`` comes from dimension ``i`` of ``base``.

    Parameters
    ----------
    x
        Candidate RV to inspect.

    Returns
    -------
    view
        Earliest base RV and composed permutation, or ``None`` when ``x`` is not a recognized
        permutation view.

    Examples
    --------
    Two consecutive transpose views compose back to the original axis order:

    >>> from pangolin import ir
    >>> matrix = ir.RV(ir.Constant([[1, 2, 3], [4, 5, 6]]))
    >>> rows = ir.RV(ir.Constant(range(2)))
    >>> columns = ir.RV(ir.Constant(range(3)))
    >>> transposed = ir.RV(
    ...     ir.VMap(ir.VMap(ir.Index(), (None, 0, None), 2), (None, None, 0), 3),
    ...     matrix, rows, columns
    ... )
    >>> transposed_rows = ir.RV(ir.Constant(range(3)))
    >>> transposed_columns = ir.RV(ir.Constant(range(2)))
    >>> restored = ir.RV(
    ...     ir.VMap(ir.VMap(ir.Index(), (None, 0, None), 3), (None, None, 0), 2),
    ...     transposed,
    ...     transposed_rows,
    ...     transposed_columns,
    ... )
    >>> canonical_permutation_view(restored) == (matrix, (0, 1))
    True
    """
    view = permutation_view(x)
    if view is None:
        return None

    base, permutation = view
    while (base_view := permutation_view(base)) is not None:
        next_base, base_permutation = base_view
        permutation = tuple(base_permutation[axis] for axis in permutation)
        base = next_base
    return base, permutation


def unpermuted_mapped_axis(
    x: ir.RV,
    *,
    is_output: bool,
) -> tuple[ir.RV, int] | None:
    """Find a base RV that can replace a mapped axis-permutation view.

    Parameters
    ----------
    x
        Candidate permutation view.
    is_output
        Whether ``x`` is a mapped function output, whose mapped axis must remain at position zero.

    Returns
    -------
    replacement
        Base RV and its mapped axis, or ``None`` when eliminating the view would change its axis
        semantics.

    Examples
    --------
    An intermediate transpose can be replaced by its matrix while remembering that axis one is
    mapped. It cannot be replaced when it is itself an output, because outputs require the mapped
    axis in position zero:

    >>> from pangolin import ir
    >>> matrix = ir.RV(ir.Constant([[1, 2, 3], [4, 5, 6]]))
    >>> rows = ir.RV(ir.Constant(range(2)))
    >>> columns = ir.RV(ir.Constant(range(3)))
    >>> column_view = ir.RV(
    ...     ir.VMap(ir.VMap(ir.Index(), (None, 0, None), 2), (None, None, 0), 3),
    ...     matrix, rows, columns
    ... )
    >>> unpermuted_mapped_axis(column_view, is_output=False) == (matrix, 1)
    True
    >>> unpermuted_mapped_axis(column_view, is_output=True) is None
    True
    """
    view = permutation_view(x)
    if view is None:
        return None

    base, permutation = view
    if not permutation:
        return None

    mapped_axis = permutation[0]
    expected_permutation = (
        mapped_axis,
        *(axis for axis in range(base.ndim) if axis != mapped_axis),
    )
    if permutation != expected_permutation:
        return None
    if is_output and mapped_axis != 0:
        return None
    return base, mapped_axis
