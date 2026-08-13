from .base import InfixRV as RV
from pangolin.ir import VMap, Index, Constant, Op
from pangolin import ir


def permutation_view(x: RV) -> tuple[RV, tuple[int, ...]] | None:
    """Recognize an axis permutation represented by VMap and Index.

    The recognized representation must have:

    * an ``Index`` leaf;
    * one index parent per dimension of the indexed RV;
    * every index equal to ``Constant(range(axis_size))``;
    * zero or more nested ``VMap`` wrappers;
    * every ``VMap`` leaving the indexed RV invariant;
    * every ``VMap`` mapping axis zero of exactly one index parent;
    * no index parent mapped more than once.

    Mapped indices produce leading output axes in outermost-to-innermost
    ``VMap`` order. Indices not mapped by a ``VMap`` remain vector inputs to
    ``Index`` and produce trailing output axes in index-parent order.

    This supports both forms:

        Index(a, range(n0), range(n1), ...)

    and a scalar-only Index lowered entirely through nested VMaps.

    The returned permutation follows NumPy's transpose convention:

        x.shape[k] == base.shape[permutation[k]]

    Parameters
    ----------
    x:
        RV to inspect.

    Returns
    -------
    tuple[RV, tuple[int, ...]] | None
        ``(base, permutation)`` when the canonical representation is
        recognized; otherwise ``None``.

    Examples
    --------
    For a matrix ``a``:

        vmap(lambda i: a[i, :])(range(rows))

    returns ``(a, (0, 1))``, while:

        vmap(lambda j: a[:, j])(range(columns))

    returns ``(a, (1, 0))``.

    Paired indexing such as:

        vmap(lambda i, j: a[i, j])(is_, js)

    is not recognized because one VMap maps multiple index parents.
    """

    # Peel VMaps in outermost-to-innermost order. Because VMap prepends
    # its output dimension, this is also output-axis order.
    levels: list[VMap] = []
    op: Op = x.op

    while isinstance(op, VMap):
        levels.append(op)
        op = op.base_op

    if not isinstance(op, Index):
        return None

    if not x.parents:
        return None

    base, *indices = x.parents
    ndim = base.ndim

    # Indexing a scalar with no indices is the zero-dimensional identity
    # permutation.
    if ndim == 0:
        if levels:
            return None

        if indices:
            return None

        if x.shape != base.shape:
            return None

        return base, ()

    # Index must have exactly one coordinate parent per base dimension.
    if len(indices) != ndim:
        return None

    # Each valid VMap consumes one previously unconsumed index dimension.
    if len(levels) > ndim:
        return None

    # Every coordinate must be the canonical identity coordinate for its
    # corresponding base dimension.
    for index_num, index_rv in enumerate(indices):
        expected_op = Constant(range(base.shape[index_num]))

        if index_rv.op != expected_op:
            return None

    num_parents = 1 + ndim
    mapped_indices: set[int] = set()
    permutation: list[int] = []

    for level in levels:
        # Every wrapper must describe the same base Index argument list.
        if len(level.in_axes) != num_parents:
            return None

        # The complete value being indexed must be shared across mapped
        # invocations. Mapping it would instead treat one of its axes as a
        # batch dimension.
        if level.in_axes[0] is not None:
            return None

        mapped_here = [index_num for index_num, axis in enumerate(level.in_axes[1:]) if axis is not None]

        # No mapped index means repetition. More than one mapped index means
        # paired/diagonal indexing. Neither represents one output axis
        # corresponding bijectively to one input axis.
        if len(mapped_here) != 1:
            return None

        index_num = mapped_here[0]
        mapped_axis = level.in_axes[index_num + 1]

        # Constant(range(n)) is one-dimensional, so its only valid mapped
        # axis is zero.
        if mapped_axis != 0:
            return None

        # Mapping the same coordinate twice cannot describe an axis
        # permutation.
        if index_num in mapped_indices:
            return None

        mapped_indices.add(index_num)

        # This VMap prepends an output axis taken from this base axis.
        permutation.append(index_num)

    # Current Pangolin Index can retain vector-valued indices at its leaf.
    # Their dimensions follow all VMap-produced dimensions and appear in
    # index-parent order.
    permutation.extend(index_num for index_num in range(ndim) if index_num not in mapped_indices)

    permutation_tuple = tuple(permutation)

    # This also confirms that the result has the same rank as the base and
    # that every result dimension has the expected size.
    expected_shape = tuple(base.shape[source_axis] for source_axis in permutation_tuple)

    if x.shape != expected_shape:
        return None

    return base, permutation_tuple


def canonical_permutation_view(
    x: RV,
) -> tuple[RV, tuple[int, ...]] | None:
    """Recognize and compose consecutive canonical permutation views.

    If:

        x = transpose(y, p)
        y = transpose(z, q)

    then:

        x = transpose(z, tuple(q[axis] for axis in p))
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
    x: RV,
    *,
    is_output: bool,
) -> tuple[RV, int] | None:
    """
    Return a base RV that can replace ``x`` without materializing an axis permutation.

    If ``x`` is a recognized permutation view of some ``base``, this function
    checks whether ``x`` was formed by moving exactly one axis of ``base`` to
    axis zero while leaving every other axis in its original relative order.

    When that condition holds, the function returns:

        (base, mapped_axis)

    where ``mapped_axis`` is the position in ``base`` of the axis appearing as
    axis zero in ``x``.

    If ``is_output`` is true, the replacement is allowed only when
    ``mapped_axis == 0``, because a vmapped output must retain its mapped
    dimension at axis zero.

    Returns ``None`` if ``x`` is not a recognized permutation view, if the
    remaining axes were reordered, or if replacing an output would move its
    mapped dimension away from axis zero.

    Parameters
    ----------
    x:
        Candidate permutation view.
    is_output:
        Whether this node is one of the outputs of the current ``vmap``.

    Returns
    -------
    tuple[RV, int] | None
        ``(base, mapped_axis)`` when the view can be eliminated. Otherwise
        ``None``.

    Examples
    --------
    Set up a three-dimensional base and its canonical coordinate indices:

    >>> base = RV(Constant(
    ...     [[[100 * i + 10 * j + k for k in range(4)]
    ...       for j in range(3)]
    ...      for i in range(2)]
    ... ))
    >>> i = RV(Constant(range(2)))
    >>> j = RV(Constant(range(3)))
    >>> k = RV(Constant(range(4)))

    Mapping the first index gives the identity axis order ``(0, 1, 2)``.
    It can be eliminated for both intermediate and output nodes:

    >>> x0 = RV(
    ...     VMap(
    ...         Index(),
    ...         [None, 0, None, None],
    ...     ),
    ...     base, i, j, k,
    ... )
    >>> permutation_view(x0) == (base, (0, 1, 2))
    True
    >>> replacement = unpermuted_mapped_axis(x0, is_output=False)
    >>> replacement[0] is base, replacement[1]
    (True, 0)
    >>> replacement = unpermuted_mapped_axis(x0, is_output=True)
    >>> replacement[0] is base, replacement[1]
    (True, 0)

    Mapping the second index moves base axis one to the front, producing
    permutation ``(1, 0, 2)``. An intermediate node can be replaced with
    ``base`` while recording that its mapped axis is now axis one:

    >>> x1 = RV(
    ...     VMap(
    ...         Index(),
    ...         [None, None, 0, None],
    ...     ),
    ...     base, i, j, k,
    ... )
    >>> permutation_view(x1) == (base, (1, 0, 2))
    True
    >>> replacement = unpermuted_mapped_axis(x1, is_output=False)
    >>> replacement[0] is base, replacement[1]
    (True, 1)

    The same replacement is not valid for an output because it would move
    the mapped output dimension away from axis zero:

    >>> unpermuted_mapped_axis(x1, is_output=True) is None
    True

    Mapping the last index similarly produces ``(2, 0, 1)`` and records
    mapped axis two:

    >>> x2 = RV(
    ...     VMap(
    ...         Index(),
    ...         [None, None, None, 0],
    ...     ),
    ...     base, i, j, k,
    ... )
    >>> permutation_view(x2) == (base, (2, 0, 1))
    True
    >>> replacement = unpermuted_mapped_axis(x2, is_output=False)
    >>> replacement[0] is base, replacement[1]
    (True, 2)
    >>> unpermuted_mapped_axis(x2, is_output=True) is None
    True

    A permutation that also changes the order of the non-mapped axes cannot
    be represented by merely changing the mapped-axis metadata. Here the
    outer VMap contributes base axis one, the inner VMap contributes base
    axis two, and the remaining vector Index contributes base axis zero,
    giving ``(1, 2, 0)``:

    >>> reordered = RV(
    ...     VMap(
    ...         VMap(
    ...             Index(),
    ...             [None, None, None, 0],
    ...         ),
    ...         [None, None, 0, None],
    ...     ),
    ...     base, i, j, k,
    ... )
    >>> permutation_view(reordered) == (base, (1, 2, 0))
    True
    >>> unpermuted_mapped_axis(reordered, is_output=False) is None
    True
    >>> unpermuted_mapped_axis(reordered, is_output=True) is None
    True

    Non-permutation operations are not eligible:

    >>> value = ir.RV(ir.Constant(1))
    >>> unpermuted_mapped_axis(value, is_output=False) is None
    True
    """

    view = permutation_view(x)

    if view is None:
        return None

    base, permutation = view

    # A newly vmapped value must have at least its mapped dimension.
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
