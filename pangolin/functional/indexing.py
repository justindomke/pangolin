"""Build orthogonal indexing from scalar Index and VMap operations."""

from pangolin import ir

__all__ = ["index_orthogonal"]


def index_orthogonal(
    x: ir.RV,
    *indices: ir.RV,
    interface: ir.RVInterface = ir.NO_INTERFACE,
) -> ir.RV:
    """Index an RV with independent indices for each axis.

    Supply exactly one integer-valued index RV per dimension of ``x``. Each index contributes
    all of its dimensions independently, so the output shape is the concatenation of the index
    shapes. This constructs scalar `ir.Index` wrapped in one `ir.VMap` per index dimension.

    Parameters
    ----------
    x
        RV to index.
    *indices
        Index RVs in axis order. Scalars select a single entry along an axis. No coercion,
        slices, or ellipses are supported.
    interface
        Interface attached to the result. Defaults to `ir.NO_INTERFACE`, regardless of the
        interfaces of ``x`` and ``indices``. Existing parents are left unchanged.

    Returns
    -------
    indexed
        RV containing the selected entries.

    Examples
    --------
    >>> from pangolin import ir
    >>> x = ir.RV(ir.Constant([[10, 20, 30], [40, 50, 60]]))
    >>> rows = ir.RV(ir.Constant([1, 0]))
    >>> column = ir.RV(ir.Constant(2))
    >>> y = index_orthogonal(x, rows, column)
    >>> y.op
    VMap(Index(), [None, 0, None], 2)
    >>> y.shape
    (2,)

    Array indices are independent, not paired:

    >>> columns = ir.RV(ir.Constant([2, 0, 1]))
    >>> y = index_orthogonal(x, rows, columns)
    >>> y.op
    VMap(VMap(Index(), [None, None, 0], 3), [None, 0, None], 2)
    >>> y.shape
    (2, 3)
    """
    if not all(isinstance(value, ir.RV) for value in (x, *indices)):
        raise TypeError("index_orthogonal requires RV arguments")
    if len(indices) != x.ndim:
        raise ValueError(f"number of indices ({len(indices)}) does not match ndim ({x.ndim})")
    op: ir.Op = ir.Index()
    for i in reversed(range(len(indices))):
        in_axes = (None,) * (i + 1) + (0,) + (None,) * (len(indices) - i - 1)
        for size in reversed(indices[i].shape):
            op = ir.VMap(op, in_axes, size)
    return ir.RV(op, x, *indices, interface=interface)
