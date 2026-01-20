"""Indexing
Interface for fully-orthogonal indexing.
"""

from __future__ import annotations

from jax._src.core import pp_aval
from pangolin.ir import Op, VMap, Constant, print_upstream, Shape
from pangolin.ir import Index
from pangolin import dag, ir, util
from collections.abc import Callable
from .base import makerv, create_rv, constant, exp, log, config, Broadcasting, get_shape
from . import base
from typing import cast, TYPE_CHECKING
import numpy as np
from jax import numpy as jnp
import types


from .base import InfixRV, RVLike

_IdxType = RVLike | slice | types.EllipsisType


# if TYPE_CHECKING: # needed?


def eliminate_ellipses(ndim: int, idx: tuple[_IdxType, ...]) -> tuple[slice | RVLike, ...]:
    """Given indices, if there is an ellipsis, insert full slices

    Examples
    --------
    >>> eliminate_ellipses(3, (0, 1, 2))
    (0, 1, 2)
    >>> eliminate_ellipses(4, (0, ..., 2))
    (0, slice(None, None, None), slice(None, None, None), 2)

    """

    num_ellipsis = len([i for i in idx if i is ...])
    if num_ellipsis == 0:
        if len(idx) != ndim:
            raise ValueError(f"Number of indices ({len(idx)}) does not match ndim ({ndim})")

    if num_ellipsis > 1:
        raise ValueError("an index can only have a single ellipsis ('...')")
    elif num_ellipsis == 1:
        where = idx.index(...)
        slices_needed = ndim - (len(idx) - 1)  # sub out ellipsis
        if where > 0:
            idx_start = idx[:where]
        else:
            idx_start = ()
        idx_mid = (slice(None),) * slices_needed
        idx_end = idx[where + 1 :]
        idx = idx_start + idx_mid + idx_end
    idx = cast(tuple[slice | RVLike, ...], idx)
    return idx


def convert_index(size: int, index: RVLike | slice) -> InfixRV:
    """Convert an index which could be RV or ArrayLike or slice to an RV

    Parameters
    ----------
    size: int
        size of axis for index
    index: RV_or_ArrayLike | slice
        Index which could be RV, a numpy array, something castable to a numpy array, or a slice.

    Examples
    --------
    If `input` is already an RV, does nothing. (Ignores `size`)
    >>> convert_index(10, constant(5))
    InfixRV(Constant(5))
    >>> convert_index(10, constant([0,1,2]))
    InfixRV(Constant([0,1,2]))

    If input is a numpy array or castable to a numpy array, create a new constant. (Ignores `size`)
    >>> convert_index(10, 5)
    InfixRV(Constant(5))
    >>> convert_index(10, [0,1,2])
    InfixRV(Constant([0,1,2]))

    If input is a slice, convert to a constant. (This is the only case where `size` is used.)
    >>> convert_index(5, slice(None))
    InfixRV(Constant([0,1,2,3,4]))
    >>> convert_index(10, slice(9,2,-2))
    InfixRV(Constant([9,7,5,3]))

    """

    if isinstance(index, InfixRV):
        return index
    elif isinstance(index, slice):
        # TODO: insist that elements of the slice are constants
        A = np.arange(size)[index]
        return constant(A)
    else:
        A = np.array(index)
        return constant(A)


def convert_indices(shape: Shape, *indices: RVLike | slice) -> tuple[InfixRV, ...]:
    """
    Examples
    --------
    >>> convert_indices((5,3), slice(None), slice(None))
    (InfixRV(Constant([0,1,2,3,4])), InfixRV(Constant([0,1,2])))
    >>> convert_indices((5,3), [4,0,1], [3,3])
    (InfixRV(Constant([4,0,1])), InfixRV(Constant([3,3])))
    """

    return tuple(convert_index(size, index) for size, index in zip(shape, indices, strict=True))


def index(var: RVLike, *indices: _IdxType) -> InfixRV[ir.Index]:
    """
    Index a RV Using fully-orthogonal indexing.

    Note that this function is (intentionally) much simpler than indexing in NumPy or JAX or PyTorch in that it performs *fully orthogonal indexing* and that *slices are treated exactly the same as 1D arrays*.

    (Similar to ``oindex`` in `NEP 21 <https://numpy.org/neps/nep-0021-advanced-indexing.html>`_ .)

    Args:
        var: The RV to be indexed
        indices: The indices into the RV. Unless there is an ellipsis, the number of indices must match the number of dimensions of ``var``.

    Returns:
        Random variable with shape equal to the shapes of all indices, concatenated.

    Examples
    --------
    >>> A = constant([[3,0,2],[4,4,4]])
    >>> B = index(A, slice(None), [2,2])
    >>> print_upstream(B)
    shape  | statement
    ------ | ---------
    (2, 3) | a = [[3 0 2] [4 4 4]]
    (2,)   | b = [0 1]
    (2,)   | c = [2 2]
    (2, 2) | d = index(a,b,c)
    >>> C = index(A, 0, ...)
    >>> print_upstream(C)
    shape  | statement
    ------ | ---------
    (2, 3) | a = [[3 0 2] [4 4 4]]
    ()     | b = 0
    (3,)   | c = [0 1 2]
    (3,)   | d = index(a,b,c)

    Technically, it's legal (although pointless) to index a 0-D array

    >>> A = constant(12.0)
    >>> B = index(A)
    >>> print_upstream(B)
    shape | statement
    ----- | ---------
    ()    | a = 12.
    ()    | b = index(a)
    """

    var = makerv(var)

    indices = eliminate_ellipses(var.ndim, indices)

    if len(var.shape) != len(indices):
        raise ValueError(f"RV has {var.ndim} dims but was given {len(indices)} indices")

    rv_indices = convert_indices(var.shape, *indices)
    return InfixRV(Index(), var, *rv_indices)


def vindex(var: RVLike, *indices: _IdxType) -> InfixRV[ir.Index] | InfixRV[ir.VMap[ir.Index]]:
    """
    Index a RV Using "fully-vectorized" indexing.

    This function treats ``var`` as a scalar function of the indices and then applys the normal scalar broadcasting rules (as configured by `config`). Note that this behavior is (intentionally) much simpler than indexing in NumPy or JAX or PyTorch in that it performs *fully orthogonal indexing* and that *slices are treated exactly the same as 1D arrays*.

    (Named in honor of ``vindex`` in `NEP 21 <https://numpy.org/neps/nep-0021-advanced-indexing.html>`_ .)

    Args:
        var: The RV to be indexed
        indices: The indices into the RV. Unless there is an ellipsis, the number of indices must match the number of dimensions of ``var``.

    Returns:
        Random variable with shape equal to the shapes of all indices, concatenated.


    Examples
    --------
    Index a 2D array with two matching 2D arrays

    >>> A = constant([[0.,0,0],[1,1,1],[2,2,2]])
    >>> B = constant([[0,1,2],[2,1,0]])
    >>> C = constant([[2,1,1],[0,0,0]])
    >>> D = vindex(A, B, C)
    >>> print_upstream(D)
    shape  | statement
    ------ | ---------
    (3, 3) | a = [[0. 0. 0.] [1. 1. 1.] [2. 2. 2.]]
    (2, 3) | b = [[0 1 2] [2 1 0]]
    (2, 3) | c = [[2 1 1] [0 0 0]]
    (2, 3) | d = vmap(vmap(index, [None, 0, 0], 3), [None, 0, 0], 2)(a,b,c)

    With a square matrix, you can use slices to extract the diagonal

    >>> A = constant([[1,2,3],[4,5,6],[7,8,9]])
    >>> B = vindex(A,slice(None),slice(None))
    >>> print_upstream(B)
    shape  | statement
    ------ | ---------
    (3, 3) | a = [[1 2 3] [4 5 6] [7 8 9]]
    (3,)   | b = [0 1 2]
    (3,)   | c = [0 1 2]
    (3,)   | d = vmap(index, [None, 0, 0], 3)(a,b,c)

    You can also get the anti-diagonal

    >>> C = vindex(A,slice(None),slice(None,None,-1))
    >>> print_upstream(C)
    shape  | statement
    ------ | ---------
    (3, 3) | a = [[1 2 3] [4 5 6] [7 8 9]]
    (3,)   | b = [0 1 2]
    (3,)   | c = [2 1 0]
    (3,)   | d = vmap(index, [None, 0, 0], 3)(a,b,c)
    """

    var = makerv(var)

    indices = eliminate_ellipses(var.ndim, indices)

    if len(var.shape) != len(indices):
        raise ValueError(f"RV has {var.ndim} dims but was given {len(indices)} indices")

    rv_indices = convert_indices(var.shape, *indices)

    @base.broadcast
    def scalar_index(*indices: InfixRV):
        if any(rv_idx.shape != () for rv_idx in indices):
            raise ValueError("non-scalar index")
        if len(indices) != var.ndim:
            raise TypeError(f"Got {len(indices)} for RV with {var.ndim} dims")

        return InfixRV(Index(), var, *indices)

    out = scalar_index(*rv_indices)
    return out


class VectorIndexProxy:
    """
    Fascilitates using ``rv.vindex[a,b,c]`` notation.
    """

    def __init__(self, var: InfixRV):
        self.var = var

    def __getitem__(self, args):
        if isinstance(args, tuple):
            return vindex(self.var, *args)
        else:
            return vindex(self.var, args)
