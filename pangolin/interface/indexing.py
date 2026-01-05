"""Indexing
Interface for fully-orthogonal indexing.
"""

from __future__ import annotations

from jax._src.core import pp_aval
from . import InfixRV
from pangolin.ir import Op, VMap, Constant, print_upstream, Shape
from pangolin.ir import SimpleIndex
from pangolin import dag, ir, util
from collections.abc import Callable
from .base import makerv, create_rv, RVLike, constant, exp, log, config, Broadcasting, get_shape
from . import base
from typing import Sequence, Type, cast
import jax.tree_util

from typing import Protocol, TypeVar, Any
from numpy.typing import ArrayLike
import numpy as np
from jax import numpy as jnp
import types

# TODO: Clarify exactly how indexing works, how related to broadcasting

_IdxType = RVLike | slice | types.EllipsisType

# _IdxTypeNoEllipsis = RVLike | slice


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


# TODO: var should be RVLike
def index(var: InfixRV, *indices: _IdxType):
    """Index a RV.

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
    (2, 2) | d = simple_index(a,b,c)
    >>> C = index(A, 0, ...)
    >>> print_upstream(C)
    shape  | statement
    ------ | ---------
    (2, 3) | a = [[3 0 2] [4 4 4]]
    ()     | b = 0
    (3,)   | c = [0 1 2]
    (3,)   | d = simple_index(a,b,c)
    """

    indices = eliminate_ellipses(var.ndim, indices)

    # TODO: Allow ellipses
    if len(var.shape) != len(indices):
        raise ValueError(f"RV has {var.ndim} dims but was given {len(indices)} indices")

    rv_indices = convert_indices(var.shape, *indices)
    return InfixRV(SimpleIndex(), var, *rv_indices)


def vmap_scalar_index_simple(var_shape: ir.Shape, *index_shapes: ir.Shape) -> ir.VectorIndex | ir.VMap:
    """ """

    array_shape = None
    for shape in index_shapes:
        if shape == ():
            continue

        if array_shape is None:
            array_shape = shape
        else:
            if shape != array_shape:
                raise ValueError(f"Can't broadcast non-matching shapes {shape} and {array_shape}")

    if array_shape is None:
        return ir.VectorIndex()

    in_axes = (None,) + tuple(0 if shape == array_shape else None for shape in index_shapes)

    new_op = ir.VectorIndex()
    for size in reversed(array_shape):
        new_op = VMap(new_op, in_axes, size)

    assert new_op.get_shape(var_shape, *index_shapes) == array_shape, "Pangolin bug"

    return new_op


def vmap_scalar_index_numpy(var_shape: ir.Shape, *index_shapes: ir.Shape) -> ir.VectorIndex | ir.VMap:
    # will raise ValueError if not broadcastable
    array_shape = np.broadcast_shapes(*index_shapes)

    if array_shape is None:
        return ir.VectorIndex()

    new_op = ir.VectorIndex()
    for n, size in enumerate(reversed(array_shape)):
        in_axes = []
        for index_shape in index_shapes:
            if len(index_shape) < n + 1:
                my_axis = None
            else:
                my_axis = 0
            in_axes.append(my_axis)

        new_op = VMap(new_op, (None,) + tuple(in_axes), size)

    assert new_op.get_shape(var_shape, *index_shapes) == array_shape, "Pangolin bug"

    return new_op


# TODO: should merge this with functionality from base? (Especially when add new numpy functionality)
def vector_index(var: RVLike, *indices: _IdxType) -> InfixRV[ir.VectorIndex | ir.VMap]:
    var_shape = get_shape(var)
    var_ndim = len(var_shape)

    indices = eliminate_ellipses(var_ndim, indices)
    rv_indices = convert_indices(var_shape, *indices)

    if config.broadcasting == Broadcasting.OFF:
        op = ir.VectorIndex()
        return create_rv(op, var, *rv_indices)
    elif config.broadcasting == Broadcasting.SIMPLE:
        var_shape = get_shape(var)
        index_shapes = [get_shape(idx) for idx in rv_indices]
        vmapped_op = vmap_scalar_index_simple(var_shape, *index_shapes)
        return create_rv(vmapped_op, var, *rv_indices)
    elif config.broadcasting == Broadcasting.NUMPY:
        var_shape = get_shape(var)
        index_shapes = [get_shape(idx) for idx in rv_indices]
        vmapped_op = vmap_scalar_index_numpy(var_shape, *index_shapes)
        return create_rv(vmapped_op, var, *rv_indices)
    else:
        raise Exception(f"Unknown scalar broadcasting model: {config.broadcasting}")
