"""Check orthogonal indexing structure and explicit interface ownership."""

import numpy as np
import pytest

from pangolin import functional, ir
from pangolin.functional.broadcasting import remove_leading_singleton


def constant(value, interface=ir.NO_INTERFACE):
    return ir.RV(ir.Constant(value), interface=interface)


def test_scalar_indices_need_no_vmap():
    x = constant([[1, 2], [3, 4]])
    row, column = constant(1), constant(0)
    result = functional.index_orthogonal(x, row, column)
    assert result.op == ir.Index()
    assert result.parents == (x, row, column)
    assert result.shape == ()


def test_scalar_array_needs_no_indices():
    x = constant(3)
    result = functional.index_orthogonal(x)
    assert result.op == ir.Index()
    assert result.parents == (x,)
    assert result.shape == ()


def test_multidimensional_indices_contribute_independent_axes():
    x = constant(np.arange(60).reshape(3, 4, 5))
    rows, middle, columns = constant([[2, 0, 1], [0, 2, 0]]), constant(2), constant([4, 1, 0, 3])
    result = functional.index_orthogonal(x, rows, middle, columns)
    assert result.op == ir.VMap(
        ir.VMap(
            ir.VMap(ir.Index(), (None, None, None, 0), 4),
            (None, 0, None, None),
            3,
        ),
        (None, 0, None, None),
        2,
    )
    assert result.parents == (x, rows, middle, columns)
    assert result.shape == (2, 3, 4)


def test_repeated_index_rv_still_selects_independently():
    x = constant([[1, 2], [3, 4]])
    index = constant([1, 0])
    result = functional.index_orthogonal(x, index, index)
    assert result.op == ir.VMap(ir.VMap(ir.Index(), (None, None, 0), 2), (None, 0, None), 2)
    assert result.parents == (x, index, index)
    assert result.shape == (2, 2)


def test_interface_is_explicit_not_inferred_from_any_parent():
    array_interface, index_interface, output_interface = (ir.RVInterface() for _ in range(3))
    x = constant([1, 2, 3], array_interface)
    index = constant([2, 0], index_interface)
    default = functional.index_orthogonal(x, index)
    explicit = functional.index_orthogonal(x, index, interface=output_interface)
    assert default.interface is ir.NO_INTERFACE
    assert explicit.interface is output_interface
    assert default.parents == explicit.parents == (x, index)
    assert x.interface is array_interface
    assert index.interface is index_interface


@pytest.mark.parametrize("count", [0, 1, 3])
def test_index_count_must_match_rank(count):
    with pytest.raises(ValueError, match="number of indices"):
        functional.index_orthogonal(
            constant([[1, 2], [3, 4]]), *(constant(0) for _ in range(count))
        )


@pytest.mark.parametrize("index", [0, [0, 1], np.array([0, 1]), slice(None), ...])
def test_indices_are_not_coerced(index):
    with pytest.raises(TypeError, match="requires RV arguments"):
        functional.index_orthogonal(constant([1, 2]), index)


def test_array_is_not_coerced():
    with pytest.raises(TypeError, match="requires RV arguments"):
        functional.index_orthogonal([1, 2], constant(0))


def test_remove_leading_singleton_maps_remaining_dimensions():
    interface = ir.RVInterface()
    x = constant(np.arange(6).reshape(1, 2, 3), interface)
    result = remove_leading_singleton(x)
    assert result.op == ir.VMap(
        ir.VMap(ir.Index(), (None, None, None, 0), 3), (None, None, 0, None), 2
    )
    assert result.shape == (2, 3)
    assert result.parents[0] is x
    assert result.interface is interface
    assert all(parent.interface is interface for parent in result.parents)
