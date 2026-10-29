from typing import Any

import numpy as np
import pytest

from pangolin import ir


@pytest.mark.parametrize(
    ("axis", "parent_shapes", "expected_shape"),
    [
        (0, ((2, 3), (4, 3)), (6, 3)),
        (1, ((2, 3), (2, 4), (2, 5)), (2, 12)),
        (-1, ((2, 3), (2, 4)), (2, 7)),
        (-2, ((2, 3), (4, 3)), (6, 3)),
    ],
)
def test_concatenate_shape(axis, parent_shapes, expected_shape):
    assert ir.Concatenate(axis).get_shape(*parent_shapes) == expected_shape


def test_concatenate_defaults_to_axis_zero():
    assert ir.Concatenate().axis == 0
    assert ir.Concatenate().get_shape((2, 3), (4, 3)) == (6, 3)


def test_concatenate_constructs_a_variadic_rv():
    first = ir.RV(ir.Constant([[1, 2]]))
    second = ir.RV(ir.Constant([[3, 4], [5, 6]]))

    result = ir.RV(ir.Concatenate(), first, second)

    assert result.shape == (3, 2)
    assert result.parents == (first, second)
    assert result.op.random is False


def test_concatenate_allows_one_parent():
    assert ir.Concatenate(axis=1).get_shape((2, 3)) == (2, 3)


@pytest.mark.parametrize("axis", [2, -3])
def test_concatenate_rejects_an_out_of_bounds_axis(axis):
    with pytest.raises(ValueError, match="axis .* is out of bounds"):
        ir.Concatenate(axis).get_shape((2, 3), (2, 4))


def test_concatenate_rejects_no_parents_or_scalars():
    with pytest.raises(ValueError, match="at least one parent"):
        ir.Concatenate().get_shape()
    with pytest.raises(ValueError, match="0 dimensions"):
        ir.Concatenate().get_shape((), ())


def test_concatenate_rejects_incompatible_parent_shapes():
    with pytest.raises(ValueError, match="same number of dimensions"):
        ir.Concatenate().get_shape((2, 3), (4,))
    with pytest.raises(ValueError, match="must match except along axis"):
        ir.Concatenate().get_shape((2, 3), (4, 5))


@pytest.mark.parametrize("axis", [True, None, 1.5, "1", np.array([0])])
def test_concatenate_requires_an_integer_axis(axis: Any):
    with pytest.raises(TypeError, match="axis for Concatenate must be an integer"):
        ir.Concatenate(axis)


def test_concatenate_has_structural_identity_and_readable_representations():
    assert ir.Concatenate(1) == ir.Concatenate(1)
    assert ir.Concatenate(1) != ir.Concatenate(0)
    assert hash(ir.Concatenate(1)) == hash(ir.Concatenate(1))
    assert repr(ir.Concatenate(1)) == "Concatenate(axis=1)"
    assert str(ir.Concatenate(1)) == "concatenate(axis=1)"
