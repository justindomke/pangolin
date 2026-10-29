import pytest

import numpy as np

from pangolin.ir import MultiNormal, Normal, StudentT, VMap, split_shape


def test_eq_vmap1():
    d1 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    d2 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    assert d1 == d2


def test_eq_vmap2():
    d1 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    d2 = VMap(StudentT(), in_axes=(0, 1), axis_size=5)
    assert d1 != d2


def test_eq_vmap3():
    d1 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    d2 = VMap(Normal(), in_axes=(1, 0), axis_size=5)
    assert d1 != d2


def test_eq_vmap4():
    d1 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    d2 = VMap(Normal(), in_axes=(0, 1), axis_size=4)
    assert d1 != d2


def test_eq_vmap5():
    d1 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    d2 = VMap(d1, in_axes=(0, 1), axis_size=5)
    d3 = VMap(d1, in_axes=(0, 1), axis_size=5)
    assert d1 != d2
    assert d1 != d3
    assert d2 == d3


def test_eq_vmap6():
    d1 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    d2 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    d3 = VMap(d1, in_axes=(0, 1), axis_size=5)
    d4 = VMap(d2, in_axes=(0, 1), axis_size=5)
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d2 != d3
    assert d2 != d4
    assert d3 == d4


def test_eq_vmap7():
    d1 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    d2 = VMap(Normal(), in_axes=(0, 1), axis_size=5)
    d3 = VMap(d1, in_axes=(0, 1), axis_size=5)
    d4 = VMap(d2, in_axes=(0, 1), axis_size=4)
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d2 != d3
    assert d2 != d4
    assert d3 != d4


def test_eq_vmap8():
    d1 = VMap(Normal(), in_axes=(0, 1))
    d2 = VMap(Normal(), in_axes=(0, 1))
    d3 = VMap(d1, in_axes=(0, 1))
    d4 = VMap(d2, in_axes=(0, 1))
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d2 != d3
    assert d2 != d4
    assert d3 == d4


def test_VMapDist1():
    # couldn't call normal here because it's not a CondDist. But I guess that's fine because user isn't expected
    # to touch VMap directly
    diag_normal = VMap(Normal(), (0, 0), 3)
    assert diag_normal.get_shape((3,), (3,)) == (3,)


def test_VMapDist2():
    diag_normal = VMap(Normal(), (0, None), 3)
    assert diag_normal.get_shape((3,), ()) == (3,)


def test_VMapDist3():
    diag_normal = VMap(Normal(), (None, 0), 3)
    assert diag_normal.get_shape((), (3,)) == (3,)


def test_VMapDist4():
    diag_normal = VMap(Normal(), (None, None), 3)
    assert diag_normal.get_shape((), ()) == (3,)


def test_VMapDist5():
    diag_normal = VMap(Normal(), (None, 0), 3)

    with pytest.raises(ValueError, match="mapped axis size 4, expected 3"):
        diag_normal.get_shape((), (4,))


def test_double_VMapDist1():
    diag_normal = VMap(Normal(), (0, 0))
    matrix_normal = VMap(diag_normal, (0, 0))
    assert matrix_normal.get_shape((4, 3), (4, 3)) == (4, 3)


# def test_vmap_index1():
#     base_op = Index(None)
#     assert base_op.get_shape((10,), ()) == ()
#     op = VMap(base_op, (None, 0))
#     assert op.get_shape((10,), (5,)) == (5,)


# def test_vmap_index2():
#     base_op = Index(slice(None), None)
#     op = VMap(base_op, (None, 0))
#     assert op.get_shape((10, 4), (5,)) == (5, 10)  # vmap dim first


def test_repr():
    d = VMap(Normal(), (0, 0))
    assert repr(d) == "VMap(Normal(), [0, 0])"


def test_str():
    d = VMap(Normal(), (0, 0))
    assert str(d) == "vmap(normal, [0, 0])"


@pytest.mark.parametrize(
    ("axis", "expected_shape", "expected_size"),
    [
        (None, (2, 3, 4), None),
        (0, (3, 4), 2),
        (1, (2, 4), 3),
        (2, (2, 3), 4),
        (-1, (2, 3), 4),
        (-2, (2, 4), 3),
        (-3, (3, 4), 2),
    ],
)
def test_split_shape(axis, expected_shape, expected_size):
    assert split_shape((2, 3, 4), axis) == (expected_shape, expected_size)


@pytest.mark.parametrize("axis", [3, -4])
def test_split_shape_rejects_out_of_bounds_axis(axis):
    with pytest.raises(ValueError, match="axis .* is out of bounds"):
        split_shape((2, 3, 4), axis)


@pytest.mark.parametrize("axis", [True, 0.5, "0"])
def test_split_shape_requires_integer_or_none_axis(axis):
    with pytest.raises(TypeError, match="axis must be an integer or None"):
        split_shape((2, 3, 4), axis)


def test_vmap_accepts_negative_axes():
    op = VMap(MultiNormal(), (-2, -3))

    assert op.get_shape((5, 3), (5, 3, 3)) == (5, 3)


def test_vmap_validates_every_mapped_axis_size():
    op = VMap(Normal(), (0, 0))

    with pytest.raises(ValueError, match="input 1 has mapped axis size 4, expected 3"):
        op.get_shape((3,), (4,))


def test_vmap_validates_inferred_size_against_explicit_axis_size():
    op = VMap(Normal(), (0, None), axis_size=3)

    assert op.get_shape((3,), ()) == (3,)
    with pytest.raises(ValueError, match="input 0 has mapped axis size 4, expected 3"):
        op.get_shape((4,), ())


def test_vmap_allows_unmapped_inputs_with_explicit_axis_size():
    op = VMap(Normal(), (None, None), axis_size=3)

    assert op.get_shape((), ()) == (3,)


@pytest.mark.parametrize("axis", [1, -2])
def test_vmap_rejects_out_of_bounds_mapped_axis(axis):
    op = VMap(Normal(), (axis, None))

    with pytest.raises(ValueError, match="axis .* is out of bounds"):
        op.get_shape((3,), ())


def test_vmap_rejects_mapping_over_scalar_input():
    op = VMap(Normal(), (0, None))

    with pytest.raises(ValueError, match="input with 0 dimensions"):
        op.get_shape((), ())


@pytest.mark.parametrize("parent_shapes", [((3,),), ((3,), (3,), (3,))])
def test_vmap_requires_one_axis_entry_per_input(parent_shapes):
    op = VMap(Normal(), (0, 0))

    with pytest.raises(ValueError, match="2 in_axes entries"):
        op.get_shape(*parent_shapes)


def test_vmap_requires_op():
    with pytest.raises(TypeError, match="base_op must be an Op"):
        VMap("normal", (0,))  # type: ignore[arg-type]


def test_vmap_requires_in_axes_sequence():
    with pytest.raises(TypeError, match="in_axes must be a sequence"):
        VMap(Normal(), None)  # type: ignore[arg-type]


@pytest.mark.parametrize("axis", [True, 0.5, "0"])
def test_vmap_requires_integer_or_none_axes(axis):
    with pytest.raises(TypeError, match="each in_axes entry must be an integer or None"):
        VMap(Normal(), (axis, None))


def test_vmap_requires_axis_size_when_all_inputs_are_unmapped():
    with pytest.raises(ValueError, match="axis_size is required when no inputs are mapped"):
        VMap(Normal(), (None, None))


@pytest.mark.parametrize("axis_size", [True, 1.5, "3"])
def test_vmap_requires_integer_axis_size(axis_size):
    with pytest.raises(TypeError, match="axis_size must be an integer or None"):
        VMap(Normal(), (None, None), axis_size)


@pytest.mark.parametrize("axis_size", [0, -1])
def test_vmap_requires_positive_axis_size(axis_size):
    with pytest.raises(ValueError, match="axis_size must be positive"):
        VMap(Normal(), (None, None), axis_size)


def test_vmap_accepts_numpy_integer_arguments():
    op = VMap(Normal(), (np.int64(0), None), np.int64(3))

    assert op.in_axes == (0, None)
    assert op.axis_size == 3
    assert op.get_shape((3,), ()) == (3,)


def test_vmap_rejects_zero_inferred_axis_size():
    op = VMap(Normal(), (0, None))

    with pytest.raises(ValueError, match="mapped axis size must be positive"):
        op.get_shape((0,), ())
