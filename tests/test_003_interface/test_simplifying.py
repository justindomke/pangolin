from pangolin.interface import InfixRV
from pangolin.ir import Constant, Index, Op, VMap
from pangolin.interface.simplifying import permutation_view, canonical_permutation_view
from pangolin import interface as pi
import numpy as np


RV = InfixRV


def assert_permutation(x, base, axes):
    view = permutation_view(x)

    assert view is not None

    found_base, found_axes = view
    assert found_base is base
    assert found_axes == axes
    assert x.shape == tuple(base.shape[axis] for axis in axes)


def make_permutation(
    base: RV,
    axes: tuple[int, ...],
) -> RV:
    indices = [RV(Constant(range(size))) for size in base.shape]

    op: Op = Index()

    # axes[0] must become the outermost VMap.
    for source_axis in reversed(axes):
        in_axes: list[int | None] = [None] * (1 + base.ndim)
        in_axes[source_axis + 1] = 0
        op = VMap(op, in_axes)

    return RV(op, base, *indices)


def test_identity():
    base = RV(Constant([[1, 2, 3], [4, 5, 6]]))
    x = make_permutation(base, (0, 1))

    assert permutation_view(x) == (base, (0, 1))


def test_transpose():
    base = RV(Constant([[1, 2, 3], [4, 5, 6]]))
    x = make_permutation(base, (1, 0))

    assert x.shape == (3, 2)
    assert permutation_view(x) == (base, (1, 0))


def test_three_dimensions():
    base = RV(Constant([[[0] * 4] * 3] * 2))
    x = make_permutation(base, (2, 0, 1))

    assert x.shape == (4, 2, 3)
    assert permutation_view(x) == (base, (2, 0, 1))


def test_rejects_non_range_index():
    base = RV(Constant([[1, 2, 3], [4, 5, 6]]))
    i = RV(Constant([1, 0]))
    j = RV(Constant(range(3)))

    op = VMap(
        VMap(Index(), [None, 0, None]),
        [None, None, 0],
    )
    x = RV(op, base, i, j)

    assert permutation_view(x) is None


def test_rejects_zipped_indices():
    base = RV(Constant([[1, 2], [3, 4]]))
    i = RV(Constant(range(2)))
    j = RV(Constant(range(2)))

    x = RV(
        VMap(Index(), [None, 0, 0]),
        base,
        i,
        j,
    )

    assert permutation_view(x) is None


def test_composes_permutations():
    base = RV(Constant([[[0] * 4] * 3] * 2))

    first = make_permutation(base, (2, 0, 1))
    second = make_permutation(first, (1, 2, 0))

    assert permutation_view(second) == (
        first,
        (1, 2, 0),
    )

    assert canonical_permutation_view(second) == (
        base,
        (0, 1, 2),
    )


# ---------------------------------------------------------------------------
# Plain indexing
# ---------------------------------------------------------------------------


def test_full_slices_are_identity():
    a = pi.constant([[1, 2, 3], [4, 5, 6]])

    x = a[:, :]

    assert_permutation(x, a, (0, 1))


def test_full_slices_of_3d_array_are_identity():
    a = pi.constant(np.arange(24).reshape(2, 3, 4))

    x = a[:, :, :]

    assert_permutation(x, a, (0, 1, 2))


# ---------------------------------------------------------------------------
# One VMap around Index
# ---------------------------------------------------------------------------


def test_vmap_over_rows_is_identity():
    """vmap(lambda i: a[i, :])(rows) == a."""

    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    rows = pi.constant(range(2))

    x = pi.vmap(lambda i: a[i, :])(rows)

    assert_permutation(x, a, (0, 1))


def test_vmap_over_columns_is_transpose():
    """vmap(lambda j: a[:, j])(columns) == a.T."""

    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    columns = pi.constant(range(3))

    x = pi.vmap(lambda j: a[:, j])(columns)

    assert_permutation(x, a, (1, 0))


def test_vmap_first_axis_of_3d_array_is_identity():
    """vmap(lambda i: a[i, :, :])(is_) has axes (0, 1, 2)."""

    a = pi.constant(np.arange(24).reshape(2, 3, 4))
    indices = pi.constant(range(2))

    x = pi.vmap(lambda i: a[i, :, :])(indices)

    assert_permutation(x, a, (0, 1, 2))


def test_vmap_middle_axis_of_3d_array():
    """vmap(lambda j: a[:, j, :])(js) has axes (1, 0, 2)."""

    a = pi.constant(np.arange(24).reshape(2, 3, 4))
    indices = pi.constant(range(3))

    x = pi.vmap(lambda j: a[:, j, :])(indices)

    assert_permutation(x, a, (1, 0, 2))


def test_vmap_last_axis_of_3d_array():
    """vmap(lambda k: a[:, :, k])(ks) has axes (2, 0, 1)."""

    a = pi.constant(np.arange(24).reshape(2, 3, 4))
    indices = pi.constant(range(4))

    x = pi.vmap(lambda k: a[:, :, k])(indices)

    assert_permutation(x, a, (2, 0, 1))


# ---------------------------------------------------------------------------
# Nested VMaps
# ---------------------------------------------------------------------------


def test_nested_vmaps_are_identity():
    """vmap(lambda i: vmap(lambda j: a[i, j])(js))(is_) == a."""

    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    rows = pi.constant(range(2))
    columns = pi.constant(range(3))

    x = pi.vmap(lambda i: pi.vmap(lambda j: a[i, j], optimize=False)(columns), optimize=False)(rows)

    assert_permutation(x, a, (0, 1))


def test_nested_vmaps_are_transpose():
    """vmap(lambda j: vmap(lambda i: a[i, j])(is_))(js) == a.T."""

    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    rows = pi.constant(range(2))
    columns = pi.constant(range(3))

    x = pi.vmap(lambda j: pi.vmap(lambda i: a[i, j], optimize=False)(rows), optimize=False)(columns)

    assert_permutation(x, a, (1, 0))


def test_nested_swapped_indices_are_transpose():
    """The same transpose expressed using a[j, i]."""

    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    rows = pi.constant(range(2))
    columns = pi.constant(range(3))

    x = pi.vmap(lambda i: pi.vmap(lambda j: a[j, i])(rows))(columns)

    assert_permutation(x, a, (1, 0))


def test_three_nested_vmaps():
    """Output coordinates are ordered k, i, j."""

    a = pi.constant(np.arange(24).reshape(2, 3, 4))
    rows = pi.constant(range(2))
    columns = pi.constant(range(3))
    depths = pi.constant(range(4))

    x = pi.vmap(lambda k: pi.vmap(lambda i: pi.vmap(lambda j: a[i, j, k])(columns))(rows))(depths)

    assert_permutation(x, a, (2, 0, 1))


# ---------------------------------------------------------------------------
# Paired indexing is not an axis permutation
# ---------------------------------------------------------------------------


def test_paired_indices_are_not_permutation():
    """vmap(lambda i, j: a[i, j]) extracts a diagonal."""

    a = pi.constant([[1, 2], [3, 4]])
    i = pi.constant(range(2))
    j = pi.constant(range(2))

    x = pi.vmap(lambda i, j: a[i, j])(i, j)

    assert x.shape == (2,)
    assert permutation_view(x) is None


def test_swapped_paired_indices_are_not_permutation():
    """Swapping paired indices does not turn them into Cartesian indexing."""

    a = pi.constant([[1, 2], [3, 4]])
    i = pi.constant(range(2))
    j = pi.constant(range(2))

    x = pi.vmap(lambda i, j: a[j, i])(i, j)

    assert x.shape == (2,)
    assert permutation_view(x) is None


def test_paired_indices_in_3d_are_not_permutation():
    """j and k share one output coordinate."""

    a = pi.constant(np.arange(18).reshape(2, 3, 3))
    j = pi.constant(range(3))
    k = pi.constant(range(3))

    x = pi.vmap(lambda j, k: a[:, j, k])(j, k)

    assert x.shape == (3, 2)
    assert permutation_view(x) is None


def test_same_index_used_twice_is_not_permutation():
    """vmap(lambda i: a[i, i]) is diagonal indexing."""

    a = pi.constant([[1, 2], [3, 4]])
    indices = pi.constant(range(2))

    x = pi.vmap(lambda i: a[i, i])(indices)

    assert x.shape == (2,)
    assert permutation_view(x) is None


# ---------------------------------------------------------------------------
# Correct shape but incorrect coordinate values
# ---------------------------------------------------------------------------


def test_reversed_axis_is_not_dimension_permutation():
    """This transposes and reverses elements along one axis."""

    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    reverse_columns = pi.constant([2, 1, 0])

    x = pi.vmap(lambda j: a[:, j])(reverse_columns)

    assert x.shape == (3, 2)
    assert permutation_view(x) is None


def test_repeated_indices_are_not_dimension_permutation():
    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    repeated_columns = pi.constant([0, 0, 2])

    x = pi.vmap(lambda j: a[:, j])(repeated_columns)

    assert x.shape == (3, 2)
    assert permutation_view(x) is None


def test_arbitrary_index_permutation_is_not_dimension_permutation():
    """Permuting elements within an axis is different from permuting axes."""

    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    shuffled_columns = pi.constant([1, 2, 0])

    x = pi.vmap(lambda j: a[:, j])(shuffled_columns)

    assert x.shape == (3, 2)
    assert permutation_view(x) is None


def test_fixed_coordinate_is_not_dimension_permutation():
    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    columns = pi.constant(range(3))

    x = pi.vmap(lambda j: a[0, j])(columns)

    assert x.shape == (3,)
    assert permutation_view(x) is None


# ---------------------------------------------------------------------------
# Cases shape analysis alone cannot distinguish
# ---------------------------------------------------------------------------


def test_equal_axis_lengths_still_detect_transpose():
    """Both identity and transpose have shape (2, 2)."""

    a = pi.constant([[1, 2], [3, 4]])
    rows = pi.constant(range(2))
    columns = pi.constant(range(2))

    x = pi.vmap(lambda j: pi.vmap(lambda i: a[i, j])(rows))(columns)

    assert x.shape == a.shape
    assert_permutation(x, a, (1, 0))


def test_singleton_axis_transpose():
    a = pi.constant([[1, 2, 3]])
    columns = pi.constant(range(3))

    x = pi.vmap(lambda j: a[:, j])(columns)

    assert_permutation(x, a, (1, 0))
