import numpy as np
import pytest

from pangolin.ir import Constant, Index, RV, VMap


def test_get_shape():
    op = Index()

    assert op.get_shape((2, 3), (), ()) == ()
    assert op.get_shape((2,), ()) == ()
    assert op.get_shape(()) == ()

    with pytest.raises(ValueError):
        op.get_shape((2, 3))
    with pytest.raises(ValueError):
        op.get_shape((2, 3), ())
    with pytest.raises(ValueError):
        op.get_shape((2, 3), (), (5, 1), ())


@pytest.mark.parametrize("index_shape", [(1,), (3,), (2, 3)])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rejects_nonscalar_indices(index_shape, axis):
    shapes = [(), (), ()]
    shapes[axis] = index_shape
    with pytest.raises(ValueError, match="scalar indices"):
        Index().get_shape((3, 4, 5), *shapes)


def test_rv_construction_rejects_array_index():
    array = RV(Constant([1, 2, 3]))
    indices = RV(Constant([0, 2]))
    with pytest.raises(ValueError, match="scalar indices"):
        RV(Index(), array, indices)

    result = RV(VMap(Index(), (None, 0), 2), array, indices)
    assert result.shape == (2,)
    assert result.parents == (array, indices)


def test_vmap_must_map_every_nonscalar_index():
    array = RV(Constant([[1, 2], [3, 4]]))
    rows, columns = RV(Constant([0, 1])), RV(Constant([1, 0]))
    with pytest.raises(ValueError, match="scalar indices"):
        RV(VMap(Index(), (None, None, 0), 2), array, rows, columns)

    paired = RV(VMap(Index(), (None, 0, 0), 2), array, rows, columns)
    assert paired.shape == (2,)
    orthogonal = RV(
        VMap(VMap(Index(), (None, None, 0), 2), (None, 0, None), 2), array, rows, columns
    )
    assert orthogonal.shape == (2, 2)


@pytest.mark.parametrize(
    ("array_shape", "indices"),
    [
        ((), ()),
        ((5,), (0,)),
        ((5,), (np.asarray(2),)),
        ((5, 7), (0, 1)),
        ((5, 7, 3), (2, 1, 0)),
    ],
)
def test_get_shape_matches_scalar_indexing(array_shape, indices):
    array = np.empty(array_shape)
    expected_shape = array[indices].shape
    index_shapes = tuple(np.asarray(index).shape for index in indices)

    assert Index().get_shape(array_shape, *index_shapes) == expected_shape
