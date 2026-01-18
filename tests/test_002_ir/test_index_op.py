import pytest
from pangolin.ir import Index, index_orthogonal
from pangolin.ir import RV, Constant, Normal
import numpy as np

fslice = slice(None)  # full slice


# def test_shape():
#     d = Index()
#     for reps in range(10):
#         dims = np.random.choice([1,2,3,4])
#         num_indices = np.random.choice([1,2,3,4])
#         index_shapes = []
#         for i in range(num_indices):
#             index_dims = np.random.choice([0,1,2])
#             index_shape = tuple(np.random.randint(5) for _ in range(index_dims))
#             index_shapes.append(index_shape)


def test_shape_no_slices():
    d = Index()

    assert d.get_shape((2, 3), (), ()) == ()
    assert d.get_shape((2, 3), (), (4, 5)) == (4, 5)
    assert d.get_shape((2, 3), (4, 5), ()) == (4, 5)
    assert d.get_shape((2, 3), (4, 5), (6, 7)) == (4, 5, 6, 7)

    with pytest.raises(ValueError):
        d.get_shape((2, 3))
    with pytest.raises(ValueError):
        d.get_shape((2, 3), ())
    with pytest.raises(ValueError):
        d.get_shape((2, 3), (), (5, 1), ())


# def test_shape_no_slices():
#     d = SimpleIndex(None, None)

#     assert d.get_shape((2, 3), (), ()) == ()
#     assert d.get_shape((2, 3), (), (4, 5)) == (4, 5)
#     assert d.get_shape((2, 3), (4, 5), ()) == (4, 5)
#     assert d.get_shape((2, 3), (4, 5), (6, 7)) == (4, 5, 6, 7)

#     with pytest.raises(ValueError):
#         d.get_shape((2, 3))
#     with pytest.raises(ValueError):
#         d.get_shape((2, 3), ())


# def test_shape_slice_first():
#     d = SimpleIndex(slice(None), None)

#     assert d.get_shape((2, 3), ()) == (2,)
#     assert d.get_shape((2, 3), (4, 5)) == (2, 4, 5)

#     with pytest.raises(ValueError):
#         d.get_shape((2, 3))
#     with pytest.raises(ValueError):
#         d.get_shape((2, 3), (), ())
#     with pytest.raises(ValueError):
#         d.get_shape((2, 3), (), (4, 5))
#     with pytest.raises(ValueError):
#         d.get_shape((2, 3), (4, 5), (6, 7))


# def test_shape_slice_second():
#     d = SimpleIndex(None, slice(None))

#     assert d.get_shape((2, 3), ()) == (3,)
#     assert d.get_shape((2, 3), (4, 5)) == (4, 5, 3)

#     with pytest.raises(ValueError):
#         d.get_shape((2, 3))
#     with pytest.raises(ValueError):
#         d.get_shape((2, 3), (), ())
#     with pytest.raises(ValueError):
#         d.get_shape((2, 3), (), (4, 5))
#     with pytest.raises(ValueError):
#         d.get_shape((2, 3), (4, 5), (6, 7))


# def test_3d_shape_no_slices():
#     d = SimpleIndex(None, None, None)

#     assert d.get_shape((2, 3, 4), (), (), ()) == ()
#     assert d.get_shape((2, 3, 4), (5, 6), (), ()) == (5, 6)
#     assert d.get_shape((2, 3, 4), (), (5, 6), ()) == (5, 6)
#     assert d.get_shape((2, 3, 4), (), (), (5, 6)) == (5, 6)
#     assert d.get_shape((2, 3, 4), (5, 6), (), (7, 8)) == (5, 6, 7, 8)
#     assert d.get_shape((2, 3, 4), (5,), (6, 7), (8, 9, 10)) == (5, 6, 7, 8, 9, 10)


# def test_3d_shape_slice_middle():
#     d = SimpleIndex(None, fslice, None)

#     assert d.get_shape((2, 3, 4), (), ()) == (3,)
#     assert d.get_shape((2, 3, 4), (5, 6), ()) == (5, 6, 3)
#     assert d.get_shape((2, 3, 4), (), (5, 6)) == (3, 5, 6)
#     assert d.get_shape((2, 3, 4), (5, 6), (7,)) == (5, 6, 3, 7)
#     assert d.get_shape((2, 3, 4), (5, 6), (7, 8, 9)) == (5, 6, 3, 7, 8, 9)


def test_index_orthogonal():
    A = np.random.randn(2, 3, 4)
    B = [0, 1, 0, 1, 1]
    C = [[0, 1], [0, 1], [0, 1]]
    assert index_orthogonal(A, 0, 1, 2).shape == ()
    assert index_orthogonal(A, B, 1, 2).shape == (5,)
    assert index_orthogonal(A, 0, B, 2).shape == (5,)
    assert index_orthogonal(A, 0, 1, B).shape == (5,)
    assert index_orthogonal(A, C, 1, 2).shape == (3, 2)
    assert index_orthogonal(A, 0, C, 2).shape == (3, 2)
    assert index_orthogonal(A, 0, 1, C).shape == (3, 2)
    assert index_orthogonal(A, B, C, 2).shape == (5, 3, 2)
    assert index_orthogonal(A, B, 1, C).shape == (5, 3, 2)
    assert index_orthogonal(A, C, B, 2).shape == (3, 2, 5)
    assert index_orthogonal(A, C, 1, B).shape == (3, 2, 5)
    assert index_orthogonal(A, C, C, 2).shape == (3, 2, 3, 2)
    assert index_orthogonal(A, C, 1, C).shape == (3, 2, 3, 2)

    assert index_orthogonal(A, fslice, 1, 2).shape == (2,)
    assert index_orthogonal(A, B, 1, fslice).shape == (5, 4)
    assert index_orthogonal(A, 0, B, fslice).shape == (5, 4)
    assert index_orthogonal(A, fslice, 1, B).shape == (
        2,
        5,
    )
    assert index_orthogonal(A, C, fslice, 2).shape == (3, 2, 3)
    assert index_orthogonal(A, 0, C, fslice).shape == (3, 2, 4)
    assert index_orthogonal(A, fslice, 1, C).shape == (2, 3, 2)
    assert index_orthogonal(A, B, C, fslice).shape == (5, 3, 2, 4)
    assert index_orthogonal(A, B, fslice, C).shape == (5, 3, 3, 2)
    assert index_orthogonal(A, C, B, fslice).shape == (3, 2, 5, 4)
    assert index_orthogonal(A, C, fslice, B).shape == (3, 2, 3, 5)
    assert index_orthogonal(A, C, C, fslice).shape == (3, 2, 3, 2, 4)
    assert index_orthogonal(A, C, fslice, C).shape == (3, 2, 3, 3, 2)


# automate testing to match numpy functionality


@pytest.mark.parametrize(
    "start_shape,idx",
    [
        ((), ()),
        ((5,), (0,)),
        # ((5,), (fslice,)),
        # ((5,), (slice(2, 4),)),
        ((5,), ((2, 3),)),
        ((5,), (np.ones(3, dtype=int),)),
        ((5,), (np.ones((3, 2), dtype=int),)),
        ((5, 7), (0, 1)),
        ((5, 7), (np.array([2, 0, 1]), 1)),
        # ((5, 7), (np.array([2, 0, 1]), fslice)),
        # ((5, 7), (np.array([2, 0, 1]), slice(2, 4, 2))),
        # ((5, 7), (slice(2, 4), slice(2, 4))),
        # ((5, 7), (slice(2, 4), slice(1, 2))),
        # ((3, 4, 5), (0, fslice, 1)),
        # ((3, 4, 5), ([0, 1, 2], fslice, 1)),
        # ((3, 4, 5), (0, fslice, [0, 1, 2])),
    ],
)
def test_Index_class(start_shape, idx):
    if len(start_shape) != len(idx):
        raise Exception("invalid test case! idx must have same length as array shape")

    x = np.array(np.random.randn(*start_shape))
    # expected_shape = x[*idx].shape
    expected_shape = index_orthogonal(x, *idx).shape
    print(f"{x.shape=}")
    print(f"{expected_shape=}")

    slices = [s if isinstance(s, slice) else None for s in idx]
    non_slice_idx = [i for i in idx if not isinstance(i, slice)]
    d = Index()
    non_slice_shapes = [np.array(i).shape for i in non_slice_idx]
    shape = d.get_shape(x.shape, *non_slice_shapes)
    print(f"{shape=}")

    assert expected_shape == shape
