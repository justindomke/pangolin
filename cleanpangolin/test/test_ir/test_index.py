import pytest
from cleanpangolin.ir.index import Index, index
from cleanpangolin.ir.rv import makerv
import numpy as np

fslice = slice(None) # full slice

def test_index1():
    d = Index(None, None)
    assert d.get_shape((3, 2), (), ()) == ()


def test_index2():
    d = Index(None, None)
    assert d.get_shape((3, 2), (4,), (4,)) == (4,)


def test_index3():
    d = Index(slice(None), None)
    assert d.get_shape((3, 2), (4,)) == (3, 4)


def test_index4():
    d = Index(None, slice(None))
    assert d.get_shape((3, 2), (4,)) == (4, 2)


def test_index5():
    d = Index(None, slice(None))
    assert d.get_shape((3, 2), (4, 5, 7)) == (4, 5, 7, 2)


# test cases we don't (yet?) cover because don't support broadcasting on indices
# ((3, 4, 5), (0, np.ones((2, 6, 7), dtype=int), 0)),

@pytest.mark.parametrize(
    "start_shape,idx",
    [
        ((5,),(0,)),
        ((5,),(fslice,)),
        ((5,),(slice(2,4),)),
        ((5,),((2,3),)),
        ((5,),(np.ones(3,dtype=int),)),
        ((5,),(np.ones((3,2),dtype=int),)),
        ((5, 7), (0, 1)),
        ((5, 7), (slice(2,4), slice(2,4))),
        ((5, 7), (slice(2,4), slice(1,2))),
        ((3,4,5), (0, fslice, 1)),
    ],
)
def test_index_class(start_shape, idx):
    if len(start_shape) != len(idx):
        raise Exception("invalid test case! idx must have same length as array shape")

    print(f"{start_shape=}")
    print(f"{idx=}")

    x = np.random.randn(*start_shape)
    expected_shape = x[idx].shape
    print(f"{expected_shape=}")

    slices = [s if isinstance(s, slice) else None for s in idx]
    d = Index(*slices)
    non_slice_idx = [i for i in idx if not isinstance(i, slice)]
    non_slice_shapes = [np.array(i).shape for i in non_slice_idx]
    shape = d.get_shape(x.shape, *non_slice_shapes)
    print(f"{shape=}")
    assert expected_shape == shape

def dotest(x,*idx):
    y = index(makerv(x),*idx)
    z = x.__getitem__(idx)
    assert y.shape == z.shape

def test_1d_full_slice():
    x = np.random.randn(5)
    dotest(x,fslice)

def test_1d_partial_slice():
    x = np.random.randn(10)
    dotest(x, slice(2, 8, None))

def test_2d_single_full_slice():
    x = np.random.randn(5, 7)
    dotest(x,fslice)

def test_2d_single_partial_slice():
    x = np.random.randn(5, 7)
    dotest(x,slice(2,4,None))

def test_2d_double_partial_slice():
    x = np.random.randn(5, 7)
    dotest(x,slice(2,4,None),slice(1,2,None))

def test_1d_advanced_indexing():
    x = np.random.randn(10)
    dotest(x,[2,3])

def test_2d_single_advanced_indexing():
    x = np.random.randn(5, 7)
    dotest(x,[2,3])

def test_2d_double_advanced_indexing():
    x = np.random.randn(5,7)
    dotest(x,[2,3],[5,1])

def test_2d_scalar_indexing():
    x = np.random.randn(3, 4)
    dotest(x,2)

def test_2d_scalar_slice_indexing():
    x = np.random.randn(3, 4)
    dotest(x,2,fslice)

def test_2d_scalar_scalar_indexing():
    x = np.random.randn(3, 3)
    dotest(x,2,1)

def test_3d_single_advanced():
    x = np.random.randn(3, 4, 5)
    dotest(x,[2,1])

def test_3d_scalar_slice_scalar():
    x = np.random.randn(3, 4, 5)
    dotest(x,2,fslice,4)

def test_2d_slice_advanced():
    x = np.random.randn(5,6)
    dotest(x,fslice,[1,4,2])

def test_3d_slice_advanced():
    x = np.random.randn(5,6,7)
    dotest(x,fslice,[1,4,2])

def test_3d_advanced_advanced():
    x = np.random.randn(5, 6, 7)
    dotest(x,[2,1,2],[1,4,2])

def test_3d_slice_advanced_advanced():
    x = np.random.randn(5, 6, 7)
    dotest(x,fslice,[2,1,2],[1,4,2])


def test_3d_advanced_partslice_advanced():
    x = np.random.randn(6, 5, 7)
    dotest(x, [2,1,2], slice(1,4), [1,4,2])

def test_3d_advanced_partslice():
    x = np.random.randn(6, 5, 7)
    dotest(x, [2,1,2], slice(1,4))

def test_truly_advanced_indexing():
    x = np.random.randn(4,5,6,7)
    dotest(x, fslice, [2,1,2], fslice, [1,4,2])

def test_1d_array():
    x = np.random.randn(5)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x,idx)

def test_2d_slice_array():
    x = np.random.randn(5,6)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x,fslice,idx)

def test_2d_array_slice():
    x = np.random.randn(5,6)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x,idx,fslice)

def test_3d_array_slice_array():
    # triggers advanced indexing
    x = np.random.randn(5,6,7)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x,idx,fslice,idx)

def test_3d_slice_array_array():
    # triggers non-advanced indexing
    x = np.random.randn(5, 6, 7)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x, fslice, idx, idx)

def test_4d_slice_array_array_slice():
    x = np.random.randn(5, 6, 5, 4)
    idx = [[0, 2], [3, 4]]
    dotest(x,fslice,idx,idx,fslice)

def test_4d_array_slice_array_slice():
    x = np.random.randn(5, 6, 5, 4)
    idx = [[0, 2], [3, 4]]
    dotest(x,idx,fslice,idx,fslice)

# broadcasting for indices not (yet?) implemented
# def test_4d_array_scalar_array_slice():
#     x = np.random.randn(5, 6, 5, 4)
#     idx = [[0, 2], [3, 4]]
#     dotest(x,idx,fslice,2,fslice)


