import pytest
from pangolin.ir.index import Index
from pangolin.ir import RV, Constant
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

