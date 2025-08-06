from pangolin.ir import RV, Constant, Index
from pangolin.interface import OperatorRV, makerv
from pangolin.interface.index import index, simplify_indices
import numpy as np

fslice = slice(None)  # full slice


def test_simplify_indices_slice():
    idx = (slice(None),)
    assert simplify_indices(1, idx) == (slice(None),)
    assert simplify_indices(2, idx) == (slice(None), slice(None))
    assert simplify_indices(3, idx) == (slice(None), slice(None), slice(None))


def test_simplify_indices_int():
    idx = (5,)
    assert simplify_indices(1, idx) == (5,)
    assert simplify_indices(2, idx) == (5, slice(None))
    assert simplify_indices(3, idx) == (5, slice(None), slice(None))


def test_simplify_indices_slice_int():
    idx = (slice(None), 5)
    assert simplify_indices(2, idx) == (slice(None), 5)
    assert simplify_indices(3, idx) == (slice(None), 5, slice(None))
    assert simplify_indices(4, idx) == (slice(None), 5, slice(None), slice(None))


def test_simplify_indices_int_slice():
    idx = (5, slice(None))
    assert simplify_indices(2, idx) == (5, slice(None))
    assert simplify_indices(3, idx) == (5, slice(None), slice(None))
    assert simplify_indices(4, idx) == (5, slice(None), slice(None), slice(None))


def test_simplify_indices_ellipses():
    idx = (...,)
    assert simplify_indices(1, idx) == (slice(None),)
    assert simplify_indices(2, idx) == (slice(None), slice(None))
    assert simplify_indices(3, idx) == (slice(None), slice(None), slice(None))


def test_simplify_indices_int_ellipses():
    idx = (5, ...)
    assert simplify_indices(1, idx) == (5,)
    assert simplify_indices(2, idx) == (5, slice(None))
    assert simplify_indices(3, idx) == (5, slice(None), slice(None))
    assert simplify_indices(4, idx) == (5, slice(None), slice(None), slice(None))


def test_simplify_indices_ellipses_int():
    idx = (..., 5)
    assert simplify_indices(1, idx) == (5,)
    assert simplify_indices(2, idx) == (slice(None), 5)
    assert simplify_indices(3, idx) == (slice(None), slice(None), 5)
    assert simplify_indices(4, idx) == (slice(None), slice(None), slice(None), 5)


def test_simplify_indices_slice_ellipses_int():
    idx = (slice(None), ..., 5)
    assert simplify_indices(2, idx) == (slice(None), 5)
    assert simplify_indices(3, idx) == (slice(None), slice(None), 5)
    assert simplify_indices(4, idx) == (slice(None), slice(None), slice(None), 5)


def test_simplify_indices_slice_int_ellipses_int():
    idx = (slice(None), 4, ..., 5)
    assert simplify_indices(3, idx) == (slice(None), 4, 5)
    assert simplify_indices(4, idx) == (slice(None), 4, slice(None), 5)
    assert simplify_indices(5, idx) == (slice(None), 4, slice(None), slice(None), 5)


def test_simplify_indices_partial_slice_int_ellipses_int():
    idx = (slice(0, 5, 2), 4, ..., 5)
    assert simplify_indices(3, idx) == (slice(0, 5, 2), 4, 5)
    assert simplify_indices(4, idx) == (slice(0, 5, 2), 4, slice(None), 5)
    assert simplify_indices(5, idx) == (slice(0, 5, 2), 4, slice(None), slice(None), 5)


# def test_simplify_indices1():
#     ndim = 2
#     idx = (slice(None), 2)
#     assert simplify_indices(ndim, idx) == (slice(None), 2, slice(None))
#
#     assert simplify_indices(ndim, idx) == (slice(None), 2, slice(None))


# def test_simplify_indices2():
#     ndim =


def dotest(x, *idx):
    x_rv = OperatorRV(Constant(x))
    y = index(x_rv, *idx)
    z = x.__getitem__(idx)
    assert y.shape == z.shape


def test_1d_full_slice():
    x = np.random.randn(5)
    dotest(x, fslice)


def test_1d_partial_slice():
    x = np.random.randn(10)
    dotest(x, slice(2, 8, None))


def test_2d_single_full_slice():
    x = np.random.randn(5, 7)
    dotest(x, fslice)


def test_2d_single_partial_slice():
    x = np.random.randn(5, 7)
    dotest(x, slice(2, 4, None))


def test_2d_double_partial_slice():
    x = np.random.randn(5, 7)
    dotest(x, slice(2, 4, None), slice(1, 2, None))


def test_1d_advanced_indexing():
    x = np.random.randn(10)
    dotest(x, [2, 3])


def test_2d_single_advanced_indexing():
    x = np.random.randn(5, 7)
    dotest(x, [2, 3])


def test_2d_double_advanced_indexing():
    x = np.random.randn(5, 7)
    dotest(x, [2, 3], [5, 1])


def test_2d_scalar_indexing():
    x = np.random.randn(3, 4)
    dotest(x, 2)


def test_2d_scalar_slice_indexing():
    x = np.random.randn(3, 4)
    dotest(x, 2, fslice)


def test_2d_scalar_scalar_indexing():
    x = np.random.randn(3, 3)
    dotest(x, 2, 1)


def test_3d_single_advanced():
    x = np.random.randn(3, 4, 5)
    dotest(x, [2, 1])


def test_3d_scalar_slice_scalar():
    x = np.random.randn(3, 4, 5)
    dotest(x, 2, fslice, 4)


def test_2d_slice_advanced():
    x = np.random.randn(5, 6)
    dotest(x, fslice, [1, 4, 2])


def test_3d_slice_advanced():
    x = np.random.randn(5, 6, 7)
    dotest(x, fslice, [1, 4, 2])


def test_3d_advanced_advanced():
    x = np.random.randn(5, 6, 7)
    dotest(x, [2, 1, 2], [1, 4, 2])


def test_3d_slice_advanced_advanced():
    x = np.random.randn(5, 6, 7)
    dotest(x, fslice, [2, 1, 2], [1, 4, 2])


def test_3d_advanced_partslice_advanced():
    x = np.random.randn(6, 5, 7)
    dotest(x, [2, 1, 2], slice(1, 4), [1, 4, 2])


def test_3d_advanced_partslice():
    x = np.random.randn(6, 5, 7)
    dotest(x, [2, 1, 2], slice(1, 4))


def test_truly_advanced_indexing():
    x = np.random.randn(4, 5, 6, 7)
    dotest(x, fslice, [2, 1, 2], fslice, [1, 4, 2])


def test_1d_array():
    x = np.random.randn(5)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x, idx)


def test_2d_slice_array():
    x = np.random.randn(5, 6)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x, fslice, idx)


def test_2d_array_slice():
    x = np.random.randn(5, 6)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x, idx, fslice)


def test_3d_array_slice_array():
    # triggers advanced indexing
    x = np.random.randn(5, 6, 7)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x, idx, fslice, idx)


def test_3d_slice_array_array():
    # triggers non-advanced indexing
    x = np.random.randn(5, 6, 7)
    idx = np.array([[0, 2], [3, 4]])
    dotest(x, fslice, idx, idx)


def test_4d_slice_array_array_slice():
    x = np.random.randn(5, 6, 5, 4)
    idx = [[0, 2], [3, 4]]
    dotest(x, fslice, idx, idx, fslice)


def test_4d_array_slice_array_slice():
    x = np.random.randn(5, 6, 5, 4)
    idx = [[0, 2], [3, 4]]
    dotest(x, idx, fslice, idx, fslice)


def test_indexing_shapes():
    x0 = np.random.randn(5, 6, 7, 8)
    x = OperatorRV(Constant(x0))
    y = x[:, :, 1:3, 0:3]
    assert y.shape == (5, 6, 2, 3)


def test_indexing_int():
    x = makerv([1, 2, 3])
    y = x[0]
    assert isinstance(y.op, Index)
    assert y.shape == ()


def test_indexing_rv_int():
    x = makerv([1, 2, 3])
    idx = makerv(0)
    y = x[idx]
    assert isinstance(y.op, Index)
    assert y.shape == ()


def test_indexing_rv_list():
    x = makerv([1, 2, 3])
    idx = [0, 1]
    y = x[idx]
    assert isinstance(y.op, Index)
    assert y.shape == (2,)


def test_indexing_rv_array():
    x = makerv([1, 2, 3])
    idx = makerv([0, 1])
    y = x[idx]
    assert isinstance(y.op, Index)
    assert y.shape == (2,)


def test_indexing3():
    x = makerv([1, 2, 3])
    idx = [0, 1]
    y = x[idx]
    assert isinstance(y.op, Index)
    assert y.shape == (2,)


def test_indexing4():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    idx = [0, 1, 1, 0]
    y = x[idx, :]
    assert isinstance(y.op, Index)
    assert y.shape == (4, 3)


def test_indexing5():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    idx = [[0, 1], [1, 1], [0, 0], [1, 0]]
    y = x[:, idx]
    assert isinstance(y.op, Index)
    assert y.shape == (2, 4, 2)


def test_indexing6():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    idx = [0, 1, 1, 0]
    y = x[idx]
    assert isinstance(y.op, Index)
    assert y.shape == (4, 3)


# disabled this optimization for now
# def test_double_indexing_optimization1():
#     x0 = np.random.randn(5, 6, 7, 8)
#     y0 = x0[:, :, 1:3, 0:4]
#     z0 = y0[0:2, 0:3]
#     x = OperatorRV(Constant(x0))
#     y = x[:, :, 1:3, 0:4]
#     z = y[0:2, 0:3]
#     assert x0.shape == x.shape
#     assert y0.shape == y.shape
#     assert z.shape == z0.shape
#     assert isinstance(z.op, Index)
#     assert z.op == Index(slice(0,2), slice(0,3), slice(1,3), slice(0,4))


# def test_double_indexing_optimization2():
#     x0 = np.random.randn(5, 6, 7, 8)
#     y0 = x0[1]
#     z0 = y0[2]
#     x = OperatorRV(Constant(x0))
#     y = x[1]
#     z = y[2]
#     assert x0.shape == x.shape
#     assert y0.shape == y.shape
#     assert z.shape == z0.shape
#     assert isinstance(z.op, Index)
#     print(f"{z.op=}")
#     print(f"{z.parents[0].op=}")
#     #assert z.op == Index(None, None, slice(None), slice(None))


# broadcasting for indices not (yet?) implemented
# def test_4d_array_scalar_array_slice():
#     x = np.random.randn(5, 6, 5, 4)
#     idx = [[0, 2], [3, 4]]
#     dotest(x,idx,fslice,2,fslice)
