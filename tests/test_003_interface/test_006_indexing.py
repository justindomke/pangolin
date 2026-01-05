from jax._src.core import Value
from pangolin import ir
from pangolin.ir import RV, Constant, Index
from pangolin.interface import InfixRV, makerv, constant, override
from pangolin.interface.indexing import index, eliminate_ellipses, convert_index, vector_index
import numpy as np
import pytest

fslice = slice(None)  # full slice


def test_eliminate_ellipses():
    assert eliminate_ellipses(3, (1, 2, 3)) == (1, 2, 3)
    assert eliminate_ellipses(3, (1, ..., 3)) == (1, fslice, 3)
    assert eliminate_ellipses(3, (1, 2, ..., 3)) == (1, 2, 3)


def test_eliminate_ellipses_bad():
    try:
        eliminate_ellipses(4, (1, 2, 3))
        assert False
    except ValueError:
        pass

    try:
        eliminate_ellipses(2, (1, 2, 3))
        assert False
    except ValueError:
        pass


def is_constant_rv_with_value(x, value):
    return isinstance(x, RV) and x.op == Constant(value)


def test_convert_index():
    x = convert_index(5, 3)
    assert is_constant_rv_with_value(x, 3)

    x = convert_index(5, fslice)
    assert is_constant_rv_with_value(x, [0, 1, 2, 3, 4])

    x = convert_index(5, slice(None, None, 2))
    assert is_constant_rv_with_value(x, [0, 2, 4])

    x = convert_index(5, [3, 0, 1])
    assert is_constant_rv_with_value(x, [3, 0, 1])

    y = constant(3)
    assert convert_index(5, y) == y

    y = constant([3, 0, 1])
    assert convert_index(5, y) == y


def test_1d_full_slice():
    x = np.random.randn(5)
    y = constant(x)

    assert (
        index(y, fslice).shape
        == index(y, ...).shape
        == index(y, fslice, ...).shape
        == index(y, ..., fslice).shape
        == y[:].shape
        == y[...].shape
        == y[:, ...].shape
        == y[..., :].shape
        == x.shape
        == x[:].shape
        == x[...].shape
        == x[:, ...].shape
        == x[..., :].shape
    )

    a = y[:]
    b = y[...]
    assert a.op == b.op
    assert a.parents[0] == b.parents[0] == y
    assert a.parents[1].op == b.parents[1].op


def test_1d_wrong_args():
    x = np.random.randn(5)
    y = constant(x)

    with pytest.raises(ValueError):
        index(y)  # not enough indices

    with pytest.raises(ValueError):
        index(y, fslice, fslice)  # too many indices

    with pytest.raises(ValueError):
        y[:, :]  # too many indices


def test_1d_partial_slice():
    x = np.random.randn(10)
    y = constant(x)

    for s in [
        slice(2, 8, None),
        slice(1, 8, 2),
        slice(2, None, None),
        slice(None, None, 3),
    ]:
        assert index(y, s).shape == y[s].shape == x[s].shape


def test_1D_array_ND_indices():
    x = np.random.randn(10)
    y = constant(x)

    for i in (0, [2, 0, 1], [[0, 1, 2], [3, 4, 5]]):
        assert (
            index(y, i).shape
            == index(y, i, ...).shape
            == index(y, ..., i).shape
            == y[i].shape
            == y[i, ...].shape
            == y[..., i].shape
            == x[i].shape
        )


def test_2D_array_ND_indices():
    x = np.random.randn(10, 11)
    y = constant(x)

    options = [0, [2, 0, 1], [[0, 1, 2], [3, 4, 5]]]
    for i in options:
        for j in options:
            expected_shape = np.array(i).shape + np.array(j).shape
            assert (
                index(y, i, j).shape
                == index(y, i, j, ...).shape
                == index(y, i, ..., j).shape
                == index(y, ..., i, j).shape
                == y[i, j].shape
                == y[i, j, ...].shape
                == y[i, ..., j].shape
                == y[..., i, j].shape
                == expected_shape
            )


def test_2D_array_slice_indices():
    x = np.random.randn(10, 11)
    y = constant(x)

    options = [slice(None), slice(1, 8, 2)]
    for i in options:
        for j in options:
            expected_shape = np.ones(10)[i].shape + np.ones(11)[j].shape
            assert (
                index(y, i, j).shape
                == index(y, i, j, ...).shape
                == index(y, i, ..., j).shape
                == index(y, ..., i, j).shape
                == y[i, j].shape
                == y[i, j, ...].shape
                == y[i, ..., j].shape
                == y[..., i, j].shape
                == expected_shape
                == x[i, j].shape
            )


def test_scalar_indexing_no_broadcasting():
    data = np.random.randn(10, 11, 12)
    x = constant(data)

    expected_parent_ops = (ir.Constant(data), ir.Constant(3), ir.Constant(4), ir.Constant(5))

    with override(broadcasting="off"):
        for y in vector_index(x, 3, 4, 5), x.s[3, 4, 5]:
            assert y.shape == ()
            assert y.op == ir.VectorIndex()
            assert y.parents[0] == x
            assert y.parent_ops == expected_parent_ops

        for y in [vector_index(data, 3, 4, 5)]:
            assert y.shape == ()
            assert y.op == ir.VectorIndex()
            assert y.parent_ops == expected_parent_ops

        a = constant(3)
        b = constant(4)
        c = constant(5)
        for y in vector_index(x, a, b, c), x.s[a, b, c]:
            assert y.shape == ()
            assert y.op == ir.VectorIndex()
            assert y.parents == (x, a, b, c)
            assert y.parent_ops == expected_parent_ops

        try:
            y = vector_index(x, 3, 4)
            assert False
        except ValueError:
            pass

        try:
            y = vector_index(x, [3, 4], [5, 6], [7, 8])
            assert False
        except ValueError:
            pass

        try:
            y = vector_index(x, slice(None), slice(None), slice(None))
            assert False
        except ValueError:
            pass


def test_scalar_indexing_simple_broadcasting():
    data = np.random.randn(3, 4, 5)
    x = constant(data)

    expected_parent_ops = (ir.Constant(data), ir.Constant(1), ir.Constant(2), ir.Constant(3))

    with override(broadcasting="simple"):
        y = vector_index(x, 1, 2, 3)
        assert y.shape == ()
        assert y.op == ir.VectorIndex()
        assert y.parents[0] == x
        assert y.parent_ops == expected_parent_ops

        y = vector_index(data, 1, 2, 3)
        assert y.shape == ()
        assert y.op == ir.VectorIndex()
        assert y.parent_ops == expected_parent_ops

        a = constant(1)
        b = constant(2)
        c = constant(3)
        y = vector_index(x, a, b, c)
        assert y.shape == ()
        assert y.op == ir.VectorIndex()
        assert y.parents == (x, a, b, c)
        assert y.parent_ops == expected_parent_ops

        try:
            y = vector_index(x, 1, 2)
            assert False
        except ValueError:
            pass

        A = constant([1, 2])
        B = constant([2, 3])
        C = constant([3, 4])
        y = vector_index(x, A, B, C)
        assert y.shape == (2,)
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, 0, 0), 2)
        assert y.parents == (x, A, B, C)

        y = vector_index(x, [1, 2], [2, 3], [3, 4])
        assert y.shape == (2,)
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, 0, 0), 2)
        assert y.parent_ops == (Constant(data), Constant([1, 2]), Constant([2, 3]), Constant([3, 4]))

        y = vector_index(x, [1, 2], 2, [3, 4])
        assert y.shape == (2,)
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, None, 0), 2)
        assert y.parents[0] == x
        assert y.parent_ops == (x.op, Constant([1, 2]), Constant(2), Constant([3, 4]))

        y = vector_index(data, [1, 2], 2, [3, 4])
        assert y.shape == (2,)
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, None, 0), 2)
        assert y.parent_ops == (Constant(data), Constant([1, 2]), Constant(2), Constant([3, 4]))

        y = vector_index(data, [[1, 2, 0], [1, 0, 1]], 0, 1)
        assert y.op == ir.VMap(ir.VMap(ir.VectorIndex(), (None, 0, None, None), 3), (None, 0, None, None), 2)
        assert y.shape == (2, 3)

        try:
            y = vector_index(data, [[1, 2, 0], [1, 0, 1]], 0, [0, 1, 0])
            assert False
        except ValueError:
            pass

        y = vector_index(x, slice(0, 2, 1), slice(0, 2, 1), slice(0, 2, 1))
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, 0, 0), 2)
        assert y.shape == (2,)
        assert y.parent_ops[1] == y.parent_ops[2] == y.parent_ops[3] == ir.Constant([0, 1])

        y = vector_index(x, slice(0, 2, 1), slice(0, 2, 1), 1)
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, 0, None), 2)
        assert y.parent_ops[1] == y.parent_ops[2] == ir.Constant([0, 1])
        assert y.parent_ops[3] == ir.Constant(1)


def test_scalar_indexing_numpy_broadcasting():
    data = np.random.randn(3, 4, 5)
    x = constant(data)

    expected_parent_ops = (ir.Constant(data), ir.Constant(1), ir.Constant(2), ir.Constant(3))

    with override(broadcasting="numpy"):
        y = vector_index(x, 1, 2, 3)
        assert y.shape == ()
        assert y.op == ir.VectorIndex()
        assert y.parents[0] == x
        assert y.parent_ops == expected_parent_ops

        y = vector_index(data, 1, 2, 3)
        assert y.shape == ()
        assert y.op == ir.VectorIndex()
        assert y.parent_ops == expected_parent_ops

        a = constant(1)
        b = constant(2)
        c = constant(3)
        y = vector_index(x, a, b, c)
        assert y.shape == ()
        assert y.op == ir.VectorIndex()
        assert y.parents == (x, a, b, c)
        assert y.parent_ops == expected_parent_ops

        try:
            y = vector_index(x, 1, 2)
            assert False
        except ValueError:
            pass

        A = constant([1, 2])
        B = constant([2, 3])
        C = constant([3, 4])
        y = vector_index(x, A, B, C)
        assert y.shape == (2,)
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, 0, 0), 2)
        assert y.parents == (x, A, B, C)

        y = vector_index(x, [1, 2], [2, 3], [3, 4])
        assert y.shape == (2,)
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, 0, 0), 2)
        assert y.parent_ops == (Constant(data), Constant([1, 2]), Constant([2, 3]), Constant([3, 4]))

        y = vector_index(x, [1, 2], 2, [3, 4])
        assert y.shape == (2,)
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, None, 0), 2)
        assert y.parents[0] == x
        assert y.parent_ops == (x.op, Constant([1, 2]), Constant(2), Constant([3, 4]))

        y = vector_index(data, [1, 2], 2, [3, 4])
        assert y.shape == (2,)
        assert y.op == ir.VMap(ir.VectorIndex(), (None, 0, None, 0), 2)
        assert y.parent_ops == (Constant(data), Constant([1, 2]), Constant(2), Constant([3, 4]))

        y = vector_index(data, [[1, 2, 0], [1, 0, 1]], 0, 1)
        assert y.op == ir.VMap(ir.VMap(ir.VectorIndex(), (None, 0, None, None), 3), (None, 0, None, None), 2)
        assert y.shape == (2, 3)

        y = vector_index(data, [[1, 2, 0], [1, 0, 1]], 0, [0, 1, 0])
        assert y.op == ir.VMap(ir.VMap(ir.VectorIndex(), (None, 0, None, 0), 3), (None, 0, None, None), 2)


# def test_2d_single_full_slice():
#     x = np.random.randn(5, 7)
#     dotest(x, fslice)


# def test_2d_single_partial_slice():
#     x = np.random.randn(5, 7)
#     dotest(x, slice(2, 4, None))


# def test_2d_double_partial_slice():
#     x = np.random.randn(5, 7)
#     dotest(x, slice(2, 4, None), slice(1, 2, None))


# def test_1d_advanced_indexing():
#     x = np.random.randn(10)
#     dotest(x, [2, 3])


# def test_2d_single_advanced_indexing():
#     x = np.random.randn(5, 7)
#     dotest(x, [2, 3])


# def test_2d_double_advanced_indexing():
#     x = np.random.randn(5, 7)
#     dotest(x, [2, 3], [5, 1])


# def test_2d_scalar_indexing():
#     x = np.random.randn(3, 4)
#     dotest(x, 2)


# def test_2d_scalar_slice_indexing():
#     x = np.random.randn(3, 4)
#     dotest(x, 2, fslice)


# def test_2d_scalar_scalar_indexing():
#     x = np.random.randn(3, 3)
#     dotest(x, 2, 1)


# def test_3d_single_advanced():
#     x = np.random.randn(3, 4, 5)
#     dotest(x, [2, 1])


# def test_3d_scalar_slice_scalar():
#     x = np.random.randn(3, 4, 5)
#     dotest(x, 2, fslice, 4)


# def test_2d_slice_advanced():
#     x = np.random.randn(5, 6)
#     dotest(x, fslice, [1, 4, 2])


# def test_3d_slice_advanced():
#     x = np.random.randn(5, 6, 7)
#     dotest(x, fslice, [1, 4, 2])


# def test_3d_advanced_advanced():
#     x = np.random.randn(5, 6, 7)
#     dotest(x, [2, 1, 2], [1, 4, 2])


# def test_3d_slice_advanced_advanced():
#     x = np.random.randn(5, 6, 7)
#     dotest(x, fslice, [2, 1, 2], [1, 4, 2])


# def test_3d_advanced_partslice_advanced():
#     x = np.random.randn(6, 5, 7)
#     dotest(x, [2, 1, 2], slice(1, 4), [1, 4, 2])


# def test_3d_advanced_partslice():
#     x = np.random.randn(6, 5, 7)
#     dotest(x, [2, 1, 2], slice(1, 4))


# def test_truly_advanced_indexing():
#     x = np.random.randn(4, 5, 6, 7)
#     dotest(x, fslice, [2, 1, 2], fslice, [1, 4, 2])


# def test_1d_array():
#     x = np.random.randn(5)
#     idx = np.array([[0, 2], [3, 4]])
#     dotest(x, idx)


# def test_2d_slice_array():
#     x = np.random.randn(5, 6)
#     idx = np.array([[0, 2], [3, 4]])
#     dotest(x, fslice, idx)


# def test_2d_array_slice():
#     x = np.random.randn(5, 6)
#     idx = np.array([[0, 2], [3, 4]])
#     dotest(x, idx, fslice)


# def test_3d_array_slice_array():
#     # triggers advanced indexing
#     x = np.random.randn(5, 6, 7)
#     idx = np.array([[0, 2], [3, 4]])
#     dotest(x, idx, fslice, idx)


# def test_3d_slice_array_array():
#     # triggers non-advanced indexing
#     x = np.random.randn(5, 6, 7)
#     idx = np.array([[0, 2], [3, 4]])
#     dotest(x, fslice, idx, idx)


# def test_4d_slice_array_array_slice():
#     x = np.random.randn(5, 6, 5, 4)
#     idx = [[0, 2], [3, 4]]
#     dotest(x, fslice, idx, idx, fslice)


# def test_4d_array_slice_array_slice():
#     x = np.random.randn(5, 6, 5, 4)
#     idx = [[0, 2], [3, 4]]
#     dotest(x, idx, fslice, idx, fslice)


# def test_indexing_shapes():
#     x0 = np.random.randn(5, 6, 7, 8)
#     x = OperatorRV(Constant(x0))
#     y = x[:, :, 1:3, 0:3]
#     assert y.shape == (5, 6, 2, 3)


# def test_indexing_int():
#     x = makerv([1, 2, 3])
#     y = x[0]
#     assert isinstance(y.op, Index)
#     assert y.shape == ()


# def test_indexing_rv_int():
#     x = makerv([1, 2, 3])
#     idx = makerv(0)
#     y = x[idx]
#     assert isinstance(y.op, Index)
#     assert y.shape == ()


# def test_indexing_rv_list():
#     x = makerv([1, 2, 3])
#     idx = [0, 1]
#     y = x[idx]
#     assert isinstance(y.op, Index)
#     assert y.shape == (2,)


# def test_indexing_rv_array():
#     x = makerv([1, 2, 3])
#     idx = makerv([0, 1])
#     y = x[idx]
#     assert isinstance(y.op, Index)
#     assert y.shape == (2,)


# def test_indexing3():
#     x = makerv([1, 2, 3])
#     idx = [0, 1]
#     y = x[idx]
#     assert isinstance(y.op, Index)
#     assert y.shape == (2,)


# def test_indexing4():
#     x = makerv([[1, 2, 3], [4, 5, 6]])
#     idx = [0, 1, 1, 0]
#     y = x[idx, :]
#     assert isinstance(y.op, Index)
#     assert y.shape == (4, 3)


# def test_indexing5():
#     x = makerv([[1, 2, 3], [4, 5, 6]])
#     idx = [[0, 1], [1, 1], [0, 0], [1, 0]]
#     y = x[:, idx]
#     assert isinstance(y.op, Index)
#     assert y.shape == (2, 4, 2)


# def test_indexing6():
#     x = makerv([[1, 2, 3], [4, 5, 6]])
#     idx = [0, 1, 1, 0]
#     y = x[idx]
#     assert isinstance(y.op, Index)
#     assert y.shape == (4, 3)


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
