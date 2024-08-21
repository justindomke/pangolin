import pytest
from pangolin.ir.vmap import VMap
#from pangolin.ir.rv import makerv
from pangolin.ir import Normal, StudentT
import numpy as np
from pangolin.ir.index import Index


def test_eq_vmap1():
    d1 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    d2 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    assert d1 == d2


def test_eq_vmap2():
    d1 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    d2 = VMap(StudentT(), in_axes=[0, 1], axis_size=5)
    assert d1 != d2


def test_eq_vmap3():
    d1 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    d2 = VMap(Normal(), in_axes=[1, 0], axis_size=5)
    assert d1 != d2


def test_eq_vmap4():
    d1 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    d2 = VMap(Normal(), in_axes=[0, 1], axis_size=4)
    assert d1 != d2


def test_eq_vmap5():
    d1 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    d2 = VMap(d1, in_axes=[0, 1], axis_size=5)
    d3 = VMap(d1, in_axes=[0, 1], axis_size=5)
    assert d1 != d2
    assert d1 != d3
    assert d2 == d3


def test_eq_vmap6():
    d1 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    d2 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    d3 = VMap(d1, in_axes=[0, 1], axis_size=5)
    d4 = VMap(d2, in_axes=[0, 1], axis_size=5)
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d2 != d3
    assert d2 != d4
    assert d3 == d4


def test_eq_vmap7():
    d1 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    d2 = VMap(Normal(), in_axes=[0, 1], axis_size=5)
    d3 = VMap(d1, in_axes=[0, 1], axis_size=5)
    d4 = VMap(d2, in_axes=[0, 1], axis_size=4)
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d2 != d3
    assert d2 != d4
    assert d3 != d4


def test_eq_vmap8():
    d1 = VMap(Normal(), in_axes=[0, 1])
    d2 = VMap(Normal(), in_axes=[0, 1])
    d3 = VMap(d1, in_axes=[0, 1])
    d4 = VMap(d2, in_axes=[0, 1])
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d2 != d3
    assert d2 != d4
    assert d3 == d4


def test_VMapDist1():
    # couldn't call normal here because it's not a CondDist. But I guess that's fine because user isn't expected
    # to touch VMap directly
    diag_normal = VMap(Normal(), [0, 0], 3)
    assert diag_normal.get_shape((3,), (3,)) == (3,)


def test_VMapDist2():
    diag_normal = VMap(Normal(), [0, None], 3)
    assert diag_normal.get_shape((3,), ()) == (3,)


def test_VMapDist3():
    diag_normal = VMap(Normal(), [None, 0], 3)
    assert diag_normal.get_shape((), (3,)) == (3,)


def test_VMapDist4():
    diag_normal = VMap(Normal(), [None, None], 3)
    assert diag_normal.get_shape((), ()) == (3,)


def test_VMapDist5():
    diag_normal = VMap(Normal(), [None, 0], 3)
    try:
        # should fail because shapes are incoherent
        x = diag_normal.get_shape((), (4,))
        assert False
    except AssertionError as e:
        assert True


def test_double_VMapDist1():
    diag_normal = VMap(Normal(), [0, 0])
    matrix_normal = VMap(diag_normal, [0, 0])
    assert matrix_normal.get_shape((4, 3), (4, 3)) == (4, 3)


def test_vmap_index1():
    base_op = Index(None)
    assert base_op.get_shape((10,), ()) == ()
    op = VMap(base_op, [None, 0])
    assert op.get_shape((10,), (5,)) == (5,)


def test_vmap_index2():
    base_op = Index(slice(None), None)
    op = VMap(base_op, [None, 0])
    assert op.get_shape((10, 4), (5,)) == (5, 10) # vmap dim first


def test_repr():
    d = VMap(Normal(), [0, 0])
    assert repr(d) == "VMap(Normal(),(0, 0))"


def test_str():
    d = VMap(Normal(), [0, 0])
    assert str(d) == "VMap(normal,(0, 0))"
