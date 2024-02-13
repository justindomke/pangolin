from pangolin import ir
from pangolin.ir import VMapDist, normal_scale, makerv, Index
import numpy as np
import pytest


def test_cond_dist():
    class DummyDist(ir.CondDist):
        def get_shape(self, *parents_shapes):
            return ()

    try:
        DummyDist("dummy")
        assert False, "failed to raise TypeError when no is_random provided"
    except TypeError as e:
        pass

    dist = DummyDist("dummy", True)
    assert dist.random

    dist = DummyDist("dummy", False)
    assert not dist.random

    try:
        dist.x = 5
        assert False, "failed to keep frozen"
    except TypeError as e:
        pass


def test_constant():
    import numpy as np

    x = np.random.randn(3, 4)
    d = ir.Constant(x)
    assert d.get_shape() == (3, 4)

    x2 = x + 0.0
    d2 = ir.Constant(x2)
    assert d == d2


def test_constant_jaxmode():
    from jax import numpy as jnp

    ir.np = jnp
    x = jnp.array([[1, 2, 3], [4, 5, 6.6]])
    d = ir.Constant(x)
    assert d.get_shape() == (2, 3)
    assert d.value.dtype == jnp.dtype("float32")

    x2 = x + 0.0
    d2 = ir.Constant(x2)
    assert d == d2


def test_makerv1():
    x = makerv(1)
    assert x.shape == ()


def test_makerv2():
    x = makerv([1, 2])
    assert x.shape == (2,)


def test_makerv3():
    x = makerv((1, 2))
    assert x.shape == (2,)


def test_makerv4():
    x = makerv(((1, 2), (3, 4), (5, 6)))
    assert x.shape == (3, 2)


def test_makerv5():
    import numpy as np

    x = np.random.randn(3, 4)
    x = makerv(x)
    assert x.shape == (3, 4)


def test_makerv6():
    from jax import numpy as np

    x = np.ones((3, 4))
    x = makerv(x)
    assert x.shape == (3, 4)


# def test_tform1():
#     x = ir.normal_scale(0, 1)
#     # y = x * x * x
#     y = ir.mul(ir.mul(x, x), x)
#     assert y.shape == ()
#     assert y.ndim == 0
#
#
# def test_tform2():
#     x = ir.normal_scale(0, 1)
#     y = ir.normal_scale(0, 1)
#     # z = x * y + y ** (y**2)
#     z = ir.add(ir.mul(x, y), ir.pow(y, ir.pow(y, 2)))
#     assert z.shape == ()
#     assert z.ndim == 0


def test_implicit_vec1():
    x = makerv(1)
    y = makerv([2,3])
    z = ir.vec_add(x,y)
    assert z.shape == (2,)

def test_VMapDist1():
    # couldn't call normal here because it's not a CondDist. But I guess that's fine because user isn't expected
    # to touch VMap directly
    diag_normal = VMapDist(normal_scale, [0, 0], 3)
    assert diag_normal.get_shape((3,), (3,)) == (3,)


def test_VMapDist2():
    diag_normal = VMapDist(normal_scale, [0, None], 3)
    assert diag_normal.get_shape((3,), ()) == (3,)


def test_VMapDist3():
    diag_normal = VMapDist(normal_scale, [None, 0], 3)
    assert diag_normal.get_shape((), (3,)) == (3,)


def test_VMapDist4():
    diag_normal = VMapDist(normal_scale, [None, None], 3)
    assert diag_normal.get_shape((), ()) == (3,)


def test_VMapDist5():
    diag_normal = VMapDist(normal_scale, [None, 0], 3)
    try:
        # should fail because shapes are incoherent
        x = diag_normal.get_shape((), (4,))
        assert False
    except AssertionError as e:
        assert True


def test_double_VMapDist1():
    # couldn't call normal here because it's not a CondDist. But I guess that's fine because user isn't expected
    # to touch VMap directly
    diag_normal = VMapDist(normal_scale, [0, 0])
    matrix_normal = VMapDist(diag_normal, [0, 0])
    assert matrix_normal.get_shape((4, 3), (4, 3)) == (4, 3)


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


fslice = slice(None, None, None)

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
def test_index(start_shape, idx):
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

def test_log_prob1():
    d = ir.normal_scale
    d2 = ir.LogProb(d)
    assert d != d2
    assert d.get_shape((),()) == ()
    assert d2.get_shape((),(),()) == ()
    print(d)
    print(d2)
