from pangolin.interface import (
    RV,
    normal,
    normal_scale,
    makerv,
    exp,
    Constant,
    vmap,
    VMapDist,
    viz_upstream,
    # Loop,
    # LoopRV,
    print_upstream,
    add,
    # VMapRV,
    # fakeloop,
    # FakeLoop,
)
from pangolin import dag
import numpy as np  # type: ignore
from pangolin.transforms.normal_normal import normal_normal
from pangolin.transforms.constant_op import constant_op
from pangolin.ir import Loop, SlicedRV, make_sliced_rv
from pangolin import *


def test_context_manager1():
    assert Loop.loops == []
    with Loop() as i:
        assert Loop.loops == [i]
        assert i.length is None
    assert Loop.loops == []


def test_context_manager2():
    assert Loop.loops == []
    with Loop(3) as i:
        assert Loop.loops == [i]
        assert i.length == 3
    assert Loop.loops == []


def test_context_manager3():
    assert Loop.loops == []
    with Loop() as i:
        assert Loop.loops == [i]
        with Loop() as j:
            assert Loop.loops == [i, j]
        assert Loop.loops == [i]
    assert Loop.loops == []


def test_context_manager4():
    assert Loop.loops == []
    with Loop(3) as i:
        assert Loop.loops == [i]
        assert i.length == 3
        with Loop(4) as j:
            assert Loop.loops == [i, j]
            assert j.length == 4
        assert Loop.loops == [i]
    assert Loop.loops == []


def test_context_manager5():
    i = Loop(3)
    j = Loop(4)
    assert Loop.loops == []
    with i:
        assert Loop.loops == [i]
        assert i.length == 3
        with j:
            assert Loop.loops == [i, j]
            assert j.length == 4
        assert Loop.loops == [i]
    assert Loop.loops == []


def test_make_sliced_rv_no_loops():
    loop = Loop(3)
    x_slice = make_sliced_rv(normal_scale, 0, 1, all_loops=[loop])
    assert isinstance(x_slice, SlicedRV)
    assert x_slice.shape == ()
    assert x_slice.full_rv.shape == (3,)


def test_make_sliced_rv_no_in_axes():
    loc = makerv(0)
    scale = makerv(1)
    loop = Loop(3)
    x_slice = make_sliced_rv(normal_scale, loc, scale, all_loops=[loop])
    assert isinstance(x_slice, SlicedRV)
    assert x_slice.full_rv.shape == (3,)
    assert x_slice.full_rv.cond_dist == VMapDist(normal_scale, (None, None), 3)
    assert x_slice.loop_axes == (0,)


def test_make_sliced_rv_no_in_axes_implicit():
    loc = makerv(0)
    scale = makerv(1)
    with Loop(3):
        x = normal_scale(loc, scale)
    assert isinstance(x, SlicedRV)
    assert x.full_rv.shape == (3,)
    assert x.full_rv.cond_dist == VMapDist(normal_scale, (None, None), 3)
    assert x.loop_axes == (0,)


def test_make_sliced_rv_no_in_axes_multivariate():
    loc = makerv([0, 1])
    cov = makerv([[3, 1], [1, 2]])
    with Loop(3):
        x = multi_normal_cov(loc, cov)
    assert isinstance(x, SlicedRV)
    assert x.full_rv.shape == (3, 2)
    assert x.full_rv.cond_dist == VMapDist(multi_normal_cov, (None, None), 3)
    assert x.loop_axes == (0,)


def test_no_in_axes_double_loop():
    loc = makerv(0)
    scale = makerv(1)
    with Loop(3):
        with Loop(5):
            x = normal_scale(loc, scale)
    assert isinstance(x, SlicedRV)
    assert x.full_rv.shape == (3, 5)
    assert x.full_rv.cond_dist == VMapDist(
        VMapDist(normal_scale, (None, None), 5), (None, None), 3
    )
    assert x.loop_axes == (0, 1)


def test_tracing():
    loc = makerv(0)
    scale = makerv(1)
    with Loop(3):
        x = normal_scale(loc, scale)
        y = add(x, x)
    assert x.full_rv.shape == (3,)
    assert x.full_rv.cond_dist == VMapDist(normal_scale, (None, None), 3)
    assert x.loop_axes == (0,)
    assert y.full_rv.shape == (3,)
    assert y.full_rv.cond_dist == VMapDist(add, (0, 0), 3)
    assert y.loop_axes == (0,)
    assert y.full_rv.parents == (x.full_rv, x.full_rv)


def test_tracing_outside_var():
    loc = makerv(0)
    scale = makerv(1)
    z = normal_scale(0, 1)
    with Loop(3):
        x = normal_scale(loc, scale)
        y = add(x, z)
    assert x.full_rv.shape == (3,)
    assert x.full_rv.cond_dist == VMapDist(normal_scale, (None, None), 3)
    assert x.loop_axes == (0,)
    assert y.full_rv.shape == (3,)
    assert y.full_rv.cond_dist == VMapDist(add, (0, None), 3)
    assert y.loop_axes == (0,)
    assert y.full_rv.parents == (x.full_rv, z)


def test_assignment():
    loc = makerv(0)
    scale = makerv(1)
    x = VMapRV()
    with Loop(3) as i:
        x[i] = normal(loc, scale)
    print_upstream(x)


def test_2d_assignment():
    loc = makerv(0)
    scale = makerv(1)
    x = VMapRV()
    y = VMapRV()
    with Loop(3) as i:
        x[i] = normal(loc, scale)
        with Loop(5) as j:
            y[i, j] = normal(loc, scale)


def test_slice_existing1():
    x = makerv([1, 2, 3])
    loop = Loop()
    x_loop = slice_existing_rv(x, [loop])

    assert x_loop.shape == ()
    assert x_loop.full_rv == x
    assert x_loop.parents == ()
    assert x_loop.loops == (loop,)
    assert x_loop.loop_axes == (0,)


def test_slice_existing2():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    loop1 = Loop()
    loop2 = Loop()
    x_loop = slice_existing_rv(x, [loop1, loop2])

    assert x_loop.shape == ()
    assert x_loop.full_rv == x
    assert x_loop.parents == ()
    assert x_loop.loops == (loop1, loop2)
    assert x_loop.loop_axes == (0, 1)


def test_slice_existing3():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    loop1 = Loop()
    loop2 = Loop()
    x_loop = slice_existing_rv(x, [slice(None), loop2])

    assert x_loop.shape == (2,)
    assert x_loop.full_rv == x
    assert x_loop.parents == ()
    assert x_loop.loops == (loop2,)
    assert x_loop.loop_axes == (1,)


def test_slice_existing4():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    loop1 = Loop()
    loop2 = Loop()
    x_loop = slice_existing_rv(x, [loop1, slice(None)])

    assert x_loop.shape == (3,)
    assert x_loop.full_rv == x
    assert x_loop.parents == ()
    assert x_loop.loops == (loop1,)
    assert x_loop.loop_axes == (0,)


def test_rhs_slicing1():
    z = makerv([1, 2, 3])
    with Loop() as i:
        z_loop = z[i]
    assert z_loop.loops == (i,)
    assert z_loop.loop_axes == (0,)


def test_loops_with_full_slicing1():
    z = makerv([1, 2, 3])
    x = VMapRV()
    scale = makerv(1)  # TODO: CURRENTLY NECESSARY!
    with Loop() as i:
        # x[i] = normal_scale(z[i], scale)
        zi = z[i]
        xi = normal_scale(zi, scale)
        assert xi.loops == (i,)
        assert xi.loop_axes == (0,)
        x[i] = xi

    print_upstream(x)


def test_double_loops1():
    z = makerv([[1, 2], [3, 4], [5, 6]])

    x = VMapRV()
    with Loop(3) as i:
        with Loop(2) as j:
            zij = z[i, j]
            xij = normal_scale(zij, zij)
            x[i, j] = xij
    print_upstream(x)


def test_double_loops2():
    x = makerv([7, 8, 9])
    y = makerv([10, 11])

    z = VMapRV()
    with Loop(3) as i:
        with Loop(2) as j:
            # xi = x[i]
            # yj = y[j]
            # zij = xi + yj
            # z[i, j] = zij
            z[i, j] = x[i] + y[j]
    print_upstream(z)

    z_samp = sample(z)[0]
    expected = x.cond_dist.value[:, None] + y.cond_dist.value[None, :]
    assert np.allclose(z_samp, expected)


def test_double_loops3():
    x = makerv([7, 8, 9])
    y = makerv([[1, 2], [3, 4], [5, 6]])

    z = VMapRV()
    with Loop(3) as i:
        with Loop(2) as j:
            # xi = x[i]
            # yij = y[i, j]
            # zij = xi + yij
            # z[i, j] = zij
            z[i, j] = x[i] + y[i, j]

    z_samp = sample(z)[0]
    expected = x.cond_dist.value[:, None] + y.cond_dist.value
    assert np.allclose(z_samp, expected)

    print_upstream(z)


def test_double_loops4():
    x = makerv([7, 8, 9])
    y = makerv([[1, 2], [3, 4], [5, 6]])

    z = VMapRV()
    u = VMapRV()
    with Loop(3) as i:
        z[i] = x[i] + x[i]
        with Loop(2) as j:
            # xi = x[i]
            # yij = y[i, j]
            # zij = xi + yij
            # z[i, j] = zij
            u[i, j] = z[i] + y[i, j]

    u_samp = sample(u)[0]
    expected = 2 * x.cond_dist.value[:, None] + y.cond_dist.value
    assert np.allclose(u_samp, expected)

    print_upstream(z)


def test_double_loops5():
    loc = makerv([7, 8, 9])
    scale_z = makerv([10, 11, 12])
    scale_x = makerv([[1, 2], [3, 4], [5, 6]])

    assert scale_x.shape == (3, 2)

    z = VMapRV()
    x = VMapRV()
    with Loop(3) as i:
        z[i] = normal_scale(loc[i], scale_z[i])
        with Loop(2) as j:
            x[i, j] = normal_scale(z[i], scale_x[i, j])
    print_upstream(x)


def test_double_loops6():
    loc = makerv([7, 8, 9])
    scale_z = makerv([10, 11, 12])
    scale_x = makerv([[1, 2], [3, 4], [5, 6]])

    assert scale_x.shape == (3, 2)

    z = VMapRV()
    x = VMapRV()
    with Loop() as i:
        z[i] = normal_scale(loc[i], scale_z[i])
        with Loop() as j:
            x[i, j] = normal_scale(z[i], scale_x[i, j])
    print_upstream(x)


# def test_shapes():
#     var = makerv(np.random.randn(3))
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[0], loops=[Loop(3)])
#     assert loop_var.shape == ()
#
#     var = makerv(np.random.randn(3, 4))
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[0], loops=[Loop(3)])
#     assert loop_var.shape == (4,)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[1], loops=[Loop(4)])
#     assert loop_var.shape == (3,)
#
#     var = makerv(np.random.randn(3, 4, 5))
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[0], loops=[Loop(3)])
#     assert loop_var.shape == (4, 5)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[1], loops=[Loop(4)])
#     assert loop_var.shape == (3, 5)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[2], loops=[Loop(5)])
#     assert loop_var.shape == (3, 4)
#
#     loop_a = Loop()
#     loop_b = Loop()
#     var = makerv(np.random.randn(3, 4, 5))
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 1], loops=[loop_a, loop_b])
#     assert loop_var.shape == (5,)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 2], loops=[loop_a, loop_b])
#     assert loop_var.shape == (4,)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[1, 2], loops=[loop_a, loop_b])
#     assert loop_var.shape == (3,)
#
#     var = makerv(np.random.randn(3, 4, 5, 6))
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 1], loops=[loop_a, loop_b])
#     assert loop_var.shape == (5, 6)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 2], loops=[loop_a, loop_b])
#     assert loop_var.shape == (4, 6)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 3], loops=[loop_a, loop_b])
#     assert loop_var.shape == (4, 5)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[1, 2], loops=[loop_a, loop_b])
#     assert loop_var.shape == (3, 6)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[1, 3], loops=[loop_a, loop_b])
#     assert loop_var.shape == (3, 5)
#     loop_var = LoopRV(None, vmap_var=var, loop_dims=[2, 3], loops=[loop_a, loop_b])
#     assert loop_var.shape == (3, 4)
#


#
# def test_generating1():
#     loop = Loop()
#     print(f"{loop.id=}")
#     x = makerv(np.random.randn(3))
#     y = makerv(np.random.randn(3))
#     x_loop = make_sliced_rv(x)
#     x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop])
#     y_loop = LoopRV(None, vmap_var=y, loop_dims=[0], loops=[loop])
#     z_loop = x_loop + y_loop
#
#     assert x_loop.shape == ()
#     assert y_loop.shape == ()
#     assert z_loop.shape == ()
#
#     assert z_loop.loops == [loop]
#     assert z_loop.loop_dims == [0]
#
#     z = z_loop.vmap_var
#     assert z.shape == (3,)
#     assert z.cond_dist == VMapDist(add, [0, 0], None)
#
#     print_upstream(z)


# def test_generating2():
#     loop1 = Loop()
#     loop2 = Loop()
#     x = makerv(np.random.randn(3))
#     y = makerv(np.random.randn(4))
#     x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop1])
#     y_loop = LoopRV(None, vmap_var=y, loop_dims=[0], loops=[loop2])
#     z_loop = add(x_loop, y_loop)
#
#     assert x_loop.shape == ()
#     assert y_loop.shape == ()
#     assert z_loop.shape == ()
#     assert z_loop.loops == [loop1, loop2]
#     assert z_loop.loop_dims == [0, 1]
#
#     z = z_loop.vmap_var
#
#     expected_cond_dist = VMapDist(VMapDist(add, [None, 0]), [0, None])
#     assert z.shape == (3, 4)
#     assert z.cond_dist == expected_cond_dist
#
#
# def test_generating3():
#     loop1 = Loop()
#     loop2 = Loop()
#     x = makerv(np.random.randn(3))
#     y = makerv(np.random.randn(3, 4))
#     x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop1])
#     y_loop = LoopRV(None, vmap_var=y, loop_dims=[0, 1], loops=[loop1, loop2])
#     z_loop = add(x_loop, y_loop)
#
#     assert x_loop.shape == ()
#     assert y_loop.shape == ()
#     assert z_loop.shape == ()
#     assert z_loop.loops == [loop1, loop2]
#     assert z_loop.loop_dims == [0, 1]
#
#     z = z_loop.vmap_var
#
#     expected_cond_dist = VMapDist(VMapDist(add, [None, 0]), [0, 0])
#     assert z.shape == (3, 4)
#     assert z.cond_dist == expected_cond_dist
#
#
# def test_generating4():
#     loop1 = Loop()
#     loop2 = Loop()
#     loop3 = Loop()
#     x = makerv(np.random.randn(3))
#     y = makerv(np.random.randn(4, 5))
#     x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop1])
#     y_loop = LoopRV(None, vmap_var=y, loop_dims=[0, 1], loops=[loop2, loop3])
#     z_loop = add(x_loop, y_loop)
#
#     assert x_loop.shape == ()
#     assert y_loop.shape == ()
#     assert z_loop.shape == ()
#     assert z_loop.loops == [loop1, loop2, loop3]
#     assert z_loop.loop_dims == [0, 1, 2]
#
#     z = z_loop.vmap_var
#
#     dist1 = VMapDist(add, [None, 0])
#     dist2 = VMapDist(dist1, [None, 0])
#     dist3 = VMapDist(dist2, [0, None])
#     expected_cond_dist = dist3
#     assert z.shape == (3, 4, 5)
#     assert z.cond_dist == expected_cond_dist
#
#
# def test_generating5():
#     loop1 = Loop()
#     loop2 = Loop()
#     x = makerv(np.random.randn(3))
#     y = makerv(np.random.randn(4, 3))
#     x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop1])
#     y_loop = LoopRV(None, vmap_var=y, loop_dims=[0, 1], loops=[loop2, loop1])
#     z_loop = add(x_loop, y_loop)
#
#     assert x_loop.shape == ()
#     assert y_loop.shape == ()
#     assert z_loop.shape == ()
#     assert z_loop.loops == [loop1, loop2]
#     assert z_loop.loop_dims == [0, 1]
#
#     z = z_loop.vmap_var
#
#     dist1 = VMapDist(add, [None, 0])
#     dist2 = VMapDist(dist1, [0, 1])
#     expected_cond_dist = dist2
#     assert z.shape == (3, 4)
#     assert z.cond_dist == expected_cond_dist
#
#
# def test_generating6():
#     loop1 = Loop()
#     loop2 = Loop()
#     x = makerv(np.random.randn(3, 4, 5))
#     y = makerv(np.random.randn(5, 4))
#     x_loop = LoopRV(None, vmap_var=x, loop_dims=[2], loops=[loop1])
#     y_loop = LoopRV(None, vmap_var=y, loop_dims=[0], loops=[loop1])
#     z_loop = x_loop @ y_loop
#
#     assert x_loop.shape == (3, 4)
#     assert y_loop.shape == (4,)
#     assert z_loop.shape == (3,)
#     assert z_loop.loops == [loop1]
#     assert z_loop.loop_dims == [0]
#
#     z = z_loop.vmap_var
#
#     assert z.shape == (5, 3)
#
#


def test_index_syntax1():
    x = makerv(np.random.randn(3, 4, 5))
    with Loop() as loop:
        x_loop = x[loop, :, :]
    assert x_loop.shape == (4, 5)
    assert x_loop.loop_axes == (0,)
    assert x_loop.loops == (loop,)


def test_index_syntax2():
    x = makerv(np.random.randn(3, 4, 5))
    # loop1 = Loop()
    # loop2 = Loop()
    with Loop() as loop1:
        with Loop() as loop2:
            x_loop = x[loop1, :, loop2]
    assert x_loop.shape == (4,)
    assert x_loop.loop_axes == (0, 2)
    assert x_loop.loops == (loop1, loop2)


def test_index_syntax3():
    x = makerv(np.random.randn(3, 4, 5))

    with Loop() as loop1:
        with Loop() as loop2:
            x_loop = x[loop2, :, loop1]
    assert x_loop.shape == (4,)
    assert x_loop.loops == (loop1, loop2)
    assert x_loop.loop_axes == (2, 0)


def test_index_syntax4():
    x = makerv(np.random.randn(3, 4, 1))
    y = makerv(np.random.randn(3, 1, 5))
    with Loop() as i:
        x_loop = x[i, :, :]
        y_loop = y[i, :, :]
        z_loop = x_loop @ y_loop
    assert x_loop.shape == (4, 1)
    assert y_loop.shape == (1, 5)
    z = z_loop.full_rv
    assert z.shape == (3, 4, 5)


def test_index_syntax5():
    x = makerv(np.random.randn(3, 4, 5))
    y = makerv(np.random.randn(5, 6, 3))
    with Loop() as i:
        x_loop = x[i, :, :]
        y_loop = y[:, :, i]
        z_loop = x_loop @ y_loop
    assert x_loop.shape == (4, 5)
    assert y_loop.shape == (5, 6)
    z = z_loop.full_rv
    assert z.shape == (3, 4, 6)


def test_assign_syntax1():
    x = makerv(np.random.randn(2, 3))
    y = VMapRV()
    with Loop() as i:
        y[i] = x[i, :]
    assert y.shape == x.shape


def test_assign_syntax2():
    x = makerv(np.random.randn(2))
    y = makerv(np.random.randn(3))
    z = VMapRV()
    with Loop() as i:
        with Loop() as j:
            z[i, j] = x[i] * y[j]
    print(f"{z=}")
    print_upstream(z)
    assert z.shape == (2, 3)


def test_assign_syntax3():
    "What if there's no loop var on right?"
    z = VMapRV()
    with Loop(3) as i:
        z[i] = normal(0, 1)
    assert z.shape == (3,)


def test_full_syntax1():
    x = VMapRV()
    y = VMapRV()
    with Loop(3) as i:
        x[i] = normal(0, 1)
        with Loop(4) as j:
            y[i, j] = normal(x[i], 1)
    assert x.shape == (3,)
    assert y.shape == (3, 4)


# def test_generating():
#     loop = Loop()
#     print(f"{loop.id=}")
#     x = makerv(np.random.randn(3, 4))
#     y = makerv(np.random.randn(3, 4))
#     x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop])
#     y_loop = LoopRV(None, vmap_var=y, loop_dims=[0], loops=[loop])
#     z_loop = x_loop + y_loop
#
#     assert x_loop.shape == (4,)
#     assert y_loop.shape == (4,)
#     assert z_loop.shape == (4,)
#
#     z = z_loop.vmap_var
#     assert z.shape == (3, 4)
#
#     print_upstream(z)
