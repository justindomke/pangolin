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
    Loop,
    LoopRV,
    print_upstream,
    add,
    VMapRV,
    fakeloop,
)
from pangolin import dag
import numpy as np  # type: ignore
from pangolin.transforms.normal_normal import normal_normal
from pangolin.transforms.constant_op import constant_op


def test_shapes():
    var = makerv(np.random.randn(3))
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[0], loops=[Loop(3)])
    assert loop_var.shape == ()

    var = makerv(np.random.randn(3, 4))
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[0], loops=[Loop(3)])
    assert loop_var.shape == (4,)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[1], loops=[Loop(4)])
    assert loop_var.shape == (3,)

    var = makerv(np.random.randn(3, 4, 5))
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[0], loops=[Loop(3)])
    assert loop_var.shape == (4, 5)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[1], loops=[Loop(4)])
    assert loop_var.shape == (3, 5)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[2], loops=[Loop(5)])
    assert loop_var.shape == (3, 4)

    loop_a = Loop()
    loop_b = Loop()
    var = makerv(np.random.randn(3, 4, 5))
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 1], loops=[loop_a, loop_b])
    assert loop_var.shape == (5,)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 2], loops=[loop_a, loop_b])
    assert loop_var.shape == (4,)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[1, 2], loops=[loop_a, loop_b])
    assert loop_var.shape == (3,)

    var = makerv(np.random.randn(3, 4, 5, 6))
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 1], loops=[loop_a, loop_b])
    assert loop_var.shape == (5, 6)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 2], loops=[loop_a, loop_b])
    assert loop_var.shape == (4, 6)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[0, 3], loops=[loop_a, loop_b])
    assert loop_var.shape == (4, 5)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[1, 2], loops=[loop_a, loop_b])
    assert loop_var.shape == (3, 6)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[1, 3], loops=[loop_a, loop_b])
    assert loop_var.shape == (3, 5)
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[2, 3], loops=[loop_a, loop_b])
    assert loop_var.shape == (3, 4)


def test_shapes_with_none():
    loop = Loop(10)

    var = makerv(np.random.randn(3))
    loop_var = LoopRV(None, vmap_var=var, loop_dims=[None], loops=[loop])
    assert loop_var.shape == ()


def test_generating1():
    loop = Loop()
    print(f"{loop.id=}")
    x = makerv(np.random.randn(3))
    y = makerv(np.random.randn(3))
    x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop])
    y_loop = LoopRV(None, vmap_var=y, loop_dims=[0], loops=[loop])
    z_loop = x_loop + y_loop

    assert x_loop.shape == ()
    assert y_loop.shape == ()
    assert z_loop.shape == ()

    assert z_loop.loops == [loop]
    assert z_loop.loop_dims == [0]

    z = z_loop.vmap_var
    assert z.shape == (3,)
    assert z.cond_dist == VMapDist(add, [0, 0], None)

    print_upstream(z)


def test_generating2():
    loop1 = Loop()
    loop2 = Loop()
    x = makerv(np.random.randn(3))
    y = makerv(np.random.randn(4))
    x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop1])
    y_loop = LoopRV(None, vmap_var=y, loop_dims=[0], loops=[loop2])
    z_loop = add(x_loop, y_loop)

    assert x_loop.shape == ()
    assert y_loop.shape == ()
    assert z_loop.shape == ()
    assert z_loop.loops == [loop1, loop2]
    assert z_loop.loop_dims == [0, 1]

    z = z_loop.vmap_var

    expected_cond_dist = VMapDist(VMapDist(add, [None, 0]), [0, None])
    assert z.shape == (3, 4)
    assert z.cond_dist == expected_cond_dist


def test_generating3():
    loop1 = Loop()
    loop2 = Loop()
    x = makerv(np.random.randn(3))
    y = makerv(np.random.randn(3, 4))
    x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop1])
    y_loop = LoopRV(None, vmap_var=y, loop_dims=[0, 1], loops=[loop1, loop2])
    z_loop = add(x_loop, y_loop)

    assert x_loop.shape == ()
    assert y_loop.shape == ()
    assert z_loop.shape == ()
    assert z_loop.loops == [loop1, loop2]
    assert z_loop.loop_dims == [0, 1]

    z = z_loop.vmap_var

    expected_cond_dist = VMapDist(VMapDist(add, [None, 0]), [0, 0])
    assert z.shape == (3, 4)
    assert z.cond_dist == expected_cond_dist


def test_generating4():
    loop1 = Loop()
    loop2 = Loop()
    loop3 = Loop()
    x = makerv(np.random.randn(3))
    y = makerv(np.random.randn(4, 5))
    x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop1])
    y_loop = LoopRV(None, vmap_var=y, loop_dims=[0, 1], loops=[loop2, loop3])
    z_loop = add(x_loop, y_loop)

    assert x_loop.shape == ()
    assert y_loop.shape == ()
    assert z_loop.shape == ()
    assert z_loop.loops == [loop1, loop2, loop3]
    assert z_loop.loop_dims == [0, 1, 2]

    z = z_loop.vmap_var

    dist1 = VMapDist(add, [None, 0])
    dist2 = VMapDist(dist1, [None, 0])
    dist3 = VMapDist(dist2, [0, None])
    expected_cond_dist = dist3
    assert z.shape == (3, 4, 5)
    assert z.cond_dist == expected_cond_dist


def test_generating5():
    loop1 = Loop()
    loop2 = Loop()
    x = makerv(np.random.randn(3))
    y = makerv(np.random.randn(4, 3))
    x_loop = LoopRV(None, vmap_var=x, loop_dims=[0], loops=[loop1])
    y_loop = LoopRV(None, vmap_var=y, loop_dims=[0, 1], loops=[loop2, loop1])
    z_loop = add(x_loop, y_loop)

    assert x_loop.shape == ()
    assert y_loop.shape == ()
    assert z_loop.shape == ()
    assert z_loop.loops == [loop1, loop2]
    assert z_loop.loop_dims == [0, 1]

    z = z_loop.vmap_var

    dist1 = VMapDist(add, [None, 0])
    dist2 = VMapDist(dist1, [0, 1])
    expected_cond_dist = dist2
    assert z.shape == (3, 4)
    assert z.cond_dist == expected_cond_dist


def test_generating6():
    loop1 = Loop()
    loop2 = Loop()
    x = makerv(np.random.randn(3, 4, 5))
    y = makerv(np.random.randn(5, 4))
    x_loop = LoopRV(None, vmap_var=x, loop_dims=[2], loops=[loop1])
    y_loop = LoopRV(None, vmap_var=y, loop_dims=[0], loops=[loop1])
    z_loop = x_loop @ y_loop

    assert x_loop.shape == (3, 4)
    assert y_loop.shape == (4,)
    assert z_loop.shape == (3,)
    assert z_loop.loops == [loop1]
    assert z_loop.loop_dims == [0]

    z = z_loop.vmap_var

    assert z.shape == (5, 3)


def test_index_syntax1():
    x = makerv(np.random.randn(3, 4, 5))
    loop = Loop()
    x_loop = x[loop, :, :]
    assert x_loop.shape == (4, 5)
    assert x_loop.loop_dims == [0]
    assert x_loop.loops == [loop]


def test_index_syntax2():
    x = makerv(np.random.randn(3, 4, 5))
    loop1 = Loop()
    loop2 = Loop()
    x_loop = x[loop1, :, loop2]
    assert x_loop.shape == (4,)
    assert x_loop.loop_dims == [0, 2]
    assert x_loop.loops == [loop1, loop2]


def test_index_syntax3():
    x = makerv(np.random.randn(3, 4, 5))
    loop1 = Loop()
    loop2 = Loop()
    x_loop = x[loop2, :, loop1]
    assert x_loop.shape == (4,)
    assert x_loop.loops == [loop2, loop1]
    assert x_loop.loop_dims == [0, 2]
    print(f"{x_loop.vmap_var.shape=}")


def test_index_syntax4():
    x = makerv(np.random.randn(3, 4, 5))
    y = makerv(np.random.randn(5, 6, 3))
    i = Loop()
    x_loop = x[i, :, :]
    y_loop = y[:, :, i]
    assert x_loop.shape == (4, 5)
    assert y_loop.shape == (5, 6)
    z_loop = x_loop @ y_loop
    z = z_loop.vmap_var
    assert z.shape == (3, 4, 6)


def test_index_syntax5():
    x = makerv(np.random.randn(3, 4, 1))
    y = makerv(np.random.randn(3, 1, 5))
    i = Loop()
    x_loop = x[i, :, :]
    y_loop = y[i, :, :]
    assert x_loop.shape == (4, 1)
    assert y_loop.shape == (1, 5)
    z_loop = x_loop @ y_loop
    z = z_loop.vmap_var
    assert z.shape == (3, 4, 5)


def test_assign_syntax1():
    x = makerv(np.random.randn(2, 3))
    i = Loop()
    y = VMapRV()
    y[i, :] = x[i, :]
    print(f"{y=}")


def test_assign_syntax2():
    x = makerv(np.random.randn(2))
    y = makerv(np.random.randn(3))
    i = Loop()
    j = Loop()
    z = VMapRV()
    z[i, j] = x[i] * y[j]
    print(f"{z=}")
    print_upstream(z)
    assert z.shape == (2, 3)


# def test_assign_syntax3():
#     "What if there's no loop var on right?"
#     i = Loop()
#     z = VMapRV()
#     z[i] = normal(0, 1)


# def test_full_syntax1():
#     x = VMapRV()
#     y = VMapRV()
#     for i in fakeloop():
#         x[i] = normal(0,1)
#         for j in fakeloop():


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
