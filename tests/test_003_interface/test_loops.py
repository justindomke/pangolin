from pangolin.interface import (
    RV,
    normal,
    multi_normal,
    makerv,
    exp,
    Constant,
    vmap,
    # viz_upstream,
    # print_upstream,
    add,
)
from pangolin.ir import VMap

import numpy as np  # type: ignore
from pangolin.interface.loops import Loop, SlicedRV, slice_existing_rv, make_sliced_rv, \
    slot
from pangolin import *
from pangolin.interface import loops, OperatorRV, exponential, dirichlet, bernoulli_logit, mul, categorical, sum
from pangolin.interface.index import index_funs


###############################################################################
# Test the Loop context manager
###############################################################################


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


###############################################################################
# test slice_existing_rv
###############################################################################


def test_slice_existing1():
    x = makerv([1, 2, 3])
    loop = Loop()
    x_loop = slice_existing_rv(x, [loop], [loop])

    assert x_loop.shape == ()
    assert x_loop.full_rv == x
    assert x_loop.parents == ()
    assert x_loop.loops == (loop,)
    assert x_loop.loop_axes == (0,)


def test_slice_existing2():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    loop1 = Loop()
    loop2 = Loop()
    x_loop = slice_existing_rv(x, [loop1, loop2], [loop1, loop2])

    assert x_loop.shape == ()
    assert x_loop.full_rv == x
    assert x_loop.parents == ()
    assert x_loop.loops == (loop1, loop2)
    assert x_loop.loop_axes == (0, 1)


def test_slice_existing3():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    loop1 = Loop()
    loop2 = Loop()
    x_loop = slice_existing_rv(x, [slice(None), loop2], [loop1, loop2])

    assert x_loop.shape == (2,)
    assert x_loop.full_rv == x
    assert x_loop.parents == ()
    assert x_loop.loops == (loop2,)
    assert x_loop.loop_axes == (1,)


def test_slice_existing4():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    loop1 = Loop()
    loop2 = Loop()
    x_loop = slice_existing_rv(x, [loop1, slice(None)], [loop1, loop2])

    assert x_loop.shape == (3,)
    assert x_loop.full_rv == x
    assert x_loop.parents == ()
    assert x_loop.loops == (loop1,)
    assert x_loop.loop_axes == (0,)


# def test_slice_existing_fewer_dims():
#     x = makerv([[1, 2, 3], [4, 5, 6]])
#     loop = Loop()
#     x_loop = slice_existing_rv(x, [loop], [loop])
#
#     assert x_loop.shape ==

###############################################################################
# test make_sliced_rv
###############################################################################


def test_make_sliced_rv_no_loops():
    loop = Loop(3)
    loc = makerv(0)
    scale = makerv(1)
    x_slice = make_sliced_rv(ir.Normal(), loc, scale, all_loops=[loop])
    assert isinstance(x_slice, SlicedRV)
    assert x_slice.shape == ()
    assert x_slice.full_rv.shape == (3,)


def test_make_sliced_rv_no_in_axes():
    loc = makerv(0)
    scale = makerv(1)
    loop = Loop(3)
    x_slice = make_sliced_rv(ir.Normal(), loc, scale, all_loops=[loop])
    assert isinstance(x_slice, SlicedRV)
    assert x_slice.full_rv.shape == (3,)
    assert x_slice.full_rv.op == VMap(ir.Normal(), (None, None), 3)
    assert x_slice.loop_axes == (0,)


def test_make_sliced_rv_no_in_axes_implicit():
    loc = makerv(0)
    scale = makerv(1)
    with Loop(3):
        x = normal(loc, scale)
    assert isinstance(x, SlicedRV)
    assert x.full_rv.shape == (3,)
    assert x.full_rv.op == VMap(ir.Normal(), (None, None), 3)
    assert x.loop_axes == (0,)


def test_make_sliced_rv_no_in_axes_multivariate():
    loc = makerv([0, 1])
    cov = makerv([[3, 1], [1, 2]])
    with Loop(3):
        x = multi_normal(loc, cov)
    assert isinstance(x, SlicedRV)
    assert x.full_rv.shape == (3, 2)
    assert x.full_rv.op == VMap(ir.MultiNormal(), (None, None), 3)
    assert x.loop_axes == (0,)


def test_no_in_axes_double_loop():
    loc = makerv(0)
    scale = makerv(1)
    with Loop(3):
        with Loop(5):
            x = normal(loc, scale)
    assert isinstance(x, SlicedRV)
    assert x.full_rv.shape == (3, 5)
    assert x.full_rv.op == VMap(VMap(ir.Normal(), (None, None), 5), (None, None), 3)
    assert x.loop_axes == (0, 1)


###############################################################################
# you can index an RV in a Loop context to implicitly call make_sliced_rv
###############################################################################

# when you enter loop context manager
#


def test_index_fun0():
    """
    Test that we can put a toy function on the index function stack and call it
    """

    def my_index(var, *idx):
        return 7

    index_funs.append(my_index)
    assert index(makerv(5), makerv(0)) == 7
    index_funs.pop()


def test_index_fun1():
    """
    Test that we can put a slightly less toy function on the index function stack and call it
    """
    x = makerv([0, 1, 2])
    loop = Loop()

    def my_index(var, *idx):
        return slice_existing_rv(var, [loop], [loop])

    index_funs.append(my_index)
    x_loop = index(x, loop)
    index_funs.pop()
    assert isinstance(x_loop, SlicedRV)
    assert x_loop.shape == ()
    assert x_loop.full_rv.shape == (3,)


def test_index_fun2():
    """
    Test that we can put a slightly less toy function on the index function stack and call it
    """
    x = makerv([0, 1, 2])
    loop = Loop()

    def my_index(var, *idx):
        return slice_existing_rv(var, [loop], Loop.loops)

    index_funs.append(my_index)
    Loop.loops.append(loop)
    x_loop = index(x, loop)
    assert loop == Loop.loops.pop()
    assert my_index == index_funs.pop()

    assert isinstance(x_loop, SlicedRV)
    assert x_loop.shape == ()
    assert x_loop.full_rv.shape == (3,)


def test_index_fun3():
    x = makerv([0, 1, 2])
    loop = Loop()

    def my_index(var, *idx):
        return slice_existing_rv(var, idx, Loop.loops)

    index_funs.append(my_index)
    Loop.loops.append(loop)
    x_loop = index(x, loop)
    assert loop == Loop.loops.pop()
    assert my_index == index_funs.pop()

    assert isinstance(x_loop, SlicedRV)
    assert x_loop.shape == ()
    assert x_loop.full_rv.shape == (3,)


def test_index_fun4():
    x = makerv([0, 1, 2])
    loop = Loop()

    def my_index(var, *idx):
        return slice_existing_rv(var, idx, Loop.loops)

    index_funs.append(my_index)
    Loop.loops.append(loop)
    x_loop = x[loop]
    assert loop == Loop.loops.pop()
    assert my_index == index_funs.pop()

    assert isinstance(x_loop, SlicedRV)
    assert x_loop.shape == ()
    assert x_loop.full_rv.shape == (3,)


def test_index_fun5():
    x = makerv([0, 1, 2])
    loop = Loop()

    index_funs.append(loops.looped_index)
    Loop.loops.append(loop)
    x_loop = x[loop]
    assert loop == Loop.loops.pop()
    assert loops.looped_index == index_funs.pop()

    assert isinstance(x_loop, SlicedRV)
    assert x_loop.shape == ()
    assert x_loop.full_rv.shape == (3,)


def test_index_fun6():
    x = makerv([0, 1, 2])
    with Loop() as loop:
        x_loop = x[loop]


def test_looped_index0():
    print(f"{index_funs=}")
    print()
    print("creating x")
    x = makerv(np.random.randn(3))
    loop = Loop()
    loops.Loop.loops = [loop]
    x_loop = loops.looped_index(x, loop)
    print(f"{x_loop=}")
    assert isinstance(x_loop, SlicedRV)
    assert loop == loops.Loop.loops.pop()


def test_index_syntax0():
    assert loops.Loop.loops == []
    print()
    print("creating x")
    x = makerv(np.random.randn(3))
    print("entering loop")
    with Loop() as loop:
        x_loop = x[loop]
    print("loop done")
    print(f"{x_loop=}")
    assert isinstance(x_loop, SlicedRV)


def test_index_syntax1():
    assert loops.Loop.loops == []
    x = makerv(np.random.randn(3, 4, 5))
    print("STARTING LOOP")
    with Loop() as loop:
        x_loop = x[loop, :, :]
    print("ENDING LOOP")
    print(f"{x_loop.shape=}")
    assert isinstance(x_loop, SlicedRV)
    assert x_loop.shape == (4, 5)
    assert x_loop.loop_axes == (0,)
    assert x_loop.loops == (loop,)


def test_index_syntax2():
    assert loops.Loop.loops == []
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
    assert loops.Loop.loops == []
    x = makerv(np.random.randn(3, 4, 5))

    with Loop() as loop1:
        with Loop() as loop2:
            x_loop = x[loop2, :, loop1]
    assert x_loop.shape == (4,)
    assert x_loop.loops == (loop1, loop2)
    assert x_loop.loop_axes == (2, 0)


def test_index_syntax4():
    assert loops.Loop.loops == []
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
    assert loops.Loop.loops == []
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


###############################################################################
# test implicitly creating new sliced RVs
###############################################################################


def test_tracing():
    loc = makerv(0)
    scale = makerv(1)
    with Loop(3):
        x = normal(loc, scale)
        y = add(x, x)
    assert x.full_rv.shape == (3,)
    assert x.full_rv.op == VMap(ir.Normal(), (None, None), 3)
    assert x.loop_axes == (0,)
    assert y.full_rv.shape == (3,)
    assert y.full_rv.op == VMap(ir.Add(), (0, 0), 3)
    assert y.loop_axes == (0,)
    assert y.full_rv.parents == (x.full_rv, x.full_rv)


def test_tracing_outside_var():
    loc = makerv(0)
    scale = makerv(1)
    z = normal(0, 1)
    with Loop(3):
        x = normal(loc, scale)
        y = add(x, z)
    assert x.full_rv.shape == (3,)
    assert x.full_rv.op == VMap(ir.Normal(), (None, None), 3)
    assert x.loop_axes == (0,)
    assert y.full_rv.shape == (3,)
    assert y.full_rv.op == VMap(ir.Add(), (0, None), 3)
    assert y.loop_axes == (0,)
    assert y.full_rv.parents == (x.full_rv, z)


###############################################################################
# test vmapRVs and assignment
###############################################################################


def test_assignment():
    loc = makerv(0)
    scale = makerv(1)
    x = slot()
    with Loop(3) as i:
        x[i] = normal(loc, scale)
    assert x.op == VMap(ir.Normal(), (None, None), 3)
    assert x.parents == (loc, scale)
    assert isinstance(x, OperatorRV)

def test_assignment_inline():
    loc = makerv(0)
    scale = makerv(1)
    with Loop(3) as i:
        (x := slot())[i] = normal(loc, scale)
    assert x.op == VMap(ir.Normal(), (None, None), 3)
    assert x.parents == (loc, scale)
    assert isinstance(x, OperatorRV)

    # print_upstream(x)


def test_assignment_casting():
    x = slot()
    with Loop(3) as i:
        x[i] = normal(0, 1)
    assert x.op == VMap(ir.Normal(), (None, None), 3)
    assert x.parents[0].op == Constant(0)
    assert x.parents[1].op == Constant(1)
    assert isinstance(x, OperatorRV)


def test_2d_assignment():
    loc = makerv(0)
    scale = makerv(1)
    x = slot()
    y = slot()
    with Loop(3) as i:
        x[i] = normal(loc, scale)
        with Loop(5) as j:
            y[i, j] = normal(loc, scale)
    assert x.op == VMap(ir.Normal(), (None, None), 3)
    assert x.parents == (loc, scale)
    assert y.op == VMap(VMap(ir.Normal(), (None, None), 5), (None, None), 3)
    assert y.parents == (loc, scale)


def test_2d_assignment_casting():
    x = slot()
    y = slot()
    with Loop(3) as i:
        x[i] = normal(0, 1)
        with Loop(5) as j:
            y[i, j] = normal(0, 1)


def test_rhs_slicing1():
    z = makerv([1, 2, 3])
    with Loop() as i:
        z_loop = z[i]
    assert z_loop.loops == (i,)
    assert z_loop.loop_axes == (0,)


def test_loops_with_full_slicing1():
    z = makerv([1, 2, 3])
    x = slot()
    scale = makerv(1)  # TODO: CURRENTLY NECESSARY!
    with Loop() as i:
        xi = normal(z[i], scale)
        assert isinstance(xi, SlicedRV)
        assert xi.loops == (i,)
        assert xi.loop_axes == (0,)
        x[i] = xi
    assert x.shape == (3,)


def test_loops_with_full_slicing2():
    z = makerv([1, 2, 3])
    x = slot()
    with Loop(3) as i:
        # x[i] = normal_scale(z[i], scale)
        zi = z[i]
        xi = normal(zi, 1)
        assert isinstance(xi, SlicedRV)
        assert xi.loops == (i,)
        assert xi.loop_axes == (0,)
        x[i] = xi
    assert x.shape == (3,)


def test_loops_with_full_slicing3():
    z = makerv([1, 2, 3])
    x = slot()
    with Loop() as i:
        # x[i] = normal_scale(z[i], scale)
        zi = z[i]
        xi = normal(zi, 1)
        assert isinstance(xi, SlicedRV)
        assert xi.loops == (i,)
        assert xi.loop_axes == (0,)
        x[i] = xi
    assert x.shape == (3,)


###############################################################################
# older, possibly redundant assignment syntax tests
###############################################################################


def test_shape_inside_loop_1d():
    assert loops.Loop.loops == []
    x = makerv(np.random.randn(3))
    with Loop() as i:
        assert x[i].shape == ()


def test_shape_inside_loop_2d():
    assert loops.Loop.loops == []
    x = makerv(np.random.randn(3, 4))
    with Loop() as i:
        assert x[i, :].shape == (4,)


def test_shape_inside_loop_2d_implicit():
    assert loops.Loop.loops == []
    x = makerv(np.random.randn(3, 4))
    with Loop() as i:
        assert x[i].shape == (4,)


def test_shape_inside_two_loops_2d():
    assert loops.Loop.loops == []
    x = makerv(np.random.randn(3, 4))
    with Loop() as i:
        with Loop() as j:
            assert x[i, j].shape == ()


def test_assign_syntax1():
    assert loops.Loop.loops == []
    x = makerv(np.random.randn(2, 3))
    y = slot()
    with Loop() as i:
        print(f"assigning to y[i] {loops=}")
        y[i] = x[i, :]
        print("assigning to y[i] DONE {loops=}")
        print(f"{y[i]=}")
        assert y[i].shape == (3,)  # TODO: should work
    # assert isinstance(y,SlicedRV)
    assert y.shape == x.shape


def test_assign_syntax2():
    x = makerv(np.random.randn(2))
    y = makerv(np.random.randn(3))
    z = slot()
    with Loop() as i:
        with Loop() as j:
            z[i, j] = x[i] * y[j]
    print(f"{z=}")
    # print_upstream(z)
    assert z.shape == (2, 3)


def test_assign_syntax3():
    "What if there's no loop var on right?"
    z = slot()
    with Loop(3) as i:
        z[i] = normal(0, 1)
    assert z.shape == (3,)


def test_full_syntax1():
    x = slot()
    y = slot()
    with Loop(3) as i:
        x[i] = normal(0, 1)
        with Loop(4) as j:
            y[i, j] = normal(x[i], 1)
    assert x.shape == (3,)
    assert y.shape == (3, 4)


###############################################################################
# test full functionality
###############################################################################


def test_double_loops1():
    z = makerv([[1, 2], [3, 4], [5, 6]])

    x = slot()
    with Loop(3) as i:
        with Loop(2) as j:
            zij = z[i, j]
            xij = normal(zij, zij)
            x[i, j] = xij
    assert x.shape == (3, 2)


def test_double_loops1_inline():
    z = makerv([[1, 2], [3, 4], [5, 6]])

    with Loop(3) as i:
        with Loop(2) as j:
            zij = z[i, j]
            xij = normal(zij, zij)
            (x := slot())[i, j] = xij
    assert x.shape == (3, 2)


def test_double_loops2():
    x = makerv([7, 8, 9])
    y = makerv([10, 11])

    z = slot()
    with Loop(3) as i:
        with Loop(2) as j:
            z[i, j] = x[i] + y[j]

    assert z.op == VMap(VMap(ir.Add(), (None, 0), 2), (0, None), 3)
    assert z.shape == (3, 2)
    assert z.parents == (x, y)

    # z_samp = sample(z)[0]
    # expected = x.cond_dist.value[:, None] + y.cond_dist.value[None, :]
    # assert np.allclose(z_samp, expected)


def test_double_loops3():
    x = makerv([7, 8, 9])
    y = makerv([[1, 2], [3, 4], [5, 6]])

    z = slot()
    with Loop(3) as i:
        with Loop(2) as j:
            z[i, j] = x[i] + y[i, j]

    assert z.op == VMap(VMap(ir.Add(), (None, 0), 2), (0, 0), 3)
    assert z.shape == (3, 2)
    assert z.parents == (x, y)

    # z_samp = sample(z)[0]
    # expected = x.cond_dist.value[:, None] + y.cond_dist.value
    # assert np.allclose(z_samp, expected)


def test_double_loops4():
    x = makerv([7, 8, 9])
    y = makerv([[1, 2], [3, 4], [5, 6]])

    z = slot()
    u = slot()
    with Loop(3) as i:
        z[i] = x[i] + x[i]
        with Loop(2) as j:
            u[i, j] = z[i] + y[i, j]

    assert z.op == VMap(ir.Add(), (0, 0), 3)
    assert z.shape == (3,)
    assert z.parents == (x, x)

    assert u.op == VMap(VMap(ir.Add(), (None, 0), 2), (0, 0), 3)
    assert u.parents == (z, y)
    assert u.shape == (3, 2)

    # u_samp = sample(u)[0]
    # expected = 2 * x.cond_dist.value[:, None] + y.cond_dist.value
    # assert np.allclose(u_samp, expected)


def test_double_loops5():
    loc = makerv([7, 8, 9])
    scale_z = makerv([10, 11, 12])
    scale_x = makerv([[1, 2], [3, 4], [5, 6]])

    assert scale_x.shape == (3, 2)

    z = slot()
    x = slot()
    with Loop(3) as i:
        z[i] = normal(loc[i], scale_z[i])
        with Loop(2) as j:
            x[i, j] = normal(z[i], scale_x[i, j])


def test_double_loops6():
    loc = makerv([7, 8, 9])
    scale_z = makerv([10, 11, 12])
    scale_x = makerv([[1, 2], [3, 4], [5, 6]])

    assert scale_x.shape == (3, 2)

    z = slot()
    x = slot()
    with Loop() as i:
        z[i] = normal(loc[i], scale_z[i])
        with Loop() as j:
            x[i, j] = normal(z[i], scale_x[i, j])


def test_loop_index_as_constant1():
    z = slot()
    with Loop(5) as i:
        z[i] = exponential(i)
    print(f"{z=}")
    assert z.op == VMap(ir.Exponential(), (0,), 5)
    assert z.parents[0].op == Constant(range(5))


def test_loop_index_as_constant2():
    z = slot()
    with Loop(5) as i:
        z[i] = exponential(1) + i
    assert z.op == VMap(ir.Add(), (0, 0), 5)
    assert z.parents[0].op == VMap(ir.Exponential(), (None,), 5)
    assert z.parents[1].op == Constant(range(5))


def test_loop_index_as_constant3():
    z = slot()
    with Loop(5) as i:
        z[i] = i
    assert z.op == Constant(range(5))


def test_loop_index_as_constant4():
    z = slot()
    with Loop(5) as i:
        z[i] = i + exponential(1)
    assert z.op == VMap(ir.Add(), (0, 0), 5)
    assert z.parents[0].op == Constant(range(5))
    assert z.parents[1].op == VMap(ir.Exponential(), (None,), 5)


def test_loop_index_as_constant5():
    z = slot()
    with Loop(3) as i:
        with Loop(4) as j:
            z[i, j] = i + j
    assert z.op == VMap(VMap(ir.Add(), (None, 0), 4), (0, None), 3)
    assert z.parents[0].op == Constant(range(3))
    assert z.parents[1].op == Constant(range(4))


###############################################################################
# Test some real-ish examples. First up, dirichlet
###############################################################################


def coinflip():
    return bool(np.random.randint(2))


def test_dirichlet():
    K = 5
    M = 6
    N = 7
    V = 8

    # do 10 reps with different versions that should all work
    for reps in range(10):
        alpha = np.ones(K)
        if coinflip():
            alpha = makerv(alpha)
        beta = np.ones(V)
        if coinflip():
            beta = makerv(beta)

        phi = slot()
        theta = slot()
        w = slot()
        z = slot()
        with Loop(K) as k:
            phi[k] = dirichlet(beta)
        with Loop(M) as m:
            theta[m] = dirichlet(alpha)
            with Loop(N) as n:
                if coinflip():
                    z[m, n] = categorical(theta[m])
                else:
                    z[m, n] = categorical(theta[m, :])
                if coinflip():
                    w[m, n] = categorical(phi[z[m, n]])
                else:
                    w[m, n] = categorical(phi[z[m, n], :])

        assert phi.op == VMap(ir.Dirichlet(), (None,), K)
        assert phi.parents[0].op == Constant(np.ones(V))

        assert theta.op == VMap(ir.Dirichlet(), (None,), M)
        assert theta.parents[0].op == Constant(np.ones(K))

        assert z.op == VMap(VMap(ir.Categorical(), (None,), N), (0,), M)
        assert z.parents[0] == theta


###############################################################################
# next, logistic regression
###############################################################################


def test_logistic_regression_1d():
    ndata = 100
    X = makerv(np.random.randn(ndata))
    y_obs = np.random.randint(ndata)
    w = normal(0, 1)

    y = slot()
    with Loop(ndata) as n:
        y[n] = bernoulli_logit(w * X[n])

        # score = sum(vmap(mul)(w,X[n]))


def test_elementwise_mul_in_loop():
    elementwise_mul = vmap(mul)

    w = makerv(np.ones(5))
    X = makerv(np.ones((7, 5)))
    y = slot()
    with Loop(7) as n:
        y[n] = elementwise_mul(w, X[n])
        assert y[n].shape == (5,)
    assert y.shape == (7, 5)
    assert y.op == VMap(VMap(ir.Mul(), (0, 0), 5), (None, 0), 7)


def test_elementwise_mul_in_loop_after_vmap():
    elementwise_mul = vmap(mul)

    w = slot()
    with Loop(5) as i:
        w[i] = normal(0, 1)

    X = makerv(np.ones((7, 5)))
    y = slot()
    with Loop(7) as n:
        y[n] = elementwise_mul(w, X[n])
        assert y[n].shape == (5,)
    assert y.shape == (7, 5)
    assert y.op == VMap(VMap(ir.Mul(), (0, 0), 5), (None, 0), 7)


def check_logistic_regression_outputs(ndims, ndata, w, y):
    assert w.shape == (ndims,)
    assert w.op == VMap(ir.Normal(), (None, None), ndims)
    assert w.parents[0].op == ir.Constant(0)
    assert w.parents[1].op == ir.Constant(1)
    assert y.op == VMap(ir.BernoulliLogit(), (0,), ndata)
    [scores] = y.parents
    if scores.op == VMap(ir.MatMul(), (None, 0), ndata):
        assert scores.parents[0] == w
        assert isinstance(scores.parents[1].op, Constant)
        assert scores.parents[1].shape == (ndata, ndims)
    else:
        assert scores.op == VMap(ir.Sum(0), (0,), ndata)
        [elementwise_scores] = scores.parents
        assert elementwise_scores.op == VMap(VMap(ir.Mul(), (0, 0), ndims), (None, 0), ndata)


def test_logistic_regression_all_variants():
    for reps in range(10):
        ndims = np.random.choice([2, 5, 10])
        ndata = np.random.choice([2, 5, 10])
        print(f"{ndims=}")
        print(f"{ndata=}")

        X = makerv(np.random.randn(ndata, ndims))

        if coinflip():
            w = slot()
            with Loop(ndims) as i:
                w[i] = normal(0, 1)
        else:
            w = vmap(normal, None, ndims)(0, 1)

        y = slot()
        score_elementwise = slot()
        with Loop(ndata) as n:
            if coinflip():
                score = w @ X[n]
            elif coinflip():
                score = sum(vmap(mul)(w, X[n]), axis=0)
            else:
                with Loop(ndims) as i:
                    score_elementwise[n, i] = w[i] * X[n, i]
                score = sum(score_elementwise[n], axis=0)

            y[n] = bernoulli_logit(score)

        check_logistic_regression_outputs(ndims, ndata, w, y)


def test_logistic_regression():
    ndims = 20
    ndata = 100
    X = makerv(np.random.randn(ndata, ndims))

    w = slot()
    with Loop(ndims) as i:
        w[i] = normal(0, 1)

    elementwise_mul = vmap(mul)

    y = slot()
    with Loop(ndata) as n:
        score = sum(elementwise_mul(w, X[n]), axis=0)
        y[n] = bernoulli_logit(score)

    check_logistic_regression_outputs(ndims, ndata, w, y)


def test_logistic_regression_fully_looped():
    ndims = 20
    ndata = 100
    X = makerv(np.random.randn(ndata, ndims))

    w = slot()
    with Loop(ndims) as i:
        w[i] = normal(0, 1)

    y = slot()
    score_elementwise = slot()
    with Loop(ndata) as n:
        with Loop(ndims) as i:
            score_elementwise[n, i] = w[i] * X[n, i]
        score = sum(score_elementwise[n], axis=0)
        y[n] = bernoulli_logit(score)

    check_logistic_regression_outputs(ndims, ndata, w, y)


###############################################################################
# next, heirarchical models
###############################################################################

def test_heirarchical():
    nusers = 5
    ndims = 6
    nobs = 7

    x = makerv(np.random.randn(nusers, nobs, ndims))

    w = slot()
    with Loop(nusers) as i:
        with Loop(ndims) as j:
            w[i, j] = normal(0, 1)

    assert w.op == VMap(VMap(ir.Normal(), (None, None), ndims), (None, None), nusers)

    y = slot()
    with Loop(nusers) as i:
        with Loop(nobs) as k:
            y[i, k] = normal(w[i] @ x[i, k, :], 1)

    assert y.op == VMap(VMap(ir.Normal(), (0, None), nobs), (0, None), nusers)
    scores = y.parents[0]
    assert scores.op == VMap(VMap(ir.MatMul(), (None, 0), nobs), (0, 0), nusers)

###############################################################################
# Test that you can run generated_nodes on a function involving loops
###############################################################################

def test_generated_nodes_with_loops():
    from pangolin.interface.vmap import generated_nodes

    def f(x):
        y = slot()
        with Loop() as i:
            y[i] = x[i]
        return [y]

    x = makerv([1,2,3])

    generated_nodes(f,x)