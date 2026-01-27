from jax._src.core import pp_aval
from numpy import extract
from pangolin.interface.fields import (
    extract_from_rvs,
    extract_from_rvs_single_axis,
    vfor,
    Axis,
    axis,
    vmap_axis,
    popout_axis,
    caxis,
)
from pangolin import interface as pi
from pangolin import ir


def test_popout_axis1():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(2)
    xi = x[i, :]

    x_out, where = popout_axis(xi, i)

    assert x_out == x
    assert where == 0


def test_popout_axis2():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(3)
    xi = x[:, i]

    x_out, where = popout_axis(xi, i)

    assert x_out == x
    assert where == 1


def test_popout_axis3():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(3)
    xi = x[1, i]

    x_out, where = popout_axis(xi, i)

    assert where == 0
    assert x_out.op == ir.Index()
    assert x_out.parents[0] == x
    assert x_out.parent_ops[1:] == (ir.Constant(1), ir.Constant(range(3)))


def test_popout_axis4():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(2)
    j = axis(3)
    xij = x[i, j]

    x_out, where = popout_axis(xij, i)

    assert where == 0
    assert x_out.shape == (2,)
    assert x_out.op == ir.Index()
    assert x_out.parents[0] == x
    assert x_out.parent_ops[1] == ir.Constant(range(2))
    assert x_out.parents[2] == j

    x_out, where = popout_axis(xij, j)

    assert where == 0
    assert x_out.shape == (3,)
    assert x_out.op == ir.Index()
    assert x_out.parents[0] == x
    assert x_out.parents[1] == i
    assert x_out.parent_ops[2] == ir.Constant(range(3))


def test_double_popout():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(2)
    j = axis(3)
    xij = x[i, j]

    xj, where_i = popout_axis(xij, i)

    assert where_i == 0
    assert xj.shape == (2,)
    assert xj.op == ir.Index()
    assert xj.parents[0] == x
    assert xj.parent_ops[1] == ir.Constant(range(2))
    assert xj.parents[2] == j

    x_out, where_j = popout_axis(xj, j)
    assert x_out == x
    assert where_j == 1


def test_double_popout_reverse():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(2)
    j = axis(3)
    xij = x[i, j]

    xi, where_j = popout_axis(xij, j)

    assert where_j == 0
    assert xi.shape == (3,)
    assert xi.op == ir.Index()
    assert xi.parents[0] == x
    assert xi.parents[1] == i
    assert xi.parent_ops[2] == ir.Constant(range(3))

    x_out, where_i = popout_axis(xi, i)
    assert where_i == 0
    assert x_out == x


def test_popout_with_array1():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    j = axis(3)
    yj = x[[1, 1, 0, 0], j]

    assert yj.shape == (4,)

    y, where_j = popout_axis(yj, j)

    assert where_j == 1
    assert y.parents[0] == x
    assert y.parent_ops[1] == ir.Constant([1, 1, 0, 0])
    assert y.parent_ops[2] == ir.Constant(range(3))


def test_popout_with_array2():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    j = axis(3)
    yj = x[[[1, 1, 0, 0], [0, 0, 1, 1]], j]

    assert yj.shape == (2, 4)

    y, where_j = popout_axis(yj, j)

    assert where_j == 2
    assert y.parents[0] == x
    assert y.parent_ops[1] == ir.Constant([[1, 1, 0, 0], [0, 0, 1, 1]])
    assert y.parent_ops[2] == ir.Constant(range(3))


def test_mul_by_two():
    x = pi.constant([1.1, 2.2, 3.3])
    i = axis(3)
    y = x[i] * 2

    arrays, dims, replay = extract_from_rvs([i], y)

    assert arrays == [x]
    assert dims == [[0]]
    out = replay(pi.constant(3.3))
    assert repr(out) == "InfixRV(Mul(), InfixRV(Constant(3.3)), InfixRV(Constant(2)))"

    fun = lambda i: x[i] * 2
    out = vfor(fun)
    assert out.op == ir.VMap(ir.Mul(), (0, None), 3)

    arrays, dims, replay = extract_from_rvs_single_axis(i, y)
    assert arrays == [x]
    assert dims == [0]
    out = replay(pi.constant(3.3))
    assert repr(out) == "InfixRV(Mul(), InfixRV(Constant(3.3)), InfixRV(Constant(2)))"


def test_two_indices():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(3)
    j = axis(2)
    y = x[j, i] + 2.2

    arrays, dims, replay = extract_from_rvs([i, j], y)
    assert arrays == [x]
    assert dims == [[1, 0]]
    out = replay(pi.constant(3.3))
    assert repr(out) == "InfixRV(Add(), InfixRV(Constant(3.3)), InfixRV(Constant(2.2)))"

    fun = lambda i, j: x[j, i] + 2.2
    out = vfor(fun)
    assert out.op == ir.VMap(ir.VMap(ir.Add(), [0, None], 2), [1, None], 3)

    arrays, dims, replay = extract_from_rvs_single_axis(i, y)
    assert arrays[0].op == ir.Index()
    assert arrays[0].parents[0] == x
    assert arrays[0].parents[1] == j
    assert arrays[0].parents[2].op == ir.Constant(range(3))
    assert dims == [0]
    out = replay(pi.constant(3.3))
    assert repr(out) == "InfixRV(Add(), InfixRV(Constant(3.3)), InfixRV(Constant(2.2)))"


def test_same_array_twice():
    a = pi.constant([1, 2, 3])
    i = pi.constant(0)
    y = a[i] * a[i]
    arrays, dims, replay = extract_from_rvs([i], y)
    assert arrays == [a, a]
    assert dims == [[0], [0]]

    out = replay(pi.constant(1.1), pi.constant(1.1))
    assert repr(out) == "InfixRV(Mul(), InfixRV(Constant(1.1)), InfixRV(Constant(1.1)))"

    fun = lambda i: a[i] * a[i]
    y = vfor(fun)
    assert y.op == ir.VMap(ir.Mul(), [0, 0], 3)


def test_outer_product():
    a = pi.constant([1, 2, 3])
    b = pi.constant([4, 5])
    i = pi.constant(0)
    j = pi.constant(0)
    out = a[i] * b[j]
    arrays, dims, replay = extract_from_rvs([i, j], out)

    assert arrays == [a, b]
    assert dims == [[0, None], [None, 0]]
    out = replay(pi.constant(1.1), pi.constant(2.2))
    assert repr(out) == "InfixRV(Mul(), InfixRV(Constant(1.1)), InfixRV(Constant(2.2)))"

    fun = lambda i, j: a[i] * b[j]
    y = vfor(fun)
    assert y.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, None], 3)


def test_varying_dims():
    a = pi.constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    b = pi.constant([7.7, 8.8])
    i = pi.constant(0)
    out = a[i, :] * b[i]
    arrays, dims, replay = extract_from_rvs([i], out)

    assert arrays == [a, b]
    assert dims == [[0], [0]]
    out = replay(pi.constant([1.1, 2.2, 3.3]), pi.constant(7.7))
    assert repr(out) == "InfixRV(VMap(Mul(), [0, None], 3), InfixRV(Constant([1.1,2.2,3.3])), InfixRV(Constant(7.7)))"

    fun = lambda i: a[i, :] * b[i]
    y = vfor(fun)
    assert y.op == ir.VMap(ir.VMap(ir.Mul(), [0, None], 3), [0, 0], 2)


def test_different_dims():
    a = pi.constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    b = pi.constant([[7.7, 8.8], [9.9, 10.10], [11.11, 12.12]])
    i = pi.constant(0)
    out = a[:, i] @ b[i, :]
    arrays, dims, replay = extract_from_rvs([i], out)

    assert arrays == [a, b]
    assert dims == [[1], [0]]
    out = replay(pi.constant(a.op.value[:, 0]), pi.constant(b.op.value[0, :]))
    assert repr(out) == "InfixRV(Matmul(), InfixRV(Constant([1.1,4.4])), InfixRV(Constant([7.7,8.8])))"

    fun = lambda i: a[:, i] @ b[i, :]
    y = vfor(fun)
    assert y.op == ir.VMap(ir.Matmul(), [1, 0], 3)


def test_different_dims_different_indices():
    a = pi.constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    b = pi.constant([[7.7, 8.8], [9.9, 10.10], [11.11, 12.12]])
    i = pi.constant(0)
    j = pi.constant(0)
    out = a[:, i] @ b[j, :]
    arrays, dims, replay = extract_from_rvs([i, j], out)

    assert arrays == [a, b]
    assert dims == [[1, None], [None, 0]]
    out = replay(pi.constant(a.op.value[:, 0]), pi.constant(b.op.value[0, :]))
    assert repr(out) == "InfixRV(Matmul(), InfixRV(Constant([1.1,4.4])), InfixRV(Constant([7.7,8.8])))"

    fun = lambda i, j: a[:, i] @ b[j, :]
    y = vfor(fun)
    assert y.op == ir.VMap(ir.VMap(ir.Matmul(), [None, 0], 3), [1, None], 3)


def test_two_outputs():
    a = pi.constant([1, 2, 3])
    b = pi.constant([4, 5])
    i = pi.constant(0)
    j = pi.constant(0)
    out = [a[i] * b[j], a[j] + b[i]]
    arrays, dims, replay = extract_from_rvs([i, j], out)

    assert arrays == [a, b, a, b]
    assert dims == [[0, None], [None, 0], [None, 0], [0, None]]
    out = replay(pi.constant(1.1), pi.constant(2.2), pi.constant(1.1), pi.constant(2.2))
    assert (
        repr(out)
        == "[InfixRV(Mul(), InfixRV(Constant(1.1)), InfixRV(Constant(2.2))), InfixRV(Add(), InfixRV(Constant(1.1)), InfixRV(Constant(2.2)))]"
    )


def test_vmap_axis_double():
    x = pi.constant([1, 2, 3])
    i = axis(3)
    yi = x[i] * 2
    [y] = vmap_axis([yi], i)

    assert y.op == ir.VMap(ir.Mul(), [0, None], 3)
    assert y.parents[0] == x


def test_vmap_axis_size_clash():
    x = pi.constant([1, 2, 3])
    i = axis(5)
    yi = x[i] * 2
    try:
        [y] = vmap_axis([yi], i)
        assert False
    except ValueError:
        pass


def test_vmap_axis_two_inputs():
    x = pi.constant([1, 2, 3])
    y = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(3)
    xi = x[i]
    yi = y[:, i]
    zi = xi * yi
    [z] = vmap_axis([zi], i)

    assert z.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, 1], 3)
    assert z.parents == (x, y)


def test_vmap_axis_two_inputs_two_outputs():
    x = pi.constant([1, 2, 3])
    y = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(3)
    xi = x[i]
    yi = y[:, i]
    zi = xi * yi
    ui = pi.normal(zi, 1)
    [z, u] = vmap_axis([zi, ui], i)

    assert z.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, 1], 3)
    assert z.parents == (x, y)
    assert u.op == ir.VMap(ir.VMap(ir.Normal(), [0, None], 2), [0, None], 3)
    assert u.parents[0] == z


def test_recursive():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis(2)
    j = axis(3)
    c = pi.constant(7.7)  # must create after axes!
    xij = x[i, j]
    assert xij.shape == ()
    yij = xij * c

    arrays, dims, replay = extract_from_rvs([j], yij)
    assert arrays == [x]
    assert dims == [[1]]
    out = replay(pi.constant(3.3))
    assert out.op == ir.Mul()
    assert out.parent_ops == (ir.Constant(3.3), ir.Constant(7.7))

    [yi] = vmap_axis([yij], j)
    assert yi.op == ir.VMap(ir.Mul(), [0, None], 3)
    assert yi.parents


def test_outer_product_recursive():
    x = pi.constant([1, 2, 3])
    y = pi.constant([4, 5])
    i = axis(3)
    j = axis(2)
    xi = x[i]
    yj = y[j]
    zij = xi * yj
    assert zij.shape == ()

    arrays, dims, replay = extract_from_rvs_single_axis(i, zij)
    assert arrays == [x]
    assert dims == [0]

    a = pi.constant(3.3)
    out = replay(a)
    assert out.op == ir.Mul()
    assert out.parents[0] == a
    assert out.parents[1] == yj

    [zi] = vmap_axis([zij], j)
    assert zi.op == ir.VMap(ir.Mul(), [None, 0], 2)
    assert zi.parents == (xi, y)

    [z] = vmap_axis([zi], i)
    assert z.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, None], 3)
    assert z.parents == (x, y)


def test_context_manager():
    x = pi.constant([1, 2, 3])
    y = pi.constant([4, 5])
    with caxis(3) as i:
        with caxis(2) as j:
            xi = x[i]
            yj = y[j]
            zij = xi * yj

        [zi] = vmap_axis([zij], j)
        assert zi.op == ir.VMap(ir.Mul(), [None, 0], 2)
        assert zi.parents == (xi, y)

    [z] = vmap_axis([zi], i)
    assert z.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, None], 3)
    assert z.parents == (x, y)
