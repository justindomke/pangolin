from jax._src.core import pp_aval
from jax._src.traceback_util import C
from numpy import extract
from pangolin.interface.axes import (
    # extract_from_rvs,
    # extract_from_rvs_single_axis,
    # vfor,
    # Axis,
    # axis,
    # vmap_axis,
    popout,
    axis,
    axis_debug,
    vmap_axis,
    Slot,
    update_slots,
)
from pangolin import interface as pi
from pangolin import ir

###############################################################################
# popout
###############################################################################


def test_popout1():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis_debug(2)
    xi = x[i, :]

    x_out, where = popout(xi, i)

    assert x_out == x
    assert where == 0


def test_popout2():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis_debug(3)
    xi = x[:, i]

    x_out, where = popout(xi, i)

    assert x_out == x
    assert where == 1


def test_popout3():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis_debug(3)
    xi = x[1, i]

    x_out, where = popout(xi, i)

    assert where == 0
    assert x_out.op == ir.Index()
    assert x_out.parents[0] == x
    assert x_out.parent_ops[1:] == (ir.Constant(1), ir.Constant(range(3)))


def test_popout4():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis_debug(2)
    j = axis_debug(3)
    xij = x[i, j]

    x_out, where = popout(xij, i)

    assert where == 0
    assert x_out.shape == (2,)
    assert x_out.op == ir.Index()
    assert x_out.parents[0] == x
    assert x_out.parent_ops[1] == ir.Constant(range(2))
    assert x_out.parents[2] == j

    x_out, where = popout(xij, j)

    assert where == 0
    assert x_out.shape == (3,)
    assert x_out.op == ir.Index()
    assert x_out.parents[0] == x
    assert x_out.parents[1] == i
    assert x_out.parent_ops[2] == ir.Constant(range(3))


def test_double_popout():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis_debug(2)
    j = axis_debug(3)
    xij = x[i, j]

    xj, where_i = popout(xij, i)

    assert where_i == 0
    assert xj.shape == (2,)
    assert xj.op == ir.Index()
    assert xj.parents[0] == x
    assert xj.parent_ops[1] == ir.Constant(range(2))
    assert xj.parents[2] == j

    x_out, where_j = popout(xj, j)
    assert x_out == x
    assert where_j == 1


def test_double_popout_reverse():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = axis_debug(2)
    j = axis_debug(3)
    xij = x[i, j]

    xi, where_j = popout(xij, j)

    assert where_j == 0
    assert xi.shape == (3,)
    assert xi.op == ir.Index()
    assert xi.parents[0] == x
    assert xi.parents[1] == i
    assert xi.parent_ops[2] == ir.Constant(range(3))

    x_out, where_i = popout(xi, i)
    assert where_i == 0
    assert x_out == x


def test_popout_with_array1():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    j = axis_debug(3)
    yj = x[[1, 1, 0, 0], j]

    assert yj.shape == (4,)

    y, where_j = popout(yj, j)

    assert where_j == 1
    assert y.parents[0] == x
    assert y.parent_ops[1] == ir.Constant([1, 1, 0, 0])
    assert y.parent_ops[2] == ir.Constant(range(3))


def test_popout_with_array2():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    j = axis_debug(3)
    yj = x[[[1, 1, 0, 0], [0, 0, 1, 1]], j]

    assert yj.shape == (2, 4)

    y, where_j = popout(yj, j)

    assert where_j == 2
    assert y.parents[0] == x
    assert y.parent_ops[1] == ir.Constant([[1, 1, 0, 0], [0, 0, 1, 1]])
    assert y.parent_ops[2] == ir.Constant(range(3))


###############################################################################
# extract
###############################################################################

# nothing for now...

###############################################################################
# manual vmap
###############################################################################


def test_context_manager1():
    x = pi.constant([1, 2, 3])
    y = pi.constant([4, 5])
    with axis_debug(3) as i:
        with axis_debug(2) as j:
            xi = x[i]
            yj = y[j]
            zij = xi * yj

        [zi] = vmap_axis([zij], j)
        assert zi.op == ir.VMap(ir.Mul(), [None, 0], 2)
        assert ir.rv_equal(zi.parents[0], xi)
        assert zi.parents[1] == y

    [z] = vmap_axis([zi], i)
    assert z.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, None], 3)
    assert z.parents == (x, y)


def test_context_manager2():
    x = pi.constant([[1, 2, 3], [4, 5, 6]])
    y = pi.constant([7, 8])
    with axis_debug(3) as i:
        with axis_debug(2) as j:
            xji = x[j, i]
            yj = y[j]
            zij = xji * yj

        [zi] = vmap_axis([zij], j)
        assert zi.op == ir.VMap(ir.Mul(), [0, 0], 2)
        assert zi.parents[0].op == ir.Index()
        assert zi.parents[0].parents[0] == x
        assert zi.parents[0].parents[1].op == ir.Constant([0, 1])
        assert zi.parents[0].parents[2] == i

    [z] = vmap_axis([zi], i)
    assert z.op == ir.VMap(ir.VMap(ir.Mul(), [0, 0], 2), [1, None], 3)
    assert z.parents == (x, y)


# def test_context_manager_identity():
#     x = pi.constant([1, 2, 3])
#     y = pi.constant([7, 8])
#     with axis_debug(3) as i:
#         with axis_debug(2) as j:
#             xi = x[i]
#             yj = y[j]
#             tmp = pi.normal(xi, yj)
#             zij = pi.InfixRV(ir.Identity(), tmp)
#             uij = pi.InfixRV(ir.Identity(), tmp)

#         [zi, ui] = vmap_axis([zij, uij], j)
#         assert zi.op == ir.VMap(ir.Identity(), [0], 2)


###############################################################################
# slot basics
###############################################################################


def test_assign_slot1():
    x = Slot()
    with axis_debug(3) as i:
        x[i] = 5


def test_assign_slot2():
    x = Slot()
    with axis_debug(3) as i:
        with axis_debug(5) as j:
            x[i, j] = 5


def test_assign_slot3():
    with axis_debug(3) as i:
        x = Slot()
        with axis_debug(5) as j:
            x[j] = 5


def test_assign_slot4():
    x = Slot()
    with axis_debug(3) as i:
        x[i, :, :] = pi.constant([[1, 2, 3], [4, 5, 6]])


def test_assign_slot5():
    x = Slot()
    with axis_debug(3) as i:
        with axis_debug(5) as j:
            x[i, j, :, :] = pi.constant([[1, 2, 3], [4, 5, 6]])


###############################################################################
# manual update slots
###############################################################################


def test_update_slots_const_outside():
    c = pi.constant(3)

    x = Slot()
    with axis_debug(3) as i:
        x[i] = c

    update_slots([x], i)

    # assert x.shape == (3,)
    # assert x.op == ir.VMap(ir.Identity(), [None], 3)  # None!
    # assert x.parents == (c,)

    assert x.shape == (3,)
    assert x.op == ir.VMap(ir.Constant(3), [None], 3)  # None!
    assert x.parents == ()


def test_update_slots_const_inside():
    x = Slot()
    with axis_debug(3) as i:
        c = pi.constant(5)
        x[i] = c

    update_slots([x], i)

    assert x.shape == (3,)
    assert x.op == ir.VMap(ir.Identity(), [None], 3)  # None!
    assert x.parents[0] != c  # constant is "replayed"
    assert x.parent_ops == (ir.Constant(5),)

    # assert x.shape == (3,)
    # assert x.op == ir.VMap(ir.Constant(5), [], 3)  # None!
    # assert x.parents == ()


def test_update_slots3():
    a = pi.constant(1.1)
    b = pi.constant(2.2)
    x = Slot()
    with axis_debug(3) as i:
        x[i] = pi.normal(a, b)

    update_slots([x], i)

    assert x.shape == (3,)
    assert x.op == ir.VMap(ir.Identity(), [0], 3)
    assert x.parent_ops == (ir.VMap(ir.Normal(), [None, None], 3),)
    assert x.parents[0].parents == (a, b)

    # assert x.shape == (3,)
    # assert x.op == ir.VMap(ir.Normal(), [None, None], 3)
    # assert x.parents == (a, b)


def test_update_slots4():
    x = Slot()
    with axis_debug(3) as i:
        x[i] = pi.normal(1.1, 2.2)

    update_slots([x], i)

    assert x.shape == (3,)
    assert x.op == ir.VMap(ir.Identity(), [0], 3)
    assert x.parent_ops == (ir.VMap(ir.Normal(), [None, None], 3),)
    assert x.parents[0].parent_ops == (ir.Constant(1.1), ir.Constant(2.2))

    # assert x.shape == (3,)
    # assert x.op == ir.VMap(ir.Normal(), [None, None], 3)
    # assert x.parent_ops == (ir.Constant(1.1), ir.Constant(2.2))


def test_update_slots5():
    c = pi.constant(3)

    x = Slot()
    y = Slot()
    with axis_debug(3) as i:
        x[i] = c
        y[i] = c

    update_slots([x, y], i)

    # both x and y point to c

    assert x.shape == (3,)
    assert x.op == ir.VMap(ir.Identity(), [None], 3)  # None!
    assert x.parents == (c,)

    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Identity(), [None], 3)  # None!
    assert y.parents == (c,)

    assert x.parents == y.parents
    assert ir.rv_equal(x, y)


def test_update_slots6():
    a = pi.constant(1.1)
    b = pi.constant(2.2)

    x = Slot()
    y = Slot()
    with axis_debug(3) as i:
        tmp = pi.normal(a, b)
        x[i] = tmp
        y[i] = x[i]

    update_slots([x, y], i)

    # tmp remains scalar
    assert tmp.op == ir.Normal()
    assert tmp.parents == (a, b)

    # TODO: tmp could be accessed without issue.
    # what if someone else accesses it later?
    # - should we "kill" it for safety when we update the slots?
    # - should we maintain a list of "dead" variables in Slots (all upstream)?
    # -  just rely on users not to make this mistake?

    # vmapped version of tmp created during update_slots
    [new_tmp] = x.parents
    assert new_tmp.op == ir.VMap(ir.Normal(), [None, None], 3)
    assert new_tmp.parents == (a, b)

    # both x and y are "copies" of new_tmp
    assert x.shape == y.shape == (3,)
    assert x.op == y.op == ir.VMap(ir.Identity(), [0], 3)  # 0 because tmp inside with
    assert y.parents == x.parents == (new_tmp,)
    assert ir.rv_equal(x, y)


def test_update_slots7():
    a = pi.constant(1.1)
    b = pi.constant(2.2)

    tmp = pi.normal(a, b)

    x = Slot()
    y = Slot()
    with axis_debug(3) as i:
        x[i] = tmp
        y[i] = x[i]

    update_slots([x, y], i)

    # tmp remains scalar
    assert tmp.op == ir.Normal()
    assert tmp.parents == (a, b)

    # both x and y are "resized" copies of tmp
    assert x.shape == y.shape == (3,)
    assert x.op == y.op == ir.VMap(ir.Identity(), [None], 3)  # None because tmp outside with
    assert x.parents == y.parents == (tmp,)
    assert ir.rv_equal(x, y)


def test_update_slots_recursive_const_outside():
    c = pi.constant(3)
    x = Slot()
    with axis_debug(2) as i:
        with axis_debug(3) as j:
            x[i, j] = c

        update_slots([x], j)
        assert not hasattr(x, "op")
    update_slots([x], i)

    assert hasattr(x, "op")
    assert x.shape == (2, 3)
    assert x.op == ir.VMap(ir.VMap(ir.Identity(), [None], 3), [None], 2)
    assert x.parents == (c,)


def test_update_slots_recursive_const_inside():
    x = Slot()
    with axis_debug(2) as i:
        with axis_debug(3) as j:
            c = pi.constant(3)
            x[i, j] = c

        update_slots([x], j)
        assert not hasattr(x, "op")
    update_slots([x], i)

    assert hasattr(x, "op")
    assert x.shape == (2, 3)
    assert x.op == ir.VMap(ir.VMap(ir.Identity(), [None], 3), [None], 2)  # still None
    assert x.parent_ops[0] == ir.Constant(3)


def test_update_slots_recursive_const_middle():
    x = Slot()
    with axis_debug(2) as i:
        c = pi.constant(3)
        with axis_debug(3) as j:
            x[i, j] = c

        update_slots([x], j)
        assert not hasattr(x, "op")

        assert x.value.parents == (c,)
    update_slots([x], i)
    assert hasattr(x, "op")

    assert x.shape == (2, 3)
    assert x.op == ir.VMap(ir.VMap(ir.Identity(), [None], 3), [None], 2)  # still None
    assert x.parent_ops[0] == ir.Constant(3)
    assert x.parents[0] != c


def test_update_slots_recursive_normal_outside():
    a = pi.constant(1.1)
    b = pi.constant(2.2)
    x = Slot()
    with axis_debug(2) as i:
        with axis_debug(3) as j:
            x[i, j] = pi.normal(a, b)

        update_slots([x], j)
        assert not hasattr(x, "op")
    update_slots([x], i)

    assert hasattr(x, "op")
    assert x.shape == (2, 3)
    assert x.op == ir.VMap(ir.VMap(ir.Identity(), [0], 3), [0], 2)

    [tmp] = x.parents
    assert tmp.op == ir.VMap(ir.VMap(ir.Normal(), [None, None], 3), [None, None], 2)
    assert tmp.parents == (a, b)


def test_update_slots_recursive_normal_inside():
    x = Slot()
    with axis_debug(2) as i:
        with axis_debug(3) as j:
            a = pi.constant(1.1)
            b = pi.constant(2.2)
            x[i, j] = pi.normal(a, b)

        update_slots([x], j)
        assert not hasattr(x, "op")
    update_slots([x], i)

    assert hasattr(x, "op")
    assert x.shape == (2, 3)
    assert x.op == ir.VMap(ir.VMap(ir.Identity(), [0], 3), [0], 2)  # same as above

    [tmp] = x.parents
    assert tmp.op == ir.VMap(ir.VMap(ir.Normal(), [None, None], 3), [None, None], 2)  # same as above
    assert tmp.parents[0] != a  # "replayed"
    assert tmp.parents[1] != b  # "replayed"
    assert tmp.parent_ops == (ir.Constant(1.1), ir.Constant(2.2))


def test_update_slots_recursive_normal_mixed():
    a = pi.constant(1.1)
    x = Slot()
    with axis_debug(2) as i:
        with axis_debug(3) as j:
            b = pi.constant(2.2)
            x[i, j] = pi.normal(a, b)

        update_slots([x], j)
        assert not hasattr(x, "op")
    update_slots([x], i)

    assert hasattr(x, "op")
    assert x.shape == (2, 3)
    assert x.op == ir.VMap(ir.VMap(ir.Identity(), [0], 3), [0], 2)  # same as above

    [tmp] = x.parents
    assert tmp.op == ir.VMap(ir.VMap(ir.Normal(), [None, None], 3), [None, None], 2)  # same as above
    assert tmp.parents[0] == a  # not replayed
    assert tmp.parents[1] != b  # replayed
    assert tmp.parent_ops == (ir.Constant(1.1), ir.Constant(2.2))


def test_assign_range():
    x = Slot()
    with axis_debug(3) as i:
        x[i] = i
    update_slots([x], i)

    assert x.op == ir.VMap(ir.Identity(), [0], 3)
    assert x.parents == (i.range,)


def test_assign_exp_range():
    x = Slot()
    with axis_debug(3) as i:
        x[i] = pi.exp(i)
    update_slots([x], i)

    assert x.op == ir.VMap(ir.Identity(), [0], 3)
    [tmp] = x.parents
    assert tmp.op == ir.VMap(ir.Exp(), [0], 3)
    assert tmp.parents == (i.range,)


def test_assign_normal_range():
    x = Slot()
    with axis_debug(3) as i:
        x[i] = pi.normal(i, 5.5)
    update_slots([x], i)

    assert x.op == ir.VMap(ir.Identity(), [0], 3)
    [tmp] = x.parents
    assert tmp.op == ir.VMap(ir.Normal(), [0, None], 3)
    assert tmp.parents[0] == i.range
    assert tmp.parents[1].op == ir.Constant(5.5)
