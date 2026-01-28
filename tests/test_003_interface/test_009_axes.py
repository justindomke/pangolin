from pangolin.interface.axes import axis, Slot
from pangolin import interface as pi
from pangolin import ir
import numpy as np


def test_constant():
    x = Slot()
    with axis(3) as i:
        x[i] = 1.1

    # assert x.shape == (3,)
    # assert x.op == ir.VMap(ir.Identity(), [None], 3)
    # [tmp] = x.parents
    # assert tmp.op == ir.Constant(1.1)

    # x0 = x[0]
    # assert x0.shape == ()
    # assert x0.op == ir.Index()
    # assert x0.parents[0] == x
    # assert x0.parent_ops[1] == ir.Constant(0)

    assert x.shape == (3,)
    assert x.op == ir.VMap(ir.Constant(1.1), [], 3)


def test_mapping():
    w = pi.constant([1.1, 2.2, 3.3])

    x = Slot()
    with axis(3) as i:
        x[i] = pi.exponential(w[i])

    # assert x.shape == (3,)
    # assert x.op == ir.VMap(ir.Identity(), [0], 3)
    # [tmp] = x.parents
    # assert tmp.op == ir.VMap(ir.Exponential(), [0], 3)
    # assert tmp.parents == (w,)

    assert x.shape == (3,)
    assert x.op == ir.VMap(ir.Exponential(), [0], 3)
    assert x.parents == (w,)


def test_outer_product():
    x = pi.constant([1.1, 2.2, 3.3])
    y = pi.constant([4.4, 5.5])

    z = Slot()
    with axis(3) as i:
        with axis(2) as j:
            z[i, j] = x[i] * y[j]

    # assert z.shape == (3, 2)
    # assert z.op == ir.VMap(ir.VMap(ir.Identity(), [0], 2), [0], 3)

    # [tmp] = z.parents
    # assert tmp.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, None], 3)

    assert z.shape == (3, 2)
    assert z.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, None], 3)


def test_iid_normal():
    z = Slot()
    with axis(3) as i:
        z[i] = pi.normal(1.1, 2.2)

    # assert z.shape == (3,)
    # assert z.op == ir.VMap(ir.Identity(), [0], 3)

    # [tmp] = z.parents
    # assert tmp.op == ir.VMap(ir.Normal(), [None, None], 3)
    # assert tmp.parent_ops == (ir.Constant(1.1), ir.Constant(2.2))

    assert z.shape == (3,)
    assert z.op == ir.VMap(ir.Normal(), [None, None], 3)
    assert z.parent_ops == (ir.Constant(1.1), ir.Constant(2.2))


def test_sum_thing():
    a = Slot()
    with axis(3) as i:
        b = Slot()
        with axis(3) as j:
            b[j] = i + j
        a[i] = pi.sum(b, axis=0)
    pi.print_upstream(a)


def test_batch_matrix_mul():
    A = pi.constant(np.random.randn(10, 5, 4))
    B = pi.constant(np.random.randn(10, 4, 3))
    C = Slot()
    with axis(10) as i:
        C[i, :, :] = A[i, :, :] @ B[i, :, :]
    assert C.shape == (10, 5, 3)

    assert C.op == ir.VMap(ir.Matmul(), [0, 0], 10)
    assert C.parents == (A, B)

    pi.print_upstream(C)


def test_trouble():
    x = Slot()
    y = Slot()
    with axis(3) as i:
        tmp = pi.exponential(1)
        x[i] = tmp
        y[i] = tmp
    pi.print_upstream(x=x, y=y)
