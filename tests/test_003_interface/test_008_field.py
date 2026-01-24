from pangolin.interface.fields import extract_inputs, vfor
from pangolin import interface as pi
from pangolin import ir


def test_mul_by_two():
    x = pi.constant([1.1, 2.2, 3.3])
    fun = lambda i: x[i] * 2
    arrays, dims, replay = extract_inputs(fun)

    assert arrays == [x]
    assert dims == [[0]]

    out = replay(pi.constant(3.3))
    assert out.op == ir.Mul()
    assert out.parents[0].op == ir.Constant(3.3)
    assert out.parents[1].op == ir.Constant(2)

    y = vfor(fun)
    assert y.op == ir.VMap(ir.Mul(), (0, None), 3)


def test_add_two_indices():
    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    fun = lambda i, j: a[j, i] + 2.2
    arrays, dims, replay = extract_inputs(fun)

    assert arrays == [a]
    assert dims == [[1, 0]]

    out = replay(pi.constant(3.3))
    assert out.op == ir.Add()
    assert out.parents[0].op == ir.Constant(3.3)
    assert out.parents[1].op == ir.Constant(2.2)


def test_varying_dims():
    a = pi.constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    b = pi.constant([7.7, 8.8])
    fun = lambda i: a[i, :] * b[i]
    arrays, dims, replay = extract_inputs(fun)

    assert arrays == [a, b]
    assert dims == [[0], [0]]

    out = replay(pi.constant([1.1, 2.2, 3.3]), pi.constant(4.4))
    assert out.op == ir.VMap(ir.Mul(), [0, None], 3)

    c = vfor(fun)

    assert c.op == ir.VMap(ir.VMap(ir.Mul(), [0, None], 3), [0, 0], 2)
    assert c.parents == (a, b)
