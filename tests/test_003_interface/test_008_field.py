from pangolin.interface.fields import extract_from_rvs, vfor
from pangolin import interface as pi
from pangolin import ir


def test_mul_by_two():
    x = pi.constant([1.1, 2.2, 3.3])
    i = pi.constant(0)
    y = x[i] * 2

    arrays, dims, replay = extract_from_rvs([i], y)

    assert arrays == [x]
    assert dims == [[0]]

    out = replay(pi.constant(3.3))
    assert repr(out) == "InfixRV(Mul(), InfixRV(Constant(3.3)), InfixRV(Constant(2)))"

    fun = lambda i: x[i] * 2
    y = vfor(fun)
    assert y.op == ir.VMap(ir.Mul(), (0, None), 3)


def test_two_indices():
    a = pi.constant([[1, 2, 3], [4, 5, 6]])
    i = pi.constant(0)
    j = pi.constant(0)
    y = a[j, i] + 2.2

    arrays, dims, replay = extract_from_rvs([i, j], y)
    assert arrays == [a]
    assert dims == [[1, 0]]

    out = replay(pi.constant(3.3))
    assert repr(out) == "InfixRV(Add(), InfixRV(Constant(3.3)), InfixRV(Constant(2.2)))"

    fun = lambda i, j: a[j, i] + 2.2
    y = vfor(fun)
    assert y.op == ir.VMap(ir.VMap(ir.Add(), [0, None], 2), [1, None], 3)


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
