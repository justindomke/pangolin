from pangolin import interface as pi
from pangolin import ir
from pangolin.interface.vfor import extract_from_rvs


def test_simple():
    a = pi.constant([1, 2, 3])
    b = pi.vfor(lambda i: pi.exp(a[i]))

    assert b.op == ir.VMap(ir.Exp(), [0], 3)
    assert b.parents == (a,)


def test_iid():
    a = pi.constant(1.5)
    b = pi.vfor(lambda i: pi.exp(a), i=3)

    assert b.op == ir.VMap(ir.Exp(), [None], 3)


def test_outer():
    a = pi.constant([1, 2, 3])
    b = pi.constant([4, 5])
    c = pi.vfor(lambda i, j: a[i] * b[j])
    assert c.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, None], 3)
    assert c.parents == (a, b)


def test_mul1():
    a = pi.constant([1, 2, 3])
    b = pi.constant(4)
    c = pi.vfor(lambda i: a[i] * b)
    assert c.op == ir.VMap(ir.Mul(), [0, None], 3)
    assert c.parents == (a, b)


def test_mul2():
    a = pi.constant(1)
    b = pi.constant([2, 3, 4])
    c = pi.vfor(lambda i: a * b[i])
    assert c.op == ir.VMap(ir.Mul(), [None, 0], 3)
    assert c.parents == (a, b)


def test_empty():
    a = pi.constant(1.1)
    b = pi.constant(2.2)
    c = pi.vfor(lambda i, j: pi.normal(a, b), i=2, j=3)
    assert c.op == ir.VMap(ir.VMap(ir.Normal(), [None, None], 3), [None, None], 2)
    assert c.parents == (a, b)


def test_logistic_regression():
    X = pi.constant([[10, 0.1], [-5, 0.02], [5, 0.19]])
    z = pi.vmap(pi.normal, None, 2)(0, 1)
    y = pi.vfor(lambda n: pi.bernoulli_logit(X[n, :] @ z))

    assert y.op == ir.VMap(ir.BernoulliLogit(), [0], 3)
    [mul] = y.parents
    assert mul.op == ir.VMap(ir.Matmul(), [0, None], 3)
    assert mul.parents == (X, z)


def test_double():
    x, y = pi.vfor(lambda i: (pi.normal(0, 1), pi.normal(1, 2)), i=3)
    assert x.op == ir.VMap(ir.Normal(), [None, None], 3)
    assert y.op == ir.VMap(ir.Normal(), [None, None], 3)


def test_dependent():
    def myfun(i):
        x = pi.normal(0, 1)
        y = pi.normal(x, 1)
        return x, y

    x, y = pi.vfor(myfun, i=3)
    assert x.op == ir.VMap(ir.Normal(), [None, None], 3)
    assert y.op == ir.VMap(ir.Normal(), [0, None], 3)
    assert y.parents[0] == x


def test_dict():
    def myfun(i):
        x = pi.normal(0, 1)
        y = pi.normal(x, 1)
        return {"alice": x, "bob": y}

    d = pi.vfor(myfun, i=3)
    x = d["alice"]
    y = d["bob"]
    assert x.op == ir.VMap(ir.Normal(), [None, None], 3)
    assert y.op == ir.VMap(ir.Normal(), [0, None], 3)
    assert y.parents[0] == x


def test_trouble1():
    I = 10

    a = pi.vfor(lambda i: pi.normal(0, 10), i=I)
    b = pi.vfor(lambda i: pi.lognormal(0, 1), i=I)
    c = pi.vfor(lambda i: pi.normal(a[i], b[i]), i=I)
    assert c.shape == (I,)


def test_trouble2():
    I = 10
    J = 20

    a = pi.vfor(lambda i: pi.normal(0, 10), i=I)
    b = pi.vfor(lambda j: pi.lognormal(0, 1), j=J)
    c = pi.vfor(lambda i, j: pi.normal(a[i], b[j]), i=I, j=J)
    assert c.shape == (I, J)


def test_trouble3():
    I = 10
    J = 20

    a = pi.vfor(lambda i: pi.normal(0, 10), i=I)
    b = pi.vfor(lambda i, j: pi.lognormal(0, 1), i=I, j=J)
    c = pi.vfor(lambda i, j: pi.normal(a[i], b[i, j]), i=I, j=J)
    assert c.shape == (I, J)


def test_trouble4():
    I = 10
    J = 20

    a = pi.vfor(lambda i, j: pi.normal(0, 10), i=I, j=J)
    b = pi.vfor(lambda i, j: pi.lognormal(0, 1), i=I, j=J)
    c = pi.vfor(lambda i, j: pi.normal(a[i, j], b[i, j]), i=I, j=J)
    assert c.shape == (I, J)


from pangolin.interface.vfor import live_axis  # add to imports


def test_live_axis():
    # identity order: every map ends up at axis 0
    assert live_axis([0, 1, 2], 0) == 0
    assert live_axis([0, 1, 2], 1) == 0
    assert live_axis([0, 1, 2], 2) == 0
    # fully reversed order: axes stay put
    assert live_axis([2, 1, 0], 0) == 2
    assert live_axis([2, 1, 0], 1) == 1
    assert live_axis([2, 1, 0], 2) == 0
    # arbitrary permutation x[j, k, i] -> dims [2, 0, 1]
    assert live_axis([2, 0, 1], 0) == 2
    assert live_axis([2, 0, 1], 1) == 0
    assert live_axis([2, 0, 1], 2) == 0
    # None gaps: only non-None axes ahead of the slot count
    assert live_axis([0, None, 1], 2) == 0
    assert live_axis([None, 0, None], 1) == 0
    assert live_axis([0, None], 1) is None


def test_rect():
    # minimal form of the original bug: one array, two dummies, ascending
    x = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = pi.vfor(lambda i, j: pi.exp(x[i, j]))
    assert y.shape == (2, 3)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), [0], 3), [0], 2)
    assert y.parents == (x,)


def test_transpose():
    # x[j, i] transposes; dummy order determines output axes
    x = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = pi.vfor(lambda i, j: pi.exp(x[j, i]))
    assert y.shape == (3, 2)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), [0], 2), [1], 3)
    assert y.parents == (x,)


def test_dummy_order_determines_axes():
    # same indexing as test_rect, but lambda args swapped -> transposed output
    x = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = pi.vfor(lambda j, i: pi.exp(x[i, j]))
    assert y.shape == (3, 2)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), [0], 2), [1], 3)
    assert y.parents == (x,)


def test_mixed_order():
    # one array ascending, one transposed, in the same vfor
    a = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3), a[i, j]
    b = pi.constant([[1, 2], [3, 4], [5, 6]])  # (3, 2), b[j, i]
    c = pi.vfor(lambda i, j: a[i, j] * b[j, i])
    assert c.shape == (2, 3)
    assert c.op == ir.VMap(ir.VMap(ir.Mul(), [0, 0], 3), [0, 1], 2)
    assert c.parents == (a, b)


def test_identity3():
    x = pi.constant([[[0.0] * 5] * 3] * 2)  # (2, 3, 5), x[i, j, k]
    y = pi.vfor(lambda i, j, k: pi.exp(x[i, j, k]))
    assert y.shape == (2, 3, 5)
    assert y.op == ir.VMap(ir.VMap(ir.VMap(ir.Exp(), [0], 5), [0], 3), [0], 2)
    assert y.parents == (x,)


def test_reversed3():
    x = pi.constant([[[0.0] * 2] * 3] * 5)  # (5, 3, 2), x[k, j, i]
    y = pi.vfor(lambda i, j, k: pi.exp(x[k, j, i]))
    assert y.shape == (2, 3, 5)
    assert y.op == ir.VMap(ir.VMap(ir.VMap(ir.Exp(), [0], 5), [1], 3), [2], 2)
    assert y.parents == (x,)


def test_permuted3():
    x = pi.constant([[[0.0] * 2] * 5] * 3)  # (3, 5, 2), x[j, k, i]
    y = pi.vfor(lambda i, j, k: pi.exp(x[j, k, i]))
    assert y.shape == (2, 3, 5)
    assert y.op == ir.VMap(ir.VMap(ir.VMap(ir.Exp(), [0], 5), [0], 3), [2], 2)
    assert y.parents == (x,)


def test_skip_middle():
    # dummy j indexes only z; x is indexed by i and k -> dims [0, None, 1]
    x = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    z = pi.constant([7, 8, 9, 10])  # (4,)
    y = pi.vfor(lambda i, j, k: x[i, k] * z[j])
    assert y.shape == (2, 4, 3)
    assert y.op == ir.VMap(ir.VMap(ir.VMap(ir.Mul(), [0, None], 3), [None, 0], 4), [0, None], 2)
    assert y.parents == (x, z)


def test_trouble5():
    # trouble3 with the inner array transposed
    I = 10
    J = 20
    a = pi.vfor(lambda i: pi.normal(0, 10), i=I)  # (I,)
    b = pi.vfor(lambda j, i: pi.lognormal(0, 1), i=I, j=J)  # (J, I)
    c = pi.vfor(lambda i, j: pi.normal(a[i], b[j, i]), i=I, j=J)
    assert c.shape == (I, J)
    assert c.op == ir.VMap(ir.VMap(ir.Normal(), [None, 0], J), [0, 1], I)


def test_raw_indexing_control():
    a = pi.constant([0, 1, 2])
    x = pi.vfor(lambda i, j: pi.exp(a[i]), i=3, j=4)
    assert x.op == ir.VMap(ir.VMap(ir.Exp(), [None], 4), [0], 3)
    assert x.parents == (a,)


def test_raw_indexing_trouble():
    x = pi.vfor(lambda i, j: pi.exp(i), i=3, j=4)
    assert x.op == ir.VMap(ir.VMap(ir.Exp(), [None], 4), [0], 3)
    assert x.parents[0].op == ir.Constant([0, 1, 2])

def test_raw_index_simple():
    x = pi.vfor(lambda i: i, i=3)
    assert x.op == pi.Constant([0,1,2])