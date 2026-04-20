from pangolin import interface as pi
from pangolin import ir


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
