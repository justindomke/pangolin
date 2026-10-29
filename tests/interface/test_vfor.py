from pangolin import interface as pi, ir


from pangolin import dag
from pangolin.interface import vfor

import pytest

from pangolin.functional.vfor_looping import vfor_simple


def op_contains_index(op: ir.Op) -> bool:
    if isinstance(op, ir.Index):
        return True

    if isinstance(op, ir.VMap):
        return op_contains_index(op.base_op)

    if isinstance(op, ir.Scan):
        return op_contains_index(op.base_op)

    if isinstance(op, ir.Composite):
        return any(op_contains_index(component) for component in op.ops)

    return False


def contains_index(x) -> bool:
    return any(op_contains_index(node.op) for node in dag.upstream_nodes(x))


def test_vfor_simple_iid_bernoulli():
    x = vfor_simple(lambda i: pi.bernoulli(0.5), 5, interface=pi.default)

    assert x.shape == (5,)
    assert x.op.random
    assert isinstance(x.op, ir.VMap)
    assert isinstance(x.op.base_op, ir.Bernoulli)
    assert x.op.axis_size == 5
    assert x.op.in_axes == (None,)


def test_vfor_simple_nested_index_is_simplified():
    data = pi.constant(
        [
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
        ]
    )

    x = vfor_simple(lambda i, j: data[i, j] * 2, 2, 3, interface=pi.default)

    assert not contains_index(x)

    assert isinstance(x.op, ir.VMap)
    assert isinstance(x.op.base_op, ir.VMap)

    # Both transformations map the corresponding leading axis.
    assert x.op.in_axes == (0, None)
    assert x.op.base_op.in_axes == (0, None)

    assert x.parents[0] is data


def test_vfor_simple_two_unused_coordinates():
    x = vfor_simple(lambda i, j: pi.normal(0, 1), 2, 3, interface=pi.default)

    assert x.shape == (2, 3)
    assert x.op.random

    assert isinstance(x.op, ir.VMap)
    assert x.op.axis_size == 2
    assert x.op.in_axes == (None, None)

    assert isinstance(x.op.base_op, ir.VMap)
    assert x.op.base_op.axis_size == 3
    assert x.op.base_op.in_axes == (None, None)


@pytest.mark.parametrize(
    "fun,sizes",
    [
        (lambda i: i + 1, ()),
        (lambda i: i + 1, (2, 3)),
        (lambda i, j: i + j, (2,)),
        (lambda i, j: i + j, (2, 3, 4)),
    ],
)
def test_vfor_simple_rejects_wrong_number_of_sizes(fun, sizes):
    with pytest.raises(
        ValueError,
        match=r"Number of sizes .* not equal to number of args",
    ):
        vfor_simple(fun, *sizes, interface=pi.default)


def test_vfor_simple_rejects_returning_coordinate_unchanged():
    with pytest.raises(
        ValueError,
        match="every output must be created during the function call",
    ):
        vfor_simple(lambda i: i, 3, interface=pi.default)


@pytest.mark.parametrize("size", [0, -1, -10])
def test_vfor_simple_rejects_nonpositive_sizes(size):
    with pytest.raises(ValueError):
        vfor_simple(lambda i: i + 1, size, interface=pi.default)


@pytest.mark.parametrize("size", [True, False, 1.5, "3", None])
def test_vfor_simple_rejects_noninteger_sizes(size):
    with pytest.raises((TypeError, ValueError)):
        vfor_simple(lambda i: i + 1, size, interface=pi.default)


def test_simple():
    a = pi.constant([1, 2, 3])
    b = vfor(lambda i: pi.exp(a[i]), i=3)

    assert b.op == ir.VMap(ir.Exp(), [0], 3)
    assert b.parents == (a,)


def test_iid():
    a = pi.constant(1.5)
    b = vfor(lambda i: pi.exp(a), i=3)

    assert b.op == ir.VMap(ir.Exp(), [None], 3)


def test_vmap_eliminates_identity_index():
    b = pi.constant([4, 5])
    j = pi.constant(range(2))

    x = pi.vmap(lambda j: b[j] * 3)(j)

    assert x.parents[0] == b
    assert x.parents[1].op == ir.Constant(3)
    assert x.op == ir.VMap(
        ir.Mul(),
        [0, None],
        2,
    )


def test_outer():
    a = pi.constant([1, 2, 3])
    b = pi.constant([4, 5])
    c = vfor(lambda i, j: a[i] * b[j], i=3, j=2)
    assert c.op == ir.VMap(ir.VMap(ir.Mul(), [None, 0], 2), [0, None], 3)
    assert c.parents == (a, b)


def test_mul1():
    a = pi.constant([1, 2, 3])
    b = pi.constant(4)
    c = vfor(lambda i: a[i] * b, i=3)
    assert c.op == ir.VMap(ir.Mul(), [0, None], 3)
    assert c.parents == (a, b)


def test_mul2():
    a = pi.constant(1)
    b = pi.constant([2, 3, 4])
    c = vfor(lambda i: a * b[i], i=3)
    assert c.op == ir.VMap(ir.Mul(), [None, 0], 3)
    assert c.parents == (a, b)


def test_empty():
    a = pi.constant(1.1)
    b = pi.constant(2.2)
    c = vfor(lambda i, j: pi.normal(a, b), i=2, j=3)
    assert c.op == ir.VMap(ir.VMap(ir.Normal(), [None, None], 3), [None, None], 2)
    assert c.parents == (a, b)


def test_logistic_regression():
    X = pi.constant([[10, 0.1], [-5, 0.02], [5, 0.19]])
    z = pi.vmap(pi.normal, None, 2)(pi.constant(0), pi.constant(1))
    y = vfor(lambda n: pi.bernoulli_logit(X[n, :] @ z), n=3)

    assert y.op == ir.VMap(ir.BernoulliLogit(), [0], 3)
    [mul] = y.parents
    assert mul.op == ir.VMap(ir.Matmul(), [0, None], 3)
    assert mul.parents == (X, z)


def test_double():
    x, y = vfor(lambda i: (pi.normal(0, 1), pi.normal(1, 2)), i=3)
    assert x.op == ir.VMap(ir.Normal(), [None, None], 3)
    assert y.op == ir.VMap(ir.Normal(), [None, None], 3)


def test_dependent():
    def myfun(i):
        x = pi.normal(0, 1)
        y = pi.normal(x, 1)
        return x, y

    x, y = vfor(myfun, i=3)
    assert x.op == ir.VMap(ir.Normal(), [None, None], 3)
    assert y.op == ir.VMap(ir.Normal(), [0, None], 3)
    assert y.parents[0] == x


def test_dict():
    def myfun(i):
        x = pi.normal(0, 1)
        y = pi.normal(x, 1)
        return {"alice": x, "bob": y}

    d = vfor(myfun, i=3)
    x = d["alice"]
    y = d["bob"]
    assert x.op == ir.VMap(ir.Normal(), [None, None], 3)
    assert y.op == ir.VMap(ir.Normal(), [0, None], 3)
    assert y.parents[0] == x


def test_trouble1():
    I = 10

    a = vfor(lambda i: pi.normal(0, 10), i=I)
    b = vfor(lambda i: pi.lognormal(0, 1), i=I)
    c = vfor(lambda i: pi.normal(a[i], b[i]), i=I)
    assert c.shape == (I,)


def test_trouble2():
    I = 10
    J = 20

    a = vfor(lambda i: pi.normal(0, 10), i=I)
    b = vfor(lambda j: pi.lognormal(0, 1), j=J)
    c = vfor(lambda i, j: pi.normal(a[i], b[j]), i=I, j=J)
    assert c.shape == (I, J)


def test_trouble3():
    I = 10
    J = 20

    a = vfor(lambda i: pi.normal(0, 10), i=I)
    b = vfor(lambda i, j: pi.lognormal(0, 1), i=I, j=J)
    c = vfor(lambda i, j: pi.normal(a[i], b[i, j]), i=I, j=J)
    assert c.shape == (I, J)


def test_trouble4():
    I = 10
    J = 20

    a = vfor(lambda i, j: pi.normal(0, 10), i=I, j=J)
    b = vfor(lambda i, j: pi.lognormal(0, 1), i=I, j=J)
    c = vfor(lambda i, j: pi.normal(a[i, j], b[i, j]), i=I, j=J)
    assert c.shape == (I, J)


def test_rect():
    # minimal form of the original bug: one array, two dummies, ascending
    x = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = vfor(lambda i, j: pi.exp(x[i, j]), i=2, j=3)
    assert y.shape == (2, 3)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), [0], 3), [0], 2)
    assert y.parents == (x,)


def test_transpose():
    # x[j, i] transposes; dummy order determines output axes
    x = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = vfor(lambda i, j: pi.exp(x[j, i]), i=3, j=2)
    assert y.shape == (3, 2)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), [0], 2), [1], 3)
    assert y.parents == (x,)


def test_dummy_order_determines_axes():
    # same indexing as test_rect, but lambda args swapped -> transposed output
    x = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = vfor(lambda j, i: pi.exp(x[i, j]), i=2, j=3)
    assert y.shape == (3, 2)
    assert y.op == ir.VMap(ir.VMap(ir.Exp(), [0], 2), [1], 3)
    assert y.parents == (x,)


def test_mixed_order():
    # one array ascending, one transposed, in the same vfor
    a = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3), a[i, j]
    b = pi.constant([[1, 2], [3, 4], [5, 6]])  # (3, 2), b[j, i]
    c = vfor(lambda i, j: a[i, j] * b[j, i], i=2, j=3)
    assert c.shape == (2, 3)
    assert c.op == ir.VMap(ir.VMap(ir.Mul(), [0, 0], 3), [0, 1], 2)
    assert c.parents == (a, b)


def test_identity3():
    x = pi.constant([[[0.0] * 5] * 3] * 2)  # (2, 3, 5), x[i, j, k]
    y = vfor(lambda i, j, k: pi.exp(x[i, j, k]), i=2, j=3, k=5)
    assert y.shape == (2, 3, 5)
    assert y.op == ir.VMap(ir.VMap(ir.VMap(ir.Exp(), [0], 5), [0], 3), [0], 2)
    assert y.parents == (x,)


def test_skip_middle():
    # dummy j indexes only z; x is indexed by i and k -> dims [0, None, 1]
    x = pi.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    z = pi.constant([7, 8, 9, 10])  # (4,)
    y = vfor(lambda i, j, k: x[i, k] * z[j], i=2, j=4, k=3)
    assert y.shape == (2, 4, 3)
    assert y.op == ir.VMap(ir.VMap(ir.VMap(ir.Mul(), [0, None], 3), [None, 0], 4), [0, None], 2)
    assert y.parents == (x, z)


def test_trouble5():
    # trouble3 with the inner array transposed
    I = 10
    J = 20
    a = vfor(lambda i: pi.normal(0, 10), i=I)  # (I,)
    b = vfor(lambda j, i: pi.lognormal(0, 1), i=I, j=J)  # (J, I)
    c = vfor(lambda i, j: pi.normal(a[i], b[j, i]), i=I, j=J)
    assert c.shape == (I, J)
    assert c.op == ir.VMap(ir.VMap(ir.Normal(), [None, 0], J), [0, 1], I)


def test_raw_indexing_control():
    a = pi.constant([0, 1, 2])
    x = vfor(lambda i, j: pi.exp(a[i]), i=3, j=4)
    assert x.op == ir.VMap(ir.VMap(ir.Exp(), [None], 4), [0], 3)
    assert x.parents == (a,)


def test_raw_indexing_trouble():
    x = vfor(lambda i, j: pi.exp(i), i=3, j=4)
    assert x.op == ir.VMap(ir.VMap(ir.Exp(), [None], 4), [0], 3)
    assert x.parents[0].op == ir.Constant([0, 1, 2])


# def test_raw_index_simple():
#     x = vfor(lambda i: i, i=3)
#     assert x.op == pi.Constant([0, 1, 2])
