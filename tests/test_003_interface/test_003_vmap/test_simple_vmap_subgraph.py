from pangolin.interface.vmapping import vmap_subgraph, AbstractOp
from pangolin import ir
from pangolin.ir import RV, Constant
from pangolin.interface import InfixRV, makerv


def test_add():
    a = InfixRV(Constant([1, 2, 3]))
    b = InfixRV(Constant([4, 5, 6]))
    a_dummy = InfixRV(AbstractOp())
    b_dummy = InfixRV(AbstractOp())
    c_dummy = InfixRV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], 3)
    assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert c.parents == (a, b)


def test_add_no_size():
    a = InfixRV(Constant([1, 2, 3]))
    b = InfixRV(Constant([4, 5, 6]))
    a_dummy = InfixRV(AbstractOp())
    b_dummy = InfixRV(AbstractOp())
    c_dummy = InfixRV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], None)
    assert c.op == ir.VMap(ir.Add(), (0, 0), None)
    assert c.parents == (a, b)


def test_add_non_abstract():
    a = InfixRV(Constant([1, 2, 3]))
    b = InfixRV(Constant([4, 5, 6]))
    a_dummy = InfixRV(Constant(0))
    b_dummy = InfixRV(Constant(0))
    c_dummy = InfixRV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], 3)
    assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert c.parents == (a, b)


def test_add_with_None():
    a = InfixRV(Constant([1, 2, 3]))
    b = InfixRV(Constant(4))
    a_dummy = InfixRV(AbstractOp())
    b_dummy = InfixRV(AbstractOp())
    c_dummy = InfixRV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, None], 3)
    assert c.op == ir.VMap(ir.Add(), (0, None), 3)
    assert c.parents == (a, b)


def test_normal():
    a = InfixRV(Constant([1, 2, 3]))
    b = InfixRV(Constant([4, 5, 6]))
    a_dummy = InfixRV(AbstractOp())
    b_dummy = InfixRV(AbstractOp())
    c_dummy = InfixRV(ir.Normal(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], 3)
    assert c.op == ir.VMap(ir.Normal(), (0, 0), 3)
    assert c.parents == (a, b)


def test_normal_no_size():
    a = InfixRV(Constant([1, 2, 3]))
    b = InfixRV(Constant([4, 5, 6]))
    a_dummy = InfixRV(AbstractOp())
    b_dummy = InfixRV(AbstractOp())
    c_dummy = InfixRV(ir.Normal(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], None)
    assert c.op == ir.VMap(ir.Normal(), (0, 0), None)
    assert c.parents == (a, b)


def test_normal_all_none():
    a = InfixRV(Constant(1))
    b = InfixRV(Constant(4))
    a_dummy = InfixRV(AbstractOp())
    b_dummy = InfixRV(AbstractOp())
    c_dummy = InfixRV(ir.Normal(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [None, None], 3)
    assert c.op == ir.VMap(ir.Normal(), (None, None), 3)
    assert c.parents == (a, b)


def test_add_all_none():
    a = InfixRV(Constant(1))
    b = InfixRV(Constant(2))
    a_dummy = InfixRV(AbstractOp())
    b_dummy = InfixRV(AbstractOp())
    c_dummy = InfixRV(ir.Add(), a_dummy, b_dummy)
    # should fail because no need to vmap output
    try:
        [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [None, None], 3)
    except ValueError:
        pass


def test_normal_add():
    a = InfixRV(Constant([1, 2, 3]))
    b = InfixRV(Constant([4, 5, 6]))
    a_dummy = InfixRV(AbstractOp())
    b_dummy = InfixRV(AbstractOp())
    c_dummy = InfixRV(ir.Add(), a_dummy, b_dummy)
    d_dummy = InfixRV(ir.Normal(), c_dummy, a_dummy)
    [d] = vmap_subgraph([a_dummy, b_dummy], [c_dummy, d_dummy], [d_dummy], [a, b], [0, 0], 3)
    c = d.parents[0]
    assert c.parents == (a, b)
    assert d.parents == (c, a)
    assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert d.op == ir.VMap(ir.Normal(), (0, 0), 3)


def test_normal_add_nomap():
    a = InfixRV(Constant(1))
    b = InfixRV(Constant(2))
    a_dummy = InfixRV(AbstractOp())
    b_dummy = InfixRV(AbstractOp())
    c_dummy = InfixRV(ir.Add(), a_dummy, b_dummy)
    d_dummy = InfixRV(ir.Normal(), c_dummy, a_dummy)
    [d] = vmap_subgraph([a_dummy, b_dummy], [c_dummy, d_dummy], [d_dummy], [a, b], [None, None], 3)
    c = d.parents[0]
    assert c.parents == (a, b)
    assert d.parents == (c, a)
    assert c.op == ir.Add()
    assert d.op == ir.VMap(ir.Normal(), (None, None), 3)


# def test_simple_add_indices():
#     a = OperatorRV(Constant([1, 2, 3]))
#     b = OperatorRV(Constant([4, 5, 6]))
#     a_dummy = a[0]
#     b_dummy = b[0]
#     c_dummy = RV(ir.Add(), a_dummy, b_dummy)
#     [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], 3)
#     assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
#     assert c.parents == (a, b)
