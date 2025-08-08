from pangolin.simple_interface.vmapping import vmap_subgraph, AbstractOp
from pangolin import ir
from pangolin.ir import RV, Constant
from pangolin.simple_interface import InfixRV, makerv

def test_add():
    a = RV(Constant([1, 2, 3]))
    b = RV(Constant([4, 5, 6]))
    a_dummy = RV(AbstractOp())
    b_dummy = RV(AbstractOp())
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], 3)
    assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert c.parents == (a, b)

def test_add_no_size():
    a = RV(Constant([1, 2, 3]))
    b = RV(Constant([4, 5, 6]))
    a_dummy = RV(AbstractOp())
    b_dummy = RV(AbstractOp())
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], None)
    assert c.op == ir.VMap(ir.Add(), (0, 0), None)
    assert c.parents == (a, b)

def test_add_non_abstract():
    a = RV(Constant([1, 2, 3]))
    b = RV(Constant([4, 5, 6]))
    a_dummy = RV(Constant(0))
    b_dummy = RV(Constant(0))
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], 3)
    assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert c.parents == (a, b)

def test_add_with_None():
    a = RV(Constant([1, 2, 3]))
    b = RV(Constant(4))
    a_dummy = RV(AbstractOp())
    b_dummy = RV(AbstractOp())
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, None], 3)
    assert c.op == ir.VMap(ir.Add(), (0, None), 3)
    assert c.parents == (a, b)

def test_normal():
    a = RV(Constant([1, 2, 3]))
    b = RV(Constant([4, 5, 6]))
    a_dummy = RV(AbstractOp())
    b_dummy = RV(AbstractOp())
    c_dummy = RV(ir.Normal(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], 3)
    assert c.op == ir.VMap(ir.Normal(), (0, 0), 3)
    assert c.parents == (a, b)

def test_normal_no_size():
    a = RV(Constant([1, 2, 3]))
    b = RV(Constant([4, 5, 6]))
    a_dummy = RV(AbstractOp())
    b_dummy = RV(AbstractOp())
    c_dummy = RV(ir.Normal(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], None)
    assert c.op == ir.VMap(ir.Normal(), (0, 0), None)
    assert c.parents == (a, b)

def test_normal_all_none():
    a = RV(Constant(1))
    b = RV(Constant(4))
    a_dummy = RV(AbstractOp())
    b_dummy = RV(AbstractOp())
    c_dummy = RV(ir.Normal(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [None, None], 3)
    assert c.op == ir.VMap(ir.Normal(), (None, None), 3)
    assert c.parents == (a, b)

def test_add_all_none():
    a = RV(Constant(1))
    b = RV(Constant(2))
    a_dummy = RV(AbstractOp())
    b_dummy = RV(AbstractOp())
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    # should fail because no need to vmap output
    try:
        [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [None, 
        None], 3)
    except ValueError:
        pass

def test_normal_add():
    a = RV(Constant([1,2,3]))
    b = RV(Constant([4,5,6]))
    a_dummy = RV(AbstractOp())
    b_dummy = RV(AbstractOp())
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    d_dummy = RV(ir.Normal(), c_dummy, a_dummy)
    [d] = vmap_subgraph([a_dummy, b_dummy], [c_dummy, d_dummy], [d_dummy], [a, b], [0, 0], 3)
    c = d.parents[0]
    assert c.parents == (a, b)
    assert d.parents == (c, a)
    assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert d.op == ir.VMap(ir.Normal(), (0, 0), 3)
    

def test_normal_add_nomap():
    a = RV(Constant(1))
    b = RV(Constant(2))
    a_dummy = RV(AbstractOp())
    b_dummy = RV(AbstractOp())
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    d_dummy = RV(ir.Normal(), c_dummy, a_dummy)
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


