from pangolin.ir import *
from typing import assert_type


def test_constant():
    op = Constant(0)
    assert_type(op, Constant)
    x = RV(op)
    assert_type(x, RV[Constant])


def test_add():
    a = RV(Constant(0))
    b = RV(Constant(1))
    c = RV(Add(), a, b)

    assert_type(a.op, Constant)
    assert_type(b.op, Constant)
    assert_type(c.op, Add)

    assert_type(a, RV[Constant])
    assert_type(b, RV[Constant])
    assert_type(c, RV[Add])


def test_vmap():
    op = VMap(Add(), [0, 0], 3)
    assert_type(op, VMap[Add])

    a = RV(Constant([1, 2, 3]))
    b = RV(Constant([2, 3, 4]))
    c = RV(op, a, b)
    assert_type(c, RV[VMap[Add]])


def test_composite():
    op1 = Composite(1, (Add(),), [[0, 0]])
    assert_type(op1, Composite[Add])

    op2 = Composite(1, (Add(), Normal()), [[0, 0], [0, 1]])
    assert_type(op2, Composite[Normal])

    a = RV(Constant(1))
    b = RV(op1, a)
    c = RV(op2, a)

    assert_type(a, RV[Constant])
    assert_type(b, RV[Composite[Add]])
    assert_type(c, RV[Composite[Normal]])


def test_composite_vmap():
    vmap_add = VMap(Add(), [0, 0], 3)
    assert_type(vmap_add, VMap[Add])

    vmap_normal = VMap(Normal(), [0, 0], 3)
    assert_type(vmap_normal, VMap[Normal])

    op1 = Composite(1, (vmap_add,), [[0, 0]])
    assert_type(op1, Composite[VMap[Add]])

    op2 = Composite(1, (vmap_add, vmap_normal), [[0, 0], [0, 1]])
    assert_type(op2, Composite[VMap[Normal]])

    a = RV(Constant([1, 2, 3]))
    b = RV(op1, a)
    c = RV(op2, a)

    assert_type(a, RV[Constant])
    assert_type(b, RV[Composite[VMap[Add]]])
    assert_type(c, RV[Composite[VMap[Normal]]])


def test_autoregressive():
    op = Autoregressive(Exp(), 5, in_axes=[], where_self=0)
    assert_type(op, Autoregressive[Exp])


def test_autoregressive_composite():
    op1 = Composite(1, (Add(),), [[0, 0]])

    op2 = Autoregressive(op1, 5, [])
    assert_type(op2, Autoregressive[Composite[Add]])

    op3 = Composite(1, (Add(), Normal()), [[0, 0], [0, 1]])
    assert_type(op3, Composite[Normal])

    op4 = Autoregressive(op3, 5, [])
    assert_type(op4, Autoregressive[Composite[Normal]])

    a = RV(Constant(1))
    b = RV(op2, a)
    c = RV(op4, a)

    assert_type(a, RV[Constant])
    assert_type(b, RV[Autoregressive[Composite[Add]]])
    assert_type(c, RV[Autoregressive[Composite[Normal]]])


def test_autoregressive_composite_vmap():
    vmap_add = VMap(Add(), [0, 0], 3)
    vmap_normal = VMap(Normal(), [0, 0], 3)

    op1 = Composite(1, (vmap_add,), [[0, 0]])
    op2 = Autoregressive(op1, 5, [])
    op3 = Composite(1, (vmap_add, vmap_normal), [[0, 0], [0, 1]])
    op4 = Autoregressive(op3, 5, [])

    assert_type(op2, Autoregressive[Composite[VMap[Add]]])
    assert_type(op4, Autoregressive[Composite[VMap[Normal]]])

    a = RV(Constant([1, 2, 3]))
    b = RV(op2, a)
    c = RV(op4, a)

    assert_type(a, RV[Constant])
    assert_type(b, RV[Autoregressive[Composite[VMap[Add]]]])
    assert_type(c, RV[Autoregressive[Composite[VMap[Normal]]]])


def test_vmap_vmap():
    vmap_add = VMap(Add(), [0, 0], 3)
    vmap_vmap_add = VMap(vmap_add, [0, 0], 2)
    assert_type(vmap_add, VMap[Add])
    assert_type(vmap_vmap_add, VMap[VMap[Add]])
