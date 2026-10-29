# pyright: reportUnnecessaryTypeIgnoreComment=error

from pangolin.ir import *
from typing import TYPE_CHECKING, Sequence, assert_type

if TYPE_CHECKING:
    # RV is a concrete graph representation; interfaces and Ops are the extension points.
    class UnsupportedRV(RV):  # type: ignore
        pass


def test_constant():
    op = Constant(0)
    assert_type(op, Constant)
    x = RV(op)
    assert_type(x, RV)


def test_add():
    a = RV(Constant(0))
    b = RV(Constant(1))
    c = RV(Add(), a, b)

    assert_type(a.op, Op)
    assert_type(b.op, Op)
    assert_type(c.op, Op)

    assert_type(a, RV)
    assert_type(b, RV)
    assert_type(c, RV)


def test_vmap():
    op = VMap(Add(), [0, 0], 3)
    assert_type(op, VMap[Add])

    a = RV(Constant([1, 2, 3]))
    b = RV(Constant([2, 3, 4]))
    c = RV(op, a, b)
    assert_type(c, RV)


def test_composite():
    op1 = Composite(1, (Add(),), [[0, 0]])
    assert_type(op1, Composite[Add])

    op2 = Composite(1, (Add(), Normal()), [[0, 0], [0, 1]])
    assert_type(op2, Composite[Normal])

    op3 = Composite(1, [Add(), Normal()], [[0, 0], [0, 1]])
    assert_type(op3, Composite[Op])

    a = RV(Constant(1))
    b = RV(op1, a)
    c = RV(op2, a)

    assert_type(a, RV)
    assert_type(b, RV)
    assert_type(c, RV)


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

    assert_type(a, RV)
    assert_type(b, RV)
    assert_type(c, RV)


def test_composite_sequence(ops: Sequence[Op] = (Normal(),)):
    op = Composite(2, ops, [[0, 1]])
    assert_type(op, Composite[Op])


def test_autoregressive():
    op = Scan(Exp(), 5, in_axes=[])
    assert_type(op, Scan[Exp])


def test_scan_composite():
    op1 = Composite(1, (Add(),), [[0, 0]])

    op2 = Scan(op1, 5, [])
    assert_type(op2, Scan[Composite[Add]])

    op3 = Composite(1, (Add(), Normal()), [[0, 0], [0, 1]])
    assert_type(op3, Composite[Normal])

    op4 = Scan(op3, 5, [])
    assert_type(op4, Scan[Composite[Normal]])

    a = RV(Constant(1))
    b = RV(op2, a)
    c = RV(op4, a)

    assert_type(a, RV)
    assert_type(b, RV)
    assert_type(c, RV)


def test_scan_composite_vmap():
    vmap_add = VMap(Add(), [0, 0], 3)
    vmap_normal = VMap(Normal(), [0, 0], 3)

    op1 = Composite(1, (vmap_add,), [[0, 0]])
    op2 = Scan(op1, 5, [])
    op3 = Composite(1, (vmap_add, vmap_normal), [[0, 0], [0, 1]])
    op4 = Scan(op3, 5, [])

    assert_type(op2, Scan[Composite[VMap[Add]]])
    assert_type(op4, Scan[Composite[VMap[Normal]]])

    a = RV(Constant([1, 2, 3]))
    b = RV(op2, a)
    c = RV(op4, a)

    assert_type(a, RV)
    assert_type(b, RV)
    assert_type(c, RV)


def test_vmap_vmap():
    vmap_add = VMap(Add(), [0, 0], 3)
    vmap_vmap_add = VMap(vmap_add, [0, 0], 2)
    assert_type(vmap_add, VMap[Add])
    assert_type(vmap_vmap_add, VMap[VMap[Add]])


def test_vmap_stack():
    def accepts_generic_vmap(op: VMap[Op]) -> None:
        pass

    def accepts_normal_vmap_stack(op: VMapStack[Normal]) -> None:
        pass

    vectorized_normal = VMap(Normal(), [0, 0], 3)
    matrix_normal = VMap(vectorized_normal, [0, 0], 2)

    accepts_generic_vmap(vectorized_normal)
    accepts_normal_vmap_stack(vectorized_normal)
    accepts_normal_vmap_stack(matrix_normal)


def test_immutable_generic_op_types_are_covariant():
    def accepts_composite(value: Composite[Op]) -> None:
        pass

    def accepts_scan(value: Scan[Op]) -> None:
        pass

    def accepts_transformed(value: Bijected[Op, Bijector]) -> None:
        pass

    composite = Composite(2, (Normal(),), ((0, 1),))
    scan = Scan(Normal(), 2, (None,))
    bijector = Bijector(Exp(), Log(), Composite(2, (Identity(),), ((0,),)))
    transformed = Bijected(Normal(), bijector)

    accepts_composite(composite)
    accepts_scan(scan)
    accepts_transformed(transformed)
