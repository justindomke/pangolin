from cleanpangolin.interface.rv import OperatorRV
from cleanpangolin import ir
from cleanpangolin.ir.op import SetAutoRV
from cleanpangolin.ir import makerv, RV
from cleanpangolin import interface


# def test_op_rv():
#     x = makerv(2)
#     with SetAutoRV(OperatorRV):
#         assert ir.op.current_rv == [RV, OperatorRV]
#         y = makerv(3)
#     assert ir.op.current_rv == [RV]
#     assert isinstance(x,RV)
#     assert isinstance(y,OperatorRV)

def test_add_manual():
    with SetAutoRV(OperatorRV):
        x = makerv(2)
        y = makerv(3)
    z = x+y
    assert isinstance(z.op, ir.scalar_ops.Add)

def test_add():
    x = interface.rv.makerv(2)
    y = interface.rv.makerv(3)
    z = x+y
    assert isinstance(z.op, ir.scalar_ops.Add)